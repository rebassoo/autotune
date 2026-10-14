"""
Iterative model-error calibration: optimize -> residuals -> sigma_delta -> re-optimize.

Implements §5 of docs/model_error_residual_calibration.md on top of an
already-trained multi-fidelity surrogate. Each round is an ordinary stage-2
optimization (run_two_stage.py --stage 2 --skip-gp) with
optimize.model_error set, so everything the optimizer normally writes
(checkpoints, results CSV, diagnostics) lands in <out-dir>/round_K/.

    sigma_0                    from --init-frac / the config / --round0-dir
    round k:  theta_k        = argmin cost(theta; sigma_k)
              r_s            = z_s - mu(s, theta_k)                  (all features)
              sigma2_{k+1,b} = argmin_s  block NLL(r; V = sigma2_GP + sigma2_eps + s)
                               per variable x {zonal, regional}
    stop when max|theta_k - theta_{k-1}| < --tol-theta and every sigma moved
    by less than --tol-sigma (relative), or after --max-rounds.

The global block (one feature per variable per snapshot) is never estimated:
its residual is the number the fit drives hardest to zero. Fix it with
--global VAR=SIGMA (physical units, e.g. from published obs uncertainty);
otherwise it keeps its starting value.

Everything is resumable: a finished round is skipped on rerun, and a round
interrupted mid-optimization resumes from its per-start checkpoints. With a
3-level surrogate a 40-start round takes ~4-5 h, so a full calibration
(3-4 rounds) can need more than one 24 h job -- just resubmit.

Usage (after the normal training job has written mf_gp_trained.pkl):

    python scripts/calibrate_model_error.py \\
        --config configs/perlmutter_mf3_ne256_studentt_dof4_annual.yaml \\
        --out-dir /pscratch/sd/r/rebassoo/autotune/calib_mf3_ne256 \\
        --global OSR=2.0 --global OLR=2.0

    # reuse an existing optimization (run without model error) as round 0:
    python scripts/calibrate_model_error.py --config ... --out-dir ... \\
        --round0-dir /pscratch/sd/r/rebassoo/autotune/results_mf3_ne256_studentt_dof4

Outputs in --out-dir:
    round_K/                     optimizer output for round K (config_round.yaml inside)
    calibration_log.json         theta, cost, sigma (norm + physical-equivalent),
                                 in-sample z-score check, per round
    model_error_final.yaml       optimize.model_error block to paste into a config
    config_calibrated.yaml       the input config with that block set
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from autotune_gp.config import load_config                      # noqa: E402
from autotune_gp.io import load_obs, split_zrg_obs              # noqa: E402
from autotune_gp.transforms import transform_obs                # noqa: E402
from autotune_gp.model_error import (                           # noqa: E402
    BLOCKS, block_indices, block_weights, estimate_block_sigma2, extra_variance,
)

EST_BLOCKS = ("zonal", "regional")


def _ts(msg):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
def _load_context(cfg):
    """GP, normalized obs, Y scalers and layout -- what the residual step needs."""
    mf = cfg.multi_fidelity
    if mf is None:
        raise SystemExit("calibrate_model_error.py needs a multi-fidelity config "
                         "(the single-fidelity ESEm surrogate cannot be reloaded).")
    var_names = list(cfg.data.variables)
    pre = Path(cfg.paths.preprocess_dir)
    sc_dir = Path(mf.scalers_dir) if mf.scalers_dir else pre
    with open(sc_dir / "scalers.pkl", "rb") as f:
        saved = pickle.load(f)
    Y_scalers = [saved[f"Y_pipeline_{v}"] for v in var_names]
    obs_parts = split_zrg_obs(load_obs(str(pre / "obs.pkl"))["zrg_obs"],
                              n_vars=len(var_names))
    obs_norm = transform_obs(obs_parts, Y_scalers)            # (1, n_feat, n_vars)

    n_regions = len(cfg.data.regions_list)
    n_snaps = len(cfg.preprocess.snapshots) if (cfg.preprocess and cfg.preprocess.snapshots) else 2
    n_feat = obs_norm.shape[1]
    n_zonal = n_feat // n_snaps - n_regions - 1
    return dict(var_names=var_names, Y_scalers=Y_scalers, obs_norm=obs_norm,
                n_zonal=n_zonal, n_regions=n_regions, n_snaps=n_snaps, n_feat=n_feat)


def _best_from_checkpoints(run_dir: Path):
    files = sorted(glob.glob(str(run_dir / "optimize_checkpoints" / "start_*.pkl")))
    if not files:
        raise SystemExit(f"No optimizer checkpoints in {run_dir / 'optimize_checkpoints'}")
    A = np.array([pickle.load(open(f, "rb")) for f in files], dtype=float)
    i = int(np.argmin(A[:, -1]))
    return A[i, :-1], float(A[i, -1]), len(files)


def _initial_sigma(cfg, ctx, frac, global_phys):
    """Starting model_error: config value if set, else frac (normalized) everywhere."""
    me = copy.deepcopy(cfg.optimize.model_error) if cfg.optimize.model_error else None
    if me is None:
        me = {v: {b: float(frac) for b in BLOCKS} for v in ctx["var_names"]}
    for v, s in global_phys.items():
        me.setdefault(v, {}).pop("global", None)
        me[v]["global_phys"] = float(s)
    return me


def _sigma_phys_equiv(me, ctx):
    """Median physical-unit equivalent of each block's model error, for reading."""
    idx = block_indices(ctx["n_zonal"], ctx["n_regions"], ctx["n_snaps"])
    out = {}
    for j, v in enumerate(ctx["var_names"]):
        scale = np.asarray(ctx["Y_scalers"][j].scale_, dtype=float)
        out[v] = {}
        for b in BLOCKS:
            blk = (me.get(v) or {})
            if b in blk:
                out[v][b] = float(np.median(blk[b] * scale[idx[b]]))
            elif f"{b}_phys" in blk:
                out[v][b] = float(blk[f"{b}_phys"])
    return out


def _run_round(args, raw_cfg, round_dir: Path, model_error, gp_pkl: Path):
    """One stage-2 optimization with the given model_error. Resumable."""
    round_dir.mkdir(parents=True, exist_ok=True)
    rc = copy.deepcopy(raw_cfg)
    rc["optimize"]["model_error"] = model_error
    rc["paths"]["output_dir"] = str(round_dir)
    rc.setdefault("diagnostics", {})
    rc["diagnostics"]["output_dir"] = str(round_dir / "diagnostics")
    if args.no_diagnostics:
        rc["diagnostics"]["enabled"] = False
        rc["diagnostics"]["report_uncertainty"] = False
    cfg_path = round_dir / "config_round.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(rc, f, sort_keys=False)

    link = round_dir / "mf_gp_trained.pkl"
    if not link.exists():
        link.symlink_to(gp_pkl.resolve())

    cmd = [args.python, str(ROOT / "scripts" / "run_two_stage.py"),
           "--config", str(cfg_path), "--stage", "2", "--skip-gp"]
    if args.n_xstarts:
        cmd += ["--n-xstarts", str(args.n_xstarts)]
    _ts(f"  running: {' '.join(cmd)}")
    log = round_dir / "optimize.log"
    with open(log, "a") as fh:
        rcode = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    if rcode != 0:
        raise SystemExit(f"Optimization for {round_dir.name} failed (exit {rcode}); see {log}")


def _residual_step(gp, ctx, theta, model_error, cfg, method):
    """Residuals at theta and the per-block sigma_delta update."""
    m, v = gp.predict(np.asarray(theta, dtype=float))
    m, v = np.asarray(m), np.asarray(v)
    r = (ctx["obs_norm"] - m)[0]                       # (n_feat, n_vars)
    gpv = v[0]
    obs_v = extra_variance(None, cfg.optimize.obs_error, ctx["var_names"], ctx["Y_scalers"],
                           ctx["n_zonal"], ctx["n_regions"], ctx["n_snaps"])
    obs_v = np.zeros_like(gpv) if obs_v is None else obs_v[0]

    idx = block_indices(ctx["n_zonal"], ctx["n_regions"], ctx["n_snaps"])
    new_me = copy.deepcopy(model_error)
    detail = {}
    for j, vname in enumerate(ctx["var_names"]):
        detail[vname] = {}
        for b in EST_BLOCKS:
            f = idx[b]
            w = block_weights(b, cfg.weights.zonal_weights, cfg.weights.regional_weights,
                              ctx["n_snaps"])
            est = estimate_block_sigma2(
                r[f, j], gpv[f, j], obs_v[f, j], weights=w,
                likelihood=cfg.optimize.likelihood, dof=cfg.optimize.student_t_dof,
                var_floor=cfg.optimize.var_floor, kappa=cfg.optimize.variance_penalty_kappa)
            s2 = est[method]
            blk = new_me.setdefault(vname, {})
            blk.pop(f"{b}_phys", None)
            blk[b] = float(np.sqrt(s2))
            ww = np.ones(len(f)) if w is None else w
            ww = ww / ww.sum()
            detail[vname][b] = {
                "sigma_mle": float(np.sqrt(est["mle"])),
                "sigma_moment": float(np.sqrt(est["moment"])),
                "rms_residual": float(np.sqrt(np.sum(ww * r[f, j] ** 2))),
                "rms_gp_sd": float(np.sqrt(np.sum(ww * gpv[f, j]))),
            }
        g = idx["global"]
        detail[vname]["global"] = {"residual": r[g, j].tolist(),
                                   "gp_sd": np.sqrt(gpv[g, j]).tolist()}

    # In-sample z-score check with the *updated* sigma: mean r^2/V per block
    # should be ~1 (it is biased low in-sample; see the README §5). For the
    # Student-t cost V is the squared scale; its variance is V*dof/(dof-2).
    V_new = gpv + extra_variance(new_me, cfg.optimize.obs_error, ctx["var_names"],
                                 ctx["Y_scalers"], ctx["n_zonal"], ctx["n_regions"],
                                 ctx["n_snaps"])[0]
    if cfg.optimize.likelihood == "student_t" and cfg.optimize.student_t_dof > 2:
        V_new = V_new * cfg.optimize.student_t_dof / (cfg.optimize.student_t_dof - 2)
    for j, vname in enumerate(ctx["var_names"]):
        for b in BLOCKS:
            f = idx[b]
            detail[vname][b]["mean_r2_over_V"] = float(np.mean(r[f, j] ** 2 / V_new[f, j]))
    return new_me, detail


def _sigma_change(a, b, abs_floor):
    """Largest change of any sigma, relative to max(|old|, abs_floor).

    The floor keeps a block whose sigma sits near 0 (GP variance already
    explains its residuals) from blocking convergence over a 0.01 -> 0.012 move.
    """
    worst = 0.0
    for v in a:
        for k, x in (a[v] or {}).items():
            y = (b.get(v) or {}).get(k)
            if y is None:
                return float("inf")
            worst = max(worst, abs(x - y) / max(abs(y), abs_floor))
    return worst


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="Stage-2 multi-fidelity config whose paths.output_dir holds "
                         "the trained mf_gp_trained.pkl.")
    ap.add_argument("--out-dir", required=True, help="Directory for rounds and logs.")
    ap.add_argument("--gp-pkl", default=None,
                    help="Trained surrogate (default: <config output_dir>/mf_gp_trained.pkl).")
    ap.add_argument("--round0-dir", default=None,
                    help="Existing optimization output to use as round 0 instead of "
                         "running it. Its model error is taken from --config "
                         "(none = zero), so this is the 'fit without model error first' start.")
    ap.add_argument("--init-frac", type=float, default=0.25,
                    help="Starting sigma_delta in normalized units (fraction of the PPE "
                         "spread at each feature) when the config sets none. Default 0.25.")
    ap.add_argument("--global", dest="global_sigma", action="append", default=[],
                    metavar="VAR=SIGMA",
                    help="Fixed global-mean model error for VAR, physical units. Repeatable.")
    ap.add_argument("--method", choices=["mle", "moment"], default="mle",
                    help="Per-block estimator: likelihood-consistent 1-D MLE (default) or "
                         "Gaussian method of moments.")
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--tol-theta", type=float, default=0.02,
                    help="Convergence: max change of the best point, normalized units.")
    ap.add_argument("--tol-sigma", type=float, default=0.10,
                    help="Convergence: max relative change of any estimated sigma.")
    ap.add_argument("--sigma-floor", type=float, default=0.1,
                    help="Relative sigma changes are measured against max(sigma, this) "
                         "(normalized units), so near-zero sigmas don't block convergence.")
    ap.add_argument("--n-xstarts", type=int, default=None,
                    help="Override optimize.n_xstarts for every round.")
    ap.add_argument("--sensitivity", action="store_true",
                    help="After convergence, re-optimize with sigma_delta x0.5 and x2 "
                         "(README §8) and report how far the best point moves.")
    ap.add_argument("--no-diagnostics", action="store_true",
                    help="Skip per-round diagnostics/uncertainty plots (faster).")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    cfg = load_config(args.config)
    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f)
    if cfg.optimize.likelihood == "mae":
        raise SystemExit("optimize.likelihood is 'mae'; model error needs a likelihood cost.")
    gp_pkl = Path(args.gp_pkl) if args.gp_pkl else Path(cfg.paths.output_dir) / "mf_gp_trained.pkl"
    if not gp_pkl.exists():
        raise SystemExit(f"No trained surrogate at {gp_pkl}; run training first.")
    global_phys = {}
    for item in args.global_sigma:
        k, _, val = item.partition("=")
        if k not in cfg.data.variables or not val:
            raise SystemExit(f"--global {item!r}: expected VAR=SIGMA with VAR in {cfg.data.variables}")
        global_phys[k] = float(val)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "calibration_log.json"
    log = json.load(open(log_path)) if log_path.exists() else {"rounds": []}

    _ts("Loading surrogate and observations")
    ctx = _load_context(cfg)
    with open(gp_pkl, "rb") as f:
        gp = pickle.load(f)

    if args.round0_dir:
        sigma = copy.deepcopy(cfg.optimize.model_error) or {}
    else:
        sigma = _initial_sigma(cfg, ctx, args.init_frac, global_phys)
    if log["rounds"]:
        sigma = log["rounds"][-1]["sigma_next"]
        _ts(f"Resuming after {len(log['rounds'])} finished round(s)")

    missing_global = [v for v in ctx["var_names"]
                      if not any(b.startswith("global") for b in (sigma.get(v) or {}))]
    if missing_global:
        _ts(f"  NOTE: no global-mean model error for {missing_global}; it stays 0 "
            "(it is never estimated). Set it with --global VAR=SIGMA.")

    converged = bool(log.get("converged"))
    k = len(log["rounds"])
    while not converged and k < args.max_rounds:
        _ts(f"=== Round {k} ===")
        if k == 0 and args.round0_dir:
            run_dir = Path(args.round0_dir)
            _ts(f"  using existing optimization {run_dir} as round 0")
        else:
            run_dir = out / f"round_{k}"
            _run_round(args, raw_cfg, run_dir, sigma, gp_pkl)
        theta, cost, n_starts = _best_from_checkpoints(run_dir)

        sigma_next, detail = _residual_step(gp, ctx, theta, sigma, cfg, args.method)
        for v, s in global_phys.items():        # global stays fixed
            sigma_next[v].pop("global", None)
            sigma_next[v]["global_phys"] = s

        entry = {
            "round": k, "run_dir": str(run_dir), "n_starts": n_starts,
            "sigma_used": sigma, "sigma_used_phys_equiv": _sigma_phys_equiv(sigma, ctx),
            "theta": theta.tolist(), "cost": cost,
            "sigma_next": sigma_next, "sigma_next_phys_equiv": _sigma_phys_equiv(sigma_next, ctx),
            "blocks": detail,
        }
        if log["rounds"]:
            prev = np.asarray(log["rounds"][-1]["theta"])
            entry["dtheta_max"] = float(np.max(np.abs(theta - prev)))
            entry["dsigma_rel_max"] = _sigma_change(sigma_next, sigma, args.sigma_floor)
            converged = (entry["dtheta_max"] < args.tol_theta
                         and entry["dsigma_rel_max"] < args.tol_sigma)
        log["rounds"].append(entry)
        log["converged"] = converged
        json.dump(log, open(log_path, "w"), indent=1)

        _ts(f"  best cost {cost:.5f} over {n_starts} starts"
            + (f"; max|dtheta| {entry['dtheta_max']:.4f}, max rel dsigma "
               f"{entry['dsigma_rel_max']:.3f}" if "dtheta_max" in entry else ""))
        _ts(f"  {'var':6s} {'block':9s} {'sigma used':>10s} {'sigma new':>10s} "
            f"{'moment':>8s} {'rms r':>8s} {'rms gp sd':>9s} {'r2/V':>6s}  (normalized; r2/V ~1 = calibrated)")
        for v in ctx["var_names"]:
            for b in EST_BLOCKS:
                d = detail[v][b]
                _ts(f"  {v:6s} {b:9s} {sigma.get(v, {}).get(b, float('nan')):10.3f} "
                    f"{sigma_next[v][b]:10.3f} {d['sigma_moment']:8.3f} {d['rms_residual']:8.3f} "
                    f"{d['rms_gp_sd']:9.3f} {d['mean_r2_over_V']:6.2f}")
        sigma = sigma_next
        k += 1

    if converged:
        _ts(f"Converged after {k} round(s).")
    else:
        _ts(f"Not converged after {k} round(s) (max {args.max_rounds}); "
            "the last sigma is still written out.")
    # The sigma to use from now on is the last one the optimizer actually ran
    # with when converged (theta is consistent with it); otherwise the newest.
    final = log["rounds"][-1]["sigma_used" if converged else "sigma_next"]
    with open(out / "model_error_final.yaml", "w") as f:
        yaml.safe_dump({"model_error": final}, f, sort_keys=False)
    cal = copy.deepcopy(raw_cfg)
    cal["optimize"]["model_error"] = final
    with open(out / "config_calibrated.yaml", "w") as f:
        yaml.safe_dump(cal, f, sort_keys=False)
    _ts(f"Wrote {out / 'model_error_final.yaml'} and {out / 'config_calibrated.yaml'}")

    if args.sensitivity and converged:
        base_theta = np.asarray(log["rounds"][-1]["theta"])
        sens = log.get("sensitivity", {})
        for fac in (0.5, 2.0):
            key = f"x{fac:g}"
            if key in sens:
                continue
            scaled = {v: {b: (s * fac if not b.startswith("global") else s)
                          for b, s in blocks.items()} for v, blocks in final.items()}
            run_dir = out / f"sensitivity_{key}"
            _ts(f"=== Sensitivity: sigma_delta {key} (global held fixed) ===")
            _run_round(args, raw_cfg, run_dir, scaled, gp_pkl)
            th, c, _ = _best_from_checkpoints(run_dir)
            sens[key] = {"theta": th.tolist(), "cost": c,
                         "dtheta_max": float(np.max(np.abs(th - base_theta))),
                         "dtheta_rms": float(np.sqrt(np.mean((th - base_theta) ** 2)))}
            _ts(f"  {key}: max|dtheta| {sens[key]['dtheta_max']:.3f}, "
                f"rms {sens[key]['dtheta_rms']:.3f} (normalized)")
            log["sensitivity"] = sens
            json.dump(log, open(log_path, "w"), indent=1)


if __name__ == "__main__":
    main()
