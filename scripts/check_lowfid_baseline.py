"""
Why does the 2-level (ne128+ne32) GP predict the ne256 runs better than the ne128
simulations at the same parameters do? Loads the data exactly as
evaluate_nfidelity_holdout.py does and, per variable, on the ne256 members with an
ne128 run of the same member index:

  - checks that the matched ne128 and ne256 runs really have the same parameters
  - R^2 vs ne256 of: the 2-level GP, the ne128 simulation, and the ne128 simulation
    with its member-mean bias (per feature) removed
  - R^2 of the 2-level GP against the ne128 simulations it was trained on (how much
    the GP smooths the ne128 runs)
  - MSE split into the member-mean bias (systematic ne128 - ne256 offset per feature)
    and the remainder

Needs a compute node (loads the 2-level GP).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_nfidelity_holdout import _load_level, _member_index   # noqa: E402
from autotune_gp.gp_multifidelity import align_Y_to_hf_layout                   # noqa: E402

P = Path("/global/u2/r/rebassoo/work/2026_09_10_Autotuning-Repo/mf_prod_v2")
GP = "/pscratch/sd/r/rebassoo/autotune/results_mf_ne128_ne32_prod_v2/mf_gp_trained.pkl"


def r2(t, p):
    return r2_score(t, p, multioutput="variance_weighted")


def main():
    lower = [_load_level(str(P / "preprocess_output_ne32_prod_v2")),
             _load_level(str(P / "preprocess_output_ne128_prod_v2"))]
    top = _load_level(str(P / "preprocess_output_ne256_n85"))
    saved = pickle.load(open(P / "preprocess_output_ne128_prod_v2" / "scalers.pkl", "rb"))
    var_names = [k.replace("Y_pipeline_", "") for k in saved if k.startswith("Y_pipeline_")]
    ysc = [saved[f"Y_pipeline_{v}"] for v in var_names]
    xsc = saved["X_pipeline"]
    ne128 = lower[-1]
    Y128_native = ne128["Y"]
    if ne128["Y"].shape[1] != top["Y"].shape[1]:
        ne128["Y"] = align_Y_to_hf_layout(ne128["Y"], ne128["mask"], top["mask"])
    norm = lambda Y: np.stack([ysc[j].transform(Y[:, :, j]) for j in range(len(var_names))], axis=2)
    Yt, Y1 = norm(top["Y"]), norm(ne128["Y"])
    Xt, X1 = xsc.transform(top["X"]), xsc.transform(ne128["X"])

    low_idx = {_member_index(n): i for i, n in enumerate(ne128["names"])}
    pairs = [(i, low_idx[_member_index(n)]) for i, n in enumerate(top["names"])
             if _member_index(n) in low_idx]
    ti, li = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    dx = np.abs(Xt[ti] - X1[li]).max(axis=1)
    print(f"{len(pairs)} matched members; max |x_ne256 - x_ne128| over parameters: "
          f"median {np.median(dx):.2e}, max {dx.max():.2e}; "
          f"{(dx > 1e-6).sum()} pairs with different parameters")

    gp = pickle.load(open(GP, "rb"))
    m_top, _ = gp.predict_batch(Xt)
    m_low, _ = gp.predict_batch(X1[li])
    pos = {c: i for i, c in enumerate(ne128["mask"]["valid_feat_indices"])}
    order = [pos[c] for c in top["mask"]["valid_feat_indices"]]
    m_top, m_low = np.asarray(m_top)[:, order, :], np.asarray(m_low)[:, order, :]

    print(f"\n{'':6s} {'GP->ne256':>10s} {'GP->ne256':>10s} {'ne128->ne256':>13s} {'ne128 debiased':>15s}"
          f" {'GP->ne128':>10s} {'bias share':>11s} {'bias share':>11s}")
    print(f"{'':6s} {'(all 85)':>10s} {'(matched)':>10s} {'(matched)':>13s} {'->ne256':>15s}"
          f" {'(matched)':>10s} {'ne128 MSE':>11s} {'GP MSE':>11s}")
    for j, v in enumerate(var_names):
        t = Yt[ti, :, j]
        s = Y1[li, :, j]
        g = m_top[ti, :, j]
        bias_s = (s - t).mean(axis=0)
        bias_g = (g - t).mean(axis=0)
        mse_s, mse_g = np.mean((s - t) ** 2), np.mean((g - t) ** 2)
        print(f"{v:6s} {r2(Yt[:, :, j], m_top[:, :, j]):10.3f} {r2(t, g):10.3f} {r2(t, s):13.3f} "
              f"{r2(t, s - bias_s):15.3f} {r2(s, m_low[:, :, j]):10.3f} "
              f"{np.mean(bias_s ** 2) / mse_s:11.2f} {np.mean(bias_g ** 2) / mse_g:11.2f}")
    print("\nbias share = fraction of the MSE that is the member-mean offset per feature "
          "(the same for every member, i.e. a resolution bias).")


if __name__ == "__main__":
    main()
