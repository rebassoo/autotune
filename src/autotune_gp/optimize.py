from __future__ import annotations

import contextlib
import csv
import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import basinhopping

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:                                   # pragma: no cover
    _threadpool_limits = None


# ---------------------------------------------------------------------------
# Worker plumbing
# ---------------------------------------------------------------------------
# Handoff to forked workers.
#
# ProcessPoolExecutor pickles whatever callable you submit, and cost_fn closes
# over the trained GP — 7.7 GB resident for the multi-fidelity AR1 model.
# Pickling that to every worker is not viable, and `initializer=`/`initargs=`
# would pickle it too. So the parent fills this dict *before* the pool exists;
# with the 'fork' start method each child inherits it copy-on-write and only
# (index, x0, seed) ever crosses the process boundary.
#
# The thread executor reads the same dict from the same process, so both
# executors share one code path.
_WORKER: dict = {}


def _blas_ctx(limit):
    """Cap BLAS threads inside a worker (no-op if threadpoolctl is missing)."""
    if limit and _threadpool_limits is not None:
        return _threadpool_limits(limits=limit)
    return contextlib.nullcontext()


def _run_one_start(args):
    """Run one basinhopping start. Module-level so it pickles by name only."""
    idx, x0, start_seed = args
    with _blas_ctx(_WORKER.get("blas_limit")):
        res = basinhopping(
            _WORKER["cost_fn"], x0,
            minimizer_kwargs=_WORKER["minimizer_kwargs"],
            niter=_WORKER["niter"],
            seed=int(start_seed),
        )
    return idx, np.hstack((res.x, res.fun))


def _make_executor(kind: str, max_workers):
    if kind == "process":
        # 'fork' is what makes the copy-on-write handoff above work; 'spawn'
        # would start a fresh interpreter that has never seen _WORKER.
        return ProcessPoolExecutor(max_workers=max_workers,
                                   mp_context=mp.get_context("fork"))
    if kind == "thread":
        return ThreadPoolExecutor(max_workers=max_workers)
    raise ValueError(f"Unknown executor {kind!r} (expected 'thread' or 'process')")


# ---------------------------------------------------------------------------
# Per-start checkpointing
# ---------------------------------------------------------------------------

def _run_signature(seed, n_xstarts, n_params, lo, hi, niter, method) -> dict:
    """Everything that changes what a start means. Checkpoints written by a run
    with a different signature must not be reused."""
    return {
        "seed":        int(seed),
        "n_xstarts":   int(n_xstarts),
        "n_params":    int(n_params),
        "bounds_low":  np.asarray(lo, dtype=float).tolist(),
        "bounds_high": np.asarray(hi, dtype=float).tolist(),
        "niter":       int(niter),
        "method":      str(method),
    }


def _load_checkpoints(ckpt_dir: Path, sig: dict, results: np.ndarray,
                      n_xstarts: int) -> set:
    """Fill `results` from any valid checkpoints; return the set of done indices."""
    man_path = ckpt_dir / "manifest.pkl"
    stale = False
    if man_path.exists():
        try:
            with open(man_path, "rb") as f:
                stale = pickle.load(f) != sig
        except Exception:
            stale = True
        if stale:
            print("  Warning: existing optimize checkpoints came from a different "
                  "configuration (seed/bounds/n_xstarts/niter/method) — ignoring "
                  "them and re-running every start.")
    with open(man_path, "wb") as f:
        pickle.dump(sig, f)
    if stale:
        return set()

    done = set()
    for i in range(n_xstarts):
        p = ckpt_dir / f"start_{i:04d}.pkl"
        if not p.exists():
            continue
        try:
            with open(p, "rb") as f:
                results[i] = pickle.load(f)
            done.add(i)
        except Exception as e:
            print(f"  Warning: unreadable checkpoint {p} ({e}); re-running start {i}.")
    return done


def _save_checkpoint(ckpt_dir: Path, idx: int, row: np.ndarray):
    # Write-then-rename so a kill mid-write cannot leave a truncated file that
    # would be silently trusted on resume.
    tmp = ckpt_dir / f"start_{idx:04d}.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(row, f)
    tmp.replace(ckpt_dir / f"start_{idx:04d}.pkl")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def optimize_parallel(cost_fn, n_params, bounds_low, bounds_high, seed, n_xstarts, niter,
                      method, out_dir, max_workers=None, executor="thread",
                      checkpoint_dir=None, blas_limit=None):
    """Multi-start basinhopping over cost_fn.

    executor       — 'thread' (default) or 'process'. Use 'process' for the
                     multi-fidelity GP: GPy keeps kernel slice state on the
                     shared model object (_sliced_X), so concurrent predict from
                     threads races and raises an input-dim assertion. Processes
                     each get their own copy. The single-fidelity path must stay
                     on 'thread' — it is ESEm/GPflow, and TensorFlow does not
                     survive fork.
    checkpoint_dir — if set, each start's result is written as it completes and
                     reloaded on a later run, so a walltime kill or a dead
                     worker costs only the starts still in flight.
    blas_limit     — BLAS threads per worker. Defaults to 1 for the process
                     executor: N workers x the job's OMP_NUM_THREADS would badly
                     oversubscribe the node, and BLAS threading measured as
                     worthless here anyway (232 vs 238 ms/cost-eval) because the
                     per-call matrices are tiny.
    """
    if blas_limit is None and executor == "process":
        blas_limit = 1

    rn = np.random.RandomState(seed)

    lo = np.broadcast_to(np.asarray(bounds_low,  dtype=float), (n_params,))
    hi = np.broadcast_to(np.asarray(bounds_high, dtype=float), (n_params,))

    # Draw starts inside the actual per-parameter box. A plain rand() in [0,1)
    # would put most starts outside a narrowed box, where L-BFGS-B just clips
    # them onto the boundary and every start begins from the same face.
    xstarts = lo + rn.rand(n_xstarts, n_params) * (hi - lo)

    # Each start needs its own RNG stream — reusing one `seed` across all
    # basinhopping calls makes every start follow the identical step/accept
    # sequence, so they only differ by x0. Derive one seed per start from
    # the top-level seed for reproducibility.
    start_seeds = rn.randint(0, 2**31 - 1, size=n_xstarts)

    bounds = list(zip(lo.tolist(), hi.tolist()))
    minimizer_kwargs = {"method": method, "bounds": bounds}

    # Indexed assignment below keeps this deterministic: a start's row depends
    # only on its own x0 and seed, never on completion order or worker count.
    results = np.full((n_xstarts, n_params + 1), np.nan, dtype=float)

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None
    done: set = set()
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        sig = _run_signature(seed, n_xstarts, n_params, lo, hi, niter, method)
        done = _load_checkpoints(ckpt_dir, sig, results, n_xstarts)

    pending = [i for i in range(n_xstarts) if i not in done]
    if done:
        print(f"  Resuming: {len(done)}/{n_xstarts} starts already checkpointed, "
              f"{len(pending)} to run.", flush=True)

    if pending:
        _WORKER.clear()
        _WORKER.update(cost_fn=cost_fn, minimizer_kwargs=minimizer_kwargs,
                       niter=niter, blas_limit=blas_limit)

        print(f"  Running {len(pending)} start(s) with the {executor} executor "
              f"(max_workers={max_workers}, blas_limit={blas_limit}) ...", flush=True)

        with _make_executor(executor, max_workers) as ex:
            futs = {ex.submit(_run_one_start, (i, xstarts[i], start_seeds[i])): i
                    for i in pending}
            for n, fut in enumerate(as_completed(futs), 1):
                idx, row = fut.result()
                results[idx] = row
                if ckpt_dir:
                    _save_checkpoint(ckpt_dir, idx, row)
                print(f"    [{n}/{len(pending)}] start {idx} done, "
                      f"cost={row[-1]:.6g}", flush=True)

    if np.isnan(results).any():
        missing = sorted(np.where(np.isnan(results).any(axis=1))[0].tolist())
        raise RuntimeError(f"Starts produced no result: {missing}")

    top_rows = np.argsort(np.abs(results[:, -1]))[:10]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = out_dir / ("results_%d_%d_%s.csv" % (n_xstarts, seed, date_str))

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "params", "cost"])
        for rank, ridx in enumerate(top_rows, 1):
            w.writerow([rank, results[ridx][:-1].tolist(), float(results[ridx, -1])])

    return results, top_rows, str(csv_path)
