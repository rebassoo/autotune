"""
Correctness gate for the process-parallel optimizer.

Per-start seeding makes each start's RNG stream independent of execution
order, and results are written back by index, so the parallel run must
reproduce the serial run *exactly*. Anything else means state is leaking
between starts.

Also exercises checkpoint resume: a run that is interrupted and restarted
must land on the same answer as one that ran straight through.

Uses a cheap analytic cost function — this tests the executor plumbing, not
the GP. Run the real thing separately for memory/timing.

Usage:
    python scripts/test_optimize_parallel.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autotune_gp.optimize import optimize_parallel

N_PARAMS = 6
N_STARTS = 8


def cost_fn(x):
    """Multi-modal and cheap: a couple of local minima so basinhopping has
    something to actually do."""
    x = np.asarray(x, dtype=float)
    return float(np.sum((x - 0.3) ** 2) + 0.1 * np.sum(np.sin(12.0 * x) ** 2))


def _run(tmp, executor, max_workers, ckpt=None):
    return optimize_parallel(
        cost_fn=cost_fn,
        n_params=N_PARAMS,
        bounds_low=[0.0] * N_PARAMS,
        bounds_high=[1.0] * N_PARAMS,
        seed=50,
        n_xstarts=N_STARTS,
        niter=2,
        method="L-BFGS-B",
        out_dir=tmp,
        max_workers=max_workers,
        executor=executor,
        checkpoint_dir=ckpt,
    )


def main():
    tmp = Path(tempfile.mkdtemp(prefix="opt_test_"))
    failures = []
    try:
        print("1. serial baseline (thread, 1 worker) ...")
        ser, ser_top, _ = _run(tmp / "serial", "thread", 1)
        print(f"   best cost {ser[ser_top[0], -1]:.10f}")

        print("2. process executor, 4 workers ...")
        par, par_top, _ = _run(tmp / "proc", "process", 4)
        print(f"   best cost {par[par_top[0], -1]:.10f}")

        if np.array_equal(ser, par):
            print("   PASS: bit-identical to serial")
        else:
            d = np.abs(ser - par).max()
            failures.append(f"process != serial (max abs diff {d:.3e})")
            print(f"   FAIL: max abs diff {d:.3e}")

        print("3. thread executor, 4 workers (SF path unchanged) ...")
        thr, thr_top, _ = _run(tmp / "thr", "thread", 4)
        if np.array_equal(ser, thr):
            print("   PASS: bit-identical to serial")
        else:
            failures.append("thread(4) != serial")
            print("   FAIL")

        print("4. checkpoint resume: run 8, drop 5, rerun ...")
        ck = tmp / "ck"
        full, _, _ = _run(tmp / "ck_out", "process", 4, ckpt=str(ck))
        removed = 0
        for i in (1, 3, 4, 6, 7):
            p = ck / f"start_{i:04d}.pkl"
            if p.exists():
                p.unlink()
                removed += 1
        print(f"   removed {removed} checkpoints, re-running")
        resumed, _, _ = _run(tmp / "ck_out2", "process", 4, ckpt=str(ck))
        if np.array_equal(full, resumed):
            print("   PASS: resumed run identical")
        else:
            failures.append("resumed != full")
            print("   FAIL")

        print("5. stale checkpoints (different seed) must be rejected ...")
        stale, _, _ = optimize_parallel(
            cost_fn=cost_fn, n_params=N_PARAMS,
            bounds_low=[0.0] * N_PARAMS, bounds_high=[1.0] * N_PARAMS,
            seed=999,                      # different seed -> different starts
            n_xstarts=N_STARTS, niter=2, method="L-BFGS-B",
            out_dir=tmp / "stale_out", max_workers=4,
            executor="process", checkpoint_dir=str(ck),
        )
        if np.array_equal(stale, full):
            failures.append("stale checkpoints were reused for a different seed")
            print("   FAIL: reused checkpoints across a seed change")
        else:
            print("   PASS: checkpoints invalidated and starts re-run")

        print()
        if failures:
            print("FAILURES:")
            for f in failures:
                print(f"  - {f}")
            return 1
        print("All checks passed.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
