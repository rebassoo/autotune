"""load_config must carry every optimize key the pipeline reads.

log_params was declared on OptimizeCfg but never passed through by
load_config, so log scaling was silently off in every run that set it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from autotune_gp.config import load_config

CFG = Path(__file__).parent.parent / "configs" / "perlmutter_ne128_prod_v2_annual.yaml"


def test_log_params_survive_loading():
    cfg = load_config(str(CFG))
    assert cfg.optimize.log_params == [
        "c_diag_3rd_mom", "coeff_kh", "coeff_km",
        "accretion_prefactor", "max_total_ni", "autoconversion_prefactor",
    ]
