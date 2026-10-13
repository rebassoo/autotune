from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Literal
import yaml

BackendName = Literal["numpy", "torch", "cupy"]
DeviceName = Literal["cpu", "cuda"]

@dataclass
class Paths:
    obs_pkl: str
    gp_proj_pkl: str
    output_dir: str
    preprocess_dir: str = ""   # where preprocessing writes its output pickles

@dataclass
class VariableCfg:
    """Configuration for one target variable."""
    sim_field: str              # field name in the simulation dataset
    obs_nc_var: str             # netCDF variable name inside the obs file
    obs_scale: float = 1.0      # multiply obs values by this (e.g. 0.001 for g→kg)
    sim_components: Optional[List[str]] = None  # if set, sim_field = sum of these
    # Generic N-snapshot mode: {snapshot_label: obs_filename}
    obs_files: Optional[Dict[str, str]] = None
    # Two-snapshot (DY1/DY2) backward-compat mode:
    obs_file_DY1: Optional[str] = None     # obs filename within DY1_obs_dir
    obs_file_DY2: Optional[str] = None     # obs filename within DY2_obs_dir

@dataclass
class SnapshotCfg:
    """Configuration for one time snapshot (or averaged period)."""
    label: str              # e.g. "DY1", "DY2", "ANN"
    weight: float           # contribution to cost function
    sim_dir: str            # top-level dir containing member subdirectories
    obs_dir: str            # dir containing obs files for this snapshot
    nc_suffix: Optional[str] = None  # single file per member: subdir/nc_suffix
    nc_glob: Optional[str] = None    # multi-file per member: glob in subdir/run/ (averaged)
    min_files: int = 1               # minimum matching files required per member
    max_files: Optional[int] = None  # if set, use only the first max_files sorted files

@dataclass
class PreprocessCfg:
    params_json: str
    control_file: str                        # provides area / lat / lon
    regions_file: str
    variables: Dict[str, VariableCfg] = field(default_factory=dict)
    drop_zonal_bands: Optional[List[float]] = None
    # Generic N-snapshot mode:
    snapshots: Optional[List[SnapshotCfg]] = None
    # Two-snapshot (DY1/DY2) backward-compat mode:
    DY1_sim_dir: Optional[str] = None
    DY2_sim_dir: Optional[str] = None
    DY1_nc_suffix: Optional[str] = None
    DY2_nc_suffix: Optional[str] = None
    DY1_obs_dir: Optional[str] = None
    DY2_obs_dir: Optional[str] = None

@dataclass
class DataCfg:
    n_zonal: int
    regions_list: List[str]
    variables: List[str]        # must match keys in preprocess.variables

@dataclass
class WeightsCfg:
    variables: Dict[str, float]         # keys must match preprocess.variables
    zrg: Dict[str, float]
    dy: Dict[str, float]
    zonal_weights: Optional[List[float]] = None    # per-zone sample weights (len=n_zonal); None = uniform
    regional_weights: Optional[List[float]] = None # per-region sample weights (len=n_regions); None = uniform

@dataclass
class OptimizeCfg:
    seed: int
    n_xstarts: int
    niter: int
    method: str
    bounds: Dict[str, float]
    n_params: int
    max_workers: Optional[int] = None
    param_ordering_constraints: Optional[List[List[str]]] = None  # [[low_param, high_param], ...]
    param_physical_bounds: Optional[Dict[str, List[float]]] = None  # {param_name: [low, high]}
    # Parameters sampled log-uniformly in the design, scaled in log space
    # rather than linearly so the training inputs are uniform in the GP's
    # input space (see autotune_gp.transforms.LogMinMaxScaler). Names must
    # appear in param_physical_bounds and have strictly positive bounds.
    # Empty/None reproduces the previous all-linear behaviour exactly.
    log_params: Optional[List[str]] = None
    # When true, per-parameter optimizer bounds are narrowed to the range the
    # PPE actually sampled (intersected with `bounds`), instead of searching the
    # full declared param_physical_bounds. Several params are sampled over only
    # a fraction of their declared range, so the default [0,1] search lets the
    # optimizer roam where the GP has no training data and simply reverts to
    # its prior mean.
    bounds_from_data: bool = False
    # 'thread' (default) or 'process'. Multi-fidelity must use 'process': GPy
    # keeps kernel slice state on the shared model object, so threading its
    # predict races and crashes. Single-fidelity must stay on 'thread' — it is
    # ESEm/GPflow, and TensorFlow does not survive fork.
    executor: str = "thread"
    # Cost function used by the optimizer: 'mae' (default, unchanged historical
    # behaviour, NOT a real likelihood -- no log(v_c)/normalization term, never
    # touches the GP's predictive variance at all), 'gaussian_gp_variance' — a
    # Gaussian negative-log-likelihood that uses the GP's own predictive
    # variance as the noise term (see
    # autotune_gp.cost.zrg_cost_function_gaussian_nll) — 'student_t' — same
    # idea but with a Student-t per-location term (see
    # autotune_gp.cost.zrg_cost_function_student_t), whose heavier tails keep
    # a single hard-to-fit variable/location from dominating the total cost
    # the way a squared-error term can — or 'laplace_gp_variance' — the
    # likelihood-based (heteroscedasticity-aware) counterpart to plain 'mae':
    # L1 residual growth like 'mae', but scaled by the GP's own v_c and a
    # proper normalizing term, same as 'gaussian_gp_variance'/'student_t' (see
    # autotune_gp.cost.zrg_cost_function_laplace). Only 'gaussian_gp_variance',
    # 'student_t', or 'laplace_gp_variance' make the post-hoc
    # parameter-uncertainty report (diagnostics.report_uncertainty)
    # statistically meaningful -- and even then, 'laplace_gp_variance' can
    # return NaN/unstable stderr for individual parameters whose main
    # influence is on a location where the residual sits very close to zero
    # at the optimum (the |r| kink at r=0), unlike the other two.
    likelihood: str = "mae"
    # Floor applied to the GP's predictive variance before dividing by it in the
    # 'gaussian_gp_variance'/'student_t'/'laplace_gp_variance' likelihoods, to
    # guard against blow-up where the GP is (near-)certain. Ignored when
    # likelihood='mae'.
    var_floor: float = 1e-6
    # Extra +kappa*pred_var term added to the 'gaussian_gp_variance'/'student_t'
    # likelihoods (see autotune_gp.cost.zrg_cost_function_gaussian_nll /
    # zrg_cost_function_student_t). Without it, the optimizer can lower cost by
    # moving to a higher-variance point (shrinking the residual/var term faster
    # than the log(var) term grows), rather than to a genuinely better-fitting
    # one -- corners of the sampled envelope are exactly where GP variance
    # tends to be largest, so this can pull the optimum toward parameter
    # combinations the surrogate has little training data for. kappa > 0 makes
    # higher variance strictly costlier, closing that loophole. Default 0.0 =
    # off, recovers the original cost exactly. Ignored when likelihood='mae'.
    variance_penalty_kappa: float = 0.0
    # Degrees of freedom for the 'student_t' likelihood (see
    # autotune_gp.cost.zrg_cost_function_student_t). Smaller = heavier tails =
    # more MAE-like robustness to a single large residual; as dof -> infinity
    # this recovers 'gaussian_gp_variance' exactly. Ignored otherwise.
    student_t_dof: float = 4.0
    # Extra penalty (see autotune_gp.cost.training_distance_penalty) added
    # regardless of likelihood, based on distance (in each parameter's own
    # bounds_from_data-rescaled space) from the candidate point to the
    # nearest actual training-ensemble member. Zero unless that distance
    # exceeds the training ensemble's own worst-case internal spacing, i.e.
    # this only fires once a candidate is farther from any real training
    # point than the training data ever is from itself -- catching joint
    # (multi-parameter) combinations that were never sampled together even
    # when every individual parameter stays within its own sampled range.
    # Default 0.0 = off, recovers the original cost exactly.
    training_distance_penalty_lambda: float = 0.0

@dataclass
class RuntimeCfg:
    train_gp: bool = True
    backend: BackendName = "numpy"
    device: DeviceName = "cpu"
    tf_determinism: bool = False
    # Single-fidelity GP library: 'esem' (default, GPflow joint model — the
    # historical behaviour) or 'gpy' (independent per-feature GPy models, like
    # the multi-fidelity path). 'gpy' pickles and forks, so it enables GP
    # save/reload and the process executor for single-fidelity. Ignored by the
    # multi-fidelity path (always GPy/emukit). NOTE distinct from `backend`
    # above, which is the cost-function compute backend (numpy/torch/cupy).
    sf_gp_backend: str = "esem"

@dataclass
class DiagnosticsCfg:
    enabled: bool = False
    output_dir: Optional[str] = None  # defaults to paths.output_dir/diagnostics/ if not set
    # If true (and optimize.likelihood != 'mae'), run a post-hoc parameter-
    # uncertainty report (inverse-Hessian covariance at the best-found point)
    # after optimization — see autotune_gp.uncertainty.parameter_uncertainty.
    report_uncertainty: bool = False

@dataclass
class MultiFidelityCfg:
    """Optional multi-fidelity GP configuration (emukit AR1).

    When present in the config, stage 2 trains an AR1 model that combines
    the low-fidelity data (low_fidelity_dir) with the high-fidelity data
    (paths.preprocess_dir).  Stage 1 and the single-fidelity path are
    completely unaffected.
    """
    # Two-level path. Optional when fidelity_dirs is given instead.
    low_fidelity_dir: str = ""
    method: str = "AR1"
    # Optional 3+ level chain. Preprocess dirs of every level BELOW the top,
    # lowest first; the top level stays paths.preprocess_dir. When set (with
    # >= 2 entries) stage 2 builds an NFidelityGPWrapper instead of the
    # two-level AR1 model. low_fidelity_dir is then ignored.
    fidelity_dirs: Optional[List[str]] = None
    # Whose scalers.pkl normalises X and Y. Must be the level the warm-start
    # model was fitted in, and must not be a top level with too few members to
    # estimate scalers. Defaults to paths.preprocess_dir.
    scalers_dir: Optional[str] = None
    # Trained two-level mf_gp_trained.pkl used to initialise the two lowest
    # levels; strongly recommended when the top level has few members.
    warm_start_gp: Optional[str] = None
    # Hold the warm-started lower-level hyperparameters fixed.
    freeze_lower: bool = True
    # Discrepancy kernel for the TOP level: 'auto' uses one shared lengthscale
    # below 3*n_params members, per-parameter (ARD) above.
    top_kernel: str = "auto"

@dataclass
class Config:
    paths: Paths
    data: DataCfg
    weights: WeightsCfg
    optimize: OptimizeCfg
    runtime: RuntimeCfg
    diagnostics: DiagnosticsCfg = field(default_factory=DiagnosticsCfg)
    preprocess: Optional[PreprocessCfg] = None
    multi_fidelity: Optional[MultiFidelityCfg] = None

def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f)

    paths_raw = dict(raw["paths"])
    paths_raw.setdefault("preprocess_dir", "")
    paths = Paths(**paths_raw)
    data = DataCfg(**raw["data"])
    w_raw = dict(raw["weights"])
    weights = WeightsCfg(
        variables=w_raw["variables"],
        zrg=w_raw["zrg"],
        dy=w_raw["dy"],
        zonal_weights=w_raw.get("zonal_weights", None),
        regional_weights=w_raw.get("regional_weights", None),
    )

    opt = raw["optimize"]
    optimize = OptimizeCfg(
        seed=int(opt["seed"]),
        n_xstarts=int(opt["n_xstarts"]),
        niter=int(opt["niter"]),
        method=str(opt["method"]),
        bounds=dict(opt["bounds"]),
        n_params=int(opt["n_params"]),
        max_workers=opt.get("max_workers", None),
        param_ordering_constraints=opt.get("param_ordering_constraints", None),
        param_physical_bounds=opt.get("param_physical_bounds", None),
        bounds_from_data=bool(opt.get("bounds_from_data", False)),
        executor=str(opt.get("executor", "thread")).lower(),
        likelihood=str(opt.get("likelihood", "mae")).lower(),
        var_floor=float(opt.get("var_floor", 1e-6)),
        variance_penalty_kappa=float(opt.get("variance_penalty_kappa", 0.0)),
        training_distance_penalty_lambda=float(opt.get("training_distance_penalty_lambda", 0.0)),
        student_t_dof=float(opt.get("student_t_dof", 4.0)),
    )

    rt = raw.get("runtime", {})
    runtime = RuntimeCfg(
        train_gp=bool(rt.get("train_gp", True)),
        backend=str(rt.get("backend", "numpy")).lower(),  # type: ignore
        device=str(rt.get("device", "cpu")).lower(),      # type: ignore
        tf_determinism=bool(rt.get("tf_determinism", True)),
        sf_gp_backend=str(rt.get("sf_gp_backend", "esem")).lower(),
    )

    diag_raw = raw.get("diagnostics", {})
    diagnostics = DiagnosticsCfg(
        enabled=bool(diag_raw.get("enabled", False)),
        output_dir=diag_raw.get("output_dir", None),
        report_uncertainty=bool(diag_raw.get("report_uncertainty", False)),
    )

    preprocess = None
    if "preprocess" in raw:
        pp_raw = dict(raw["preprocess"])
        variables = {}
        for var_name, var_raw in pp_raw.pop("variables", {}).items():
            variables[var_name] = VariableCfg(**var_raw)
        drop_zonal_bands = pp_raw.pop("drop_zonal_bands", None)
        snapshots = None
        if "snapshots" in pp_raw:
            snapshots = [SnapshotCfg(**s) for s in pp_raw.pop("snapshots")]
        preprocess = PreprocessCfg(**pp_raw, variables=variables,
                                   drop_zonal_bands=drop_zonal_bands,
                                   snapshots=snapshots)

    mf_raw = raw.get("multi_fidelity", None)
    multi_fidelity = MultiFidelityCfg(**mf_raw) if mf_raw else None

    return Config(paths=paths, data=data, weights=weights, optimize=optimize,
                  runtime=runtime, diagnostics=diagnostics, preprocess=preprocess,
                  multi_fidelity=multi_fidelity)
