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

@dataclass
class RuntimeCfg:
    train_gp: bool = True
    backend: BackendName = "numpy"
    device: DeviceName = "cpu"
    tf_determinism: bool = False

@dataclass
class DiagnosticsCfg:
    enabled: bool = False
    output_dir: Optional[str] = None  # defaults to paths.output_dir/diagnostics/ if not set

@dataclass
class MultiFidelityCfg:
    """Optional multi-fidelity GP configuration (emukit AR1).

    When present in the config, stage 2 trains an AR1 model that combines
    the low-fidelity data (low_fidelity_dir) with the high-fidelity data
    (paths.preprocess_dir).  Stage 1 and the single-fidelity path are
    completely unaffected.
    """
    low_fidelity_dir: str
    method: str = "AR1"

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
    )

    rt = raw.get("runtime", {})
    runtime = RuntimeCfg(
        train_gp=bool(rt.get("train_gp", True)),
        backend=str(rt.get("backend", "numpy")).lower(),  # type: ignore
        device=str(rt.get("device", "cpu")).lower(),      # type: ignore
        tf_determinism=bool(rt.get("tf_determinism", True)),
    )

    diag_raw = raw.get("diagnostics", {})
    diagnostics = DiagnosticsCfg(
        enabled=bool(diag_raw.get("enabled", False)),
        output_dir=diag_raw.get("output_dir", None),
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
