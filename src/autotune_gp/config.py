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
    obs_file_DY1: str           # obs filename within DY1_obs_dir
    obs_file_DY2: str           # obs filename within DY2_obs_dir
    obs_nc_var: str             # netCDF variable name inside the obs file
    obs_scale: float = 1.0      # multiply obs values by this (e.g. 0.001 for g→kg)
    sim_components: Optional[List[str]] = None  # if set, sim_field = sum of these

@dataclass
class PreprocessCfg:
    params_json: str
    DY1_sim_dir: str
    DY2_sim_dir: str
    DY1_nc_suffix: str
    DY2_nc_suffix: str
    DY1_obs_dir: str
    DY2_obs_dir: str
    control_file: str                       # provides area / lat / lon
    regions_file: str
    variables: Dict[str, VariableCfg] = field(default_factory=dict)

@dataclass
class DataCfg:
    n_zonal: int
    regions_list: List[str]
    variables: List[str]        # must match keys in preprocess.variables

@dataclass
class WeightsCfg:
    variables: Dict[str, float] # keys must match preprocess.variables
    zrg: Dict[str, float]
    dy: Dict[str, float]

@dataclass
class OptimizeCfg:
    seed: int
    n_xstarts: int
    niter: int
    method: str
    bounds: Dict[str, float]
    n_params: int
    max_workers: Optional[int] = None

@dataclass
class RuntimeCfg:
    train_gp: bool = True
    backend: BackendName = "numpy"
    device: DeviceName = "cpu"

@dataclass
class Config:
    paths: Paths
    data: DataCfg
    weights: WeightsCfg
    optimize: OptimizeCfg
    runtime: RuntimeCfg
    preprocess: Optional[PreprocessCfg] = None

def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f)

    paths_raw = dict(raw["paths"])
    paths_raw.setdefault("preprocess_dir", "")
    paths = Paths(**paths_raw)
    data = DataCfg(**raw["data"])
    weights = WeightsCfg(**raw["weights"])

    opt = raw["optimize"]
    optimize = OptimizeCfg(
        seed=int(opt["seed"]),
        n_xstarts=int(opt["n_xstarts"]),
        niter=int(opt["niter"]),
        method=str(opt["method"]),
        bounds=dict(opt["bounds"]),
        n_params=int(opt["n_params"]),
        max_workers=opt.get("max_workers", None),
    )

    rt = raw.get("runtime", {})
    runtime = RuntimeCfg(
        train_gp=bool(rt.get("train_gp", True)),
        backend=str(rt.get("backend", "numpy")).lower(),  # type: ignore
        device=str(rt.get("device", "cpu")).lower(),      # type: ignore
    )

    preprocess = None
    if "preprocess" in raw:
        pp_raw = dict(raw["preprocess"])
        variables = {}
        for var_name, var_raw in pp_raw.pop("variables", {}).items():
            variables[var_name] = VariableCfg(**var_raw)
        preprocess = PreprocessCfg(**pp_raw, variables=variables)

    return Config(paths=paths, data=data, weights=weights, optimize=optimize,
                  runtime=runtime, preprocess=preprocess)
