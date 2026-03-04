import os
import json
import math
import pandas as pd
import numpy as np
import netCDF4 as S
import xarray as xr
import sklearn
import glob
import pickle
from esem import gp_model
from esem.utils import get_random_params, plot_results, prettify_plot, add_121_line, leave_one_out
import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.metrics import make_scorer, r2_score
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline

from datetime import date
from datetime import datetime
import timeit
from timeit import default_timer as timer

from scipy.optimize import basinhopping
from scipy.optimize import minimize

import csv

#----------------------------------------------------------------------------------------------------

regions_file = xr.open_dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/regions.nc')
regions_list = ['poles','extratropical_land','extratropical_ocean','tropical_land','ascending_tropical_ocean','descending_tropical_ocean']
#area = ppe_dataset.area[1,:] #only taking the first row, because all rows should have the same values
control = xr.open_dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/SCREAM.2024-autocal-00.ne1024pg2/m0000/SCREAM.2024-autocal-00.ne1024pg2/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc')
area = control.variables['area'][:]
lat = control.variables['lat'][:]
lon = control.variables['lon'][:]

def zonal_means_native(data, area, lat, lon):
    lat_bands = np.linspace(-90,90,18) #currently dividing globe in 18 zones - 10 degree bands
    zonal_means = dict()
    for i in range(len(lat_bands) - 1):
        mask_zone = (lat > lat_bands[i]) & (lat < lat_bands[i+1]).squeeze()
        data_zone = np.where( mask_zone>0, data.squeeze(), np.nan)
        area_zone = np.where( mask_zone>0, area.squeeze(), np.nan)
        zone_mean = np.nansum(data_zone*area_zone) / np.nansum(area_zone)
        zone_center = abs(lat_bands[i] - lat_bands[i+1])/2 + lat_bands[i]
        zonal_means[zone_center] = zone_mean
    return zonal_means

def regional_means_native(data, area):
    region_data = xr.open_dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/regions.nc')
    regions_list = ['poles','extratropical_land','extratropical_ocean','tropical_land','ascending_tropical_ocean','descending_tropical_ocean']
    region_means = dict()
    for reg_name in regions_list:
        mask_reg = region_data[reg_name].squeeze()
        data_reg = np.where( mask_reg>0, data.squeeze(), np.nan)
        area_reg = np.where( mask_reg>0, area.squeeze(), np.nan)
        reg_mean = np.nansum(data_reg*area_reg) / np.nansum(area_reg)
        region_means[reg_name] = reg_mean
    return region_means

def global_means_native(data, area):
    global_mean = np.nanmean(data*area)/np.nanmean(area)
    return global_mean

#Load back obs
obs_filename = "/global/cfs/cdirs/e3sm/jpaige3/ESEm/TF_saving/obs_2025-07-30_09-40-13.pkl"

with open(obs_filename, 'rb') as f:
    loaded_obs = pickle.load(f)
    
zrg_obs = loaded_obs['zrg_obs']

n_cols_per_df = zrg_obs.shape[1] // 4 #50 DY1 and DY2
PCP_zrg_obs  = zrg_obs.iloc[:, 0 : n_cols_per_df]
TLWP_zrg_obs = zrg_obs.iloc[:, n_cols_per_df : 2 * n_cols_per_df]
OSR_zrg_obs  = zrg_obs.iloc[:, 2 * n_cols_per_df : 3 * n_cols_per_df]
OLR_zrg_obs  = zrg_obs.iloc[:, 3 * n_cols_per_df : 4 * n_cols_per_df]

# Load back GP proj
GP_proj_filename = "/global/cfs/cdirs/e3sm/jpaige3/ESEm/TF_saving/GPFlow_save2025-06-12/GP_ZRG_proj_2025-06-12_14-31-26.pkl"

with open(GP_proj_filename, 'rb') as f:
    loaded = pickle.load(f)

# Access data
X_pipe_sk_minmax = loaded['X_pipeline']
Y_pipe_sk_ss_PCP = loaded['Y_pipeline_PCP']
Y_pipe_sk_ss_TLWP = loaded['Y_pipeline_TLWP']
Y_pipe_sk_ss_OSR = loaded['Y_pipeline_OSR']
Y_pipe_sk_ss_OLR = loaded['Y_pipeline_OLR']
train_run_labels = loaded['X_train_index']
### normalized/transformed
X_train_norm = loaded['X_train_norm']
m_gp = loaded['proj_norm_gp']
v_gp = loaded['proj_v_norm_gp']
### unnormalized/untransformed
X_train = loaded['X_train']
Y_train_ZRG = loaded['Y_train']
proj_gp = loaded['proj_gp']
proj_v_gp = loaded['proj_v_gp']

PCP_train = Y_train_ZRG[:, :, 0]
TLWP_train = Y_train_ZRG[:, :, 1]
OSR_train = Y_train_ZRG[:, :, 2]
OLR_train = Y_train_ZRG[:, :, 3]

#transform data
X_pipe_sk_minmax = preprocessing.MinMaxScaler()
X_pipe_sk_minmax.fit(X_train)
X_train_norm = X_pipe_sk_minmax.transform(X_train)

#from scikitlearn
Y_pipe_sk_ss_PCP = preprocessing.StandardScaler()
Y_pipe_sk_ss_PCP.fit(PCP_train)
PCP_train_norm = Y_pipe_sk_ss_PCP.transform(PCP_train)

Y_pipe_sk_ss_TLWP = preprocessing.StandardScaler() #RobustScaler()
Y_pipe_sk_ss_TLWP.fit(TLWP_train)
TLWP_train_norm = Y_pipe_sk_ss_TLWP.transform(TLWP_train)

Y_pipe_sk_ss_OSR = preprocessing.StandardScaler()
Y_pipe_sk_ss_OSR.fit(OSR_train)
OSR_train_norm = Y_pipe_sk_ss_OSR.transform(OSR_train)

Y_pipe_sk_ss_OLR = preprocessing.StandardScaler()
Y_pipe_sk_ss_OLR.fit(OLR_train)
OLR_train_norm = Y_pipe_sk_ss_OLR.transform(OLR_train)

Y_train_norm = np.stack((PCP_train_norm, TLWP_train_norm, OSR_train_norm, OLR_train_norm), axis = 0)
Y_train_norm = np.transpose(Y_train_norm, (1, 2, 0))
print(X_train_norm.shape, Y_train_norm.shape)

PCP_zrg_obs.columns = PCP_zrg_obs.columns.astype(str)
TLWP_zrg_obs.columns = TLWP_zrg_obs.columns.astype(str)
OSR_zrg_obs.columns = OSR_zrg_obs.columns.astype(str)
OLR_zrg_obs.columns = OLR_zrg_obs.columns.astype(str)

PCP_obs_norm = Y_pipe_sk_ss_PCP.transform(PCP_zrg_obs)
TLWP_obs_norm = Y_pipe_sk_ss_TLWP.transform(TLWP_zrg_obs)
OSR_obs_norm = Y_pipe_sk_ss_OSR.transform(OSR_zrg_obs)
OLR_obs_norm = Y_pipe_sk_ss_OLR.transform(OLR_zrg_obs)

obs_norm = np.stack([PCP_obs_norm, TLWP_obs_norm, OSR_obs_norm, OLR_obs_norm])
obs_norm = obs_norm.transpose(1, 2, 0)
obs_norm.shape

obs_untransform = np.stack([PCP_zrg_obs, TLWP_zrg_obs, OSR_zrg_obs, OLR_zrg_obs])
obs_untransform = obs_untransform.transpose(1, 2, 0)
obs_untransform.shape

print(X_train_norm.shape, Y_train_norm.shape)

model_gp = gp_model(X_train_norm, Y_train_norm)

model_gp.train()

#variable weighting
var_weights_dict = {
    'PCP': 0.25,
    'TLWP': 0.25,
    'OSR': 0.25,
    'OLR': 0.25
    }

#zonal, regional, global weighting
zrg_weights_dict = {
    'zonal': (1/3),
    'regional': (1/3),
    'global': (1/3)
    } 

#summer/winter weights
DY_weights_dict = {
    'DY1': (1/2),
    'DY2': (1/2)
    } 

lat_bands = np.linspace(-90,90,18) 

def params_to_cost(params_guess): #, obs, area_weights, var_weights_dict, zrg_weights_dict):
    m_gp_guess, v_gp_guess = model_gp.predict(params_guess.reshape(1, -1))
    cost = ZRG_cost_function_rmse(m_gp_guess, obs_norm, var_weights_dict, zrg_weights_dict) #area_weights,
    return cost

def ZRG_cost_function_rmse(preds, obs, var_weights_dict, zrg_weights_dict): #area_weights,
    PCP_proj_c = preds[:, :, 0] #this is a numpy array
    TLWP_proj_c = preds[:, :, 1]
    OSR_proj_c = preds[:, :, 2]
    OLR_proj_c = preds[:, :, 3]
    
    PCP_obs_c = obs[:, :, 0] #this is a numpy array
    TLWP_obs_c = obs[:, :, 1]
    OSR_obs_c = obs[:, :, 2]
    OLR_obs_c = obs[:, :, 3]
    
    z_num = len(lat_bands)
    r_num = len(regions_list)
    all_num = len(lat_bands) + len(regions_list) + 1

    DY1_zonal_cost = zrg_weights_dict['zonal']*np.mean(var_weights_dict['PCP']*rmse(PCP_obs_c[:, 0:z_num], PCP_proj_c[:, 0:z_num])                            
                                                   +var_weights_dict['TLWP']*rmse(TLWP_obs_c[:, 0:z_num], TLWP_proj_c[:, 0:z_num])
                                                   +var_weights_dict['OSR']*rmse(OSR_obs_c[:, 0:z_num], OSR_proj_c[:, 0:z_num])
                                                   +var_weights_dict['OLR']*rmse(OLR_obs_c[:, 0:z_num], OLR_proj_c[:, 0:z_num]))
    DY2_zonal_cost = zrg_weights_dict['zonal']*np.mean(var_weights_dict['PCP']*rmse(PCP_obs_c[:,(all_num):(all_num+z_num)], PCP_proj_c[:,(all_num):(all_num+z_num)])
                                                   +var_weights_dict['TLWP']*rmse(TLWP_obs_c[:,(all_num):(all_num+z_num)], TLWP_proj_c[:,(all_num):(all_num+z_num)])
                                                   +var_weights_dict['OSR']*rmse(OSR_obs_c[:,(all_num):(all_num+z_num)], OSR_proj_c[:,(all_num):(all_num+z_num)])
                                                   +var_weights_dict['OLR']*rmse(OLR_obs_c[:,(all_num):(all_num+z_num)], OLR_proj_c[:,(all_num):(all_num+z_num)]))
    
    DY1_regional_cost = zrg_weights_dict['regional']*np.mean(var_weights_dict['PCP']*rmse(PCP_obs_c[:,(z_num):(z_num+r_num)], PCP_proj_c[:,(z_num):(z_num+r_num)])
                                                         +var_weights_dict['TLWP']*rmse(TLWP_obs_c[:,(z_num):(z_num+r_num)], TLWP_proj_c[:,(z_num):(z_num+r_num)])
                                                         +var_weights_dict['OSR']*rmse(OSR_obs_c[:,(z_num):(z_num+r_num)], OSR_proj_c[:,(z_num):(z_num+r_num)])
                                                         +var_weights_dict['OLR']*rmse(OLR_obs_c[:,(z_num):(z_num+r_num)], OLR_proj_c[:,(z_num):(z_num+r_num)]))
    DY2_regional_cost = zrg_weights_dict['regional']*np.mean(var_weights_dict['PCP']*rmse(PCP_obs_c[:,(all_num+z_num):(all_num+z_num+r_num)], PCP_proj_c[:,(all_num+z_num):(all_num+z_num+r_num)])
                                                         +var_weights_dict['TLWP']*rmse(TLWP_obs_c[:,(all_num+z_num):(all_num+z_num+r_num)], TLWP_proj_c[:,(all_num+z_num):(all_num+z_num+r_num)])
                                                         +var_weights_dict['OSR']*rmse(OSR_obs_c[:,(all_num+z_num):(all_num+z_num+r_num)], OSR_proj_c[:,(all_num+z_num):(all_num+z_num+r_num)])
                                                         +var_weights_dict['OLR']*rmse(OLR_obs_c[:,(all_num+z_num):(all_num+z_num+r_num)], OLR_proj_c[:,(all_num+z_num):(all_num+z_num+r_num)]))
    
    DY1_global_cost = zrg_weights_dict['global']*np.mean(var_weights_dict['PCP']*abs(PCP_obs_c[:,all_num-1] - PCP_proj_c[:,all_num-1])
                                                     +var_weights_dict['TLWP']*abs(TLWP_obs_c[:,all_num-1] - TLWP_proj_c[:,all_num-1])
                                                     +var_weights_dict['OSR']*abs(OSR_obs_c[:,all_num-1] - OSR_proj_c[:,all_num-1])
                                                     +var_weights_dict['OLR']*abs(OLR_obs_c[:,all_num-1] - OLR_proj_c[:,all_num-1]))
    DY2_global_cost = zrg_weights_dict['global']*np.mean(var_weights_dict['PCP']*abs(PCP_obs_c[:,-1] - PCP_proj_c[:,-1])
                                                     +var_weights_dict['TLWP']*abs(TLWP_obs_c[:,-1] - TLWP_proj_c[:,-1])
                                                     +var_weights_dict['OSR']*abs(OSR_obs_c[:,-1] - OSR_proj_c[:,-1])
                                                     +var_weights_dict['OLR']*abs(OLR_obs_c[:,-1] - OLR_proj_c[:,-1]))

    cost = DY_weights_dict['DY1']*(DY1_zonal_cost + DY1_regional_cost + DY1_global_cost) + DY_weights_dict['DY2']*(DY2_zonal_cost + DY2_regional_cost + DY2_global_cost)
    return cost


PCP_default_cost = PCP_train_norm[0] - PCP_obs_norm
TLWP_default_cost = TLWP_train_norm[0] - TLWP_obs_norm
OSR_default_cost = OSR_train_norm[0] - OSR_obs_norm
OLR_default_cost = OLR_train_norm[0] - OLR_obs_norm

default_cost = ZRG_cost_function_rmse(Y_train_norm[0][np.newaxis, :], obs_norm, var_weights_dict, zrg_weights_dict)
print('default direct:')
print(default_cost)

print('default function:')
print(params_to_cost(X_train_norm[0]))

regions_file = xr.open_dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/regions.nc')
regions_list = ['poles','extratropical_land','extratropical_ocean','tropical_land','ascending_tropical_ocean','descending_tropical_ocean']
#area = ppe_dataset.area[1,:] #only taking the first row, because all rows should have the same values
control = xr.open_dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/SCREAM.2024-autocal-00.ne1024pg2/m0000/SCREAM.2024-autocal-00.ne1024pg2/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc')
area = control.variables['area'][:]
lat = control.variables['lat'][:]
lon = control.variables['lon'][:]
lat_bands = np.linspace(-90,90,18) #currently dividing globe in 18 zones - 10 degree bands

def zonal_means_native(data, area, lat, lon):
    lat_bands = np.linspace(-90,90,18) #currently dividing globe in 18 zones - 10 degree bands
    zonal_means = dict()
    for i in range(len(lat_bands) - 1):
        mask_zone = (lat > lat_bands[i]) & (lat < lat_bands[i+1]).squeeze()
        data_zone = np.where( mask_zone>0, data.squeeze(), np.nan)
        area_zone = np.where( mask_zone>0, area.squeeze(), np.nan)
        zone_mean = np.nansum(data_zone*area_zone) / np.nansum(area_zone)
        zone_center = abs(lat_bands[i] - lat_bands[i+1])/2 + lat_bands[i]
        zonal_means[zone_center] = zone_mean
        #zonal_means[i] = zone_mean
    return zonal_means

def regional_means_native(data, area):
    region_data = xr.open_dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/regions.nc')
    regions_list = ['poles','extratropical_land','extratropical_ocean','tropical_land','ascending_tropical_ocean','descending_tropical_ocean']
    region_means = dict()
    for reg_name in regions_list:
        mask_reg = region_data[reg_name].squeeze()
        data_reg = np.where( mask_reg>0, data.squeeze(), np.nan)
        area_reg = np.where( mask_reg>0, area.squeeze(), np.nan)
        reg_mean = np.nansum(data_reg*area_reg) / np.nansum(area_reg)
        region_means[reg_name] = reg_mean
    return region_means

def global_means_native(data, area):
    global_mean = np.nanmean(data*area)/np.nanmean(area)
    return global_mean

from concurrent.futures import ThreadPoolExecutor
print('all start')
start_1 = timer()

def run_bh(xstart):
    local_minimizer = {
        "method": "L-BFGS-B",
        "bounds": [(0.0, 1.0)]*16,
    }
    print('start')
    start = timer()
    result = basinhopping(
        params_to_cost,
        xstart,
        minimizer_kwargs=local_minimizer,
        niter=1
    )
    print('result', result)
    end = timer()
    print(f"time: {end - start:.4f} seconds")
    return np.hstack((result.x, result.fun))

# List of starting points
seed = 62
N_xstarts = 10
rn = np.random.RandomState(seed)
xstarts = rn.rand(N_xstarts, 16)
print(xstarts)

with ThreadPoolExecutor() as executor:
    results = list(executor.map(run_bh, xstarts))
    results = np.vstack(results)

    top10_rows = np.argsort(abs(results[:, -1]))[:10]

    # Save results
    date_str = datetime.now().strftime("%Y-%m-%d")
    #csv_filename = f"/global/cfs/cdirs/e3sm/jpaige3/optimizing/Optimizing_results/results{N_xstarts}_{seed}_{date_str}.csv"
    csv_filename = f"/global/homes/r/rebassoo/work/2025_12_05_Autotuning/results{N_xstarts}_{seed}_{date_str}.csv"
    # Save as CSV
    with open(csv_filename, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["Rank", "param", "cost"])
        for idx, row in enumerate(top10_rows, 1):
            params = results[row][0:-1]
            cost = results[row, -1]
            print(f"Result {idx}:")
            print("  Parameters:", params)
            print("  Cost:", cost)
            writer.writerow([idx, params, cost])


end_1 = timer()
print(f"all time: {end_1 - start_1:.4f} seconds")

# results now contains the output from each run_bh(xstart)
