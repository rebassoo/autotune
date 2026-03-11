import os
import pandas as pd
import numpy as np
import xarray as xr
import glob
import pickle
from esem import gp_model

from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_error as mae
from sklearn import preprocessing

from datetime import datetime
from timeit import default_timer as timer

from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import basinhopping
from scipy.optimize import minimize

import csv
import importlib.util

#----------------------------------------------------------------------------------------------------

#input: 
seed = 5
N_xstarts = 10
costname = 'default' #call cost function without '_cost_fun'
#'default', 'precip', 'no_region', 'tropics_weighted'

#----------------------------------------------------------------------------------------------------

cost_function_path =  f"/global/cfs/cdirs/e3sm/jpaige3/optimizing/cost_functions/{costname}_cost_fun.py"
spec = importlib.util.spec_from_file_location("cost_function", cost_function_path)
cost_function = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cost_function)

regions_file = xr.open_dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/regions.nc')
regions_list = ['poles','extratropical_land','extratropical_ocean','tropical_land','ascending_tropical_ocean','descending_tropical_ocean']
#area = ppe_dataset.area[1,:] #only taking the first row, because all rows should have the same values
control = xr.open_dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/SCREAM.2024-autocal-00.ne1024pg2/m0000/SCREAM.2024-autocal-00.ne1024pg2/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc')
area = control.variables['area'][:]
lat = control.variables['lat'][:]
lon = control.variables['lon'][:]

# Load back GP proj
GP_proj_filename = "/global/cfs/cdirs/e3sm/jpaige3/ESEm/GP_Saved_Model_Data/CORRECT_GP_ZRG_masked_proj_2026-03-04_11-36-39.pkl" #has masks properly and norms to full parameter ranges
#has masks properly and norms to full parameter ranges

with open(GP_proj_filename, 'rb') as f:
    loaded = pickle.load(f)

# Access data
train_run_labels = loaded['X_train_index']
X_pipe_sk_minmax = loaded['X_pipeline']
Y_pipe_sk_ss_PCP = loaded['Y_pipeline_PCP']
Y_pipe_sk_ss_TLWP = loaded['Y_pipeline_TLWP']
Y_pipe_sk_ss_OSR = loaded['Y_pipeline_OSR']
Y_pipe_sk_ss_OLR = loaded['Y_pipeline_OLR']
### normalized/transformed
X_train_norm = loaded['X_train_norm']
Y_train_norm = loaded['Y_train_norm']
PCP_train_norm = loaded['PCP_train_norm']
TLWP_train_norm = loaded['TLWP_train_norm']
OSR_train_norm = loaded['OSR_train_norm']
OLR_train_norm = loaded['OLR_train_norm']
### unnormalized/untransformed
X_train = loaded['X_train']
Y_train_ZRG = loaded['Y_train']
PCP_train = loaded['PCP_train']
TLWP_train = loaded['TLWP_train']
OSR_train = loaded['OSR_train']
OLR_train = loaded['OLR_train']
### 
print(X_train_norm.shape, Y_train_norm.shape)

#Load back obs
obs_filename = "/global/cfs/cdirs/e3sm/jpaige3/ESEm/GP_Saved_Model_Data/CORRECT_obs_2026-03-04_11-36-39.pkl"

with open(obs_filename, 'rb') as f:
    loaded_obs = pickle.load(f)
    
zrg_obs = loaded_obs['zrg_obs']
PCP_zrg_obs = loaded_obs['PCP_zrg_obs']
TLWP_zrg_obs = loaded_obs['TLWP_zrg_obs']
OSR_zrg_obs = loaded_obs['OSR_zrg_obs']
OLR_zrg_obs = loaded_obs['OLR_zrg_obs']
n_cols_per_df = zrg_obs.shape[1] // 4 #50 DY1 and DY2
##

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

obs_untransform = np.stack([PCP_zrg_obs, TLWP_zrg_obs, OSR_zrg_obs, OLR_zrg_obs])
obs_untransform = obs_untransform.transpose(1, 2, 0)

print(obs_norm.shape)

print('Training...')

model_gp = gp_model(X_train_norm, Y_train_norm)

model_gp.train()

#variable weighting
var_weights_dict = cost_function.var_weights_dict
#zonal, regional, global weighting
zrg_weights_dict = cost_function.zrg_weights_dict
#summer/winter weights
DY_weights_dict = cost_function.DY_weights_dict
#zonal weights for weighted combination
zonal_weights = cost_function.zonal_weights
#regional weights for weighted combination
regional_weights = cost_function.regional_weights

lat_bands = np.linspace(-90,90,19) #modified 1/30/26

#lamlow_index = X_train.columns.get_loc('lambda_low')
#lamhigh_index = X_train.columns.get_loc('lambda_high')

def params_to_cost(params_guess): #, obs, area_weights, var_weights_dict, zrg_weights_dict):
    #if params_guess[6] > params_guess[7]:
    #if params_guess['lambda_low'] > params_guess['lambda_high']
    #    return 1e100 #upweight this result
    violation = params_guess[6] - params_guess[7]  # lambda_low - lambda_high
    if violation > 0:
        return 1e2 + 1e2 * violation  # linear penalty: gets worse the more you violate
    m_gp_guess, v_gp_guess = model_gp.predict(params_guess.reshape(1, -1))
    cost = ZRG_cost_function_mae_weighted(m_gp_guess, obs_norm, var_weights_dict, DY_weights_dict, zrg_weights_dict, zonal_weights, regional_weights)
    return cost


#def params_to_cost(params_guess): #, obs, area_weights, var_weights_dict, zrg_weights_dict):
#    if params_guess[lamlow_index] > params_guess[lamhigh_index]:
#        return 1e100 #upweight this result
#    m_gp_guess, v_gp_guess = model_gp.predict(params_guess.reshape(1, -1))
#    cost = ZRG_cost_function_mae_weighted(m_gp_guess, obs_norm, var_weights_dict, DY_weights_dict, zrg_weights_dict, zonal_weights, regional_weights)
#    return cost

def ZRG_cost_function_mae_weighted(preds, obs, var_weights_dict, DY_weights_dict, zrg_weights_dict, zonal_weights = None, regional_weights = None): #area_weights,  
    preds = preds.squeeze()
    obs = obs.squeeze()
    
    PCP_proj_c = preds[:,0] #this is a numpy array
    TLWP_proj_c = preds[:,1]
    OSR_proj_c = preds[:,2]
    OLR_proj_c = preds[:,3]
    
    PCP_obs_c = obs[:,0] #this is a numpy array
    TLWP_obs_c = obs[:,1]
    OSR_obs_c = obs[:,2]
    OLR_obs_c = obs[:,3]
    
    z_num = len(lat_bands)-1
    r_num = len(regions_list)
    all_num = z_num + r_num + 1

    DY1_zonal_cost = zrg_weights_dict['zonal']*np.mean([var_weights_dict['PCP']*mae(PCP_obs_c[0:z_num], PCP_proj_c[0:z_num], sample_weight=zonal_weights),                            
                                                   var_weights_dict['TLWP']*mae(TLWP_obs_c[0:z_num], TLWP_proj_c[0:z_num], sample_weight=zonal_weights),
                                                   var_weights_dict['OSR']*mae(OSR_obs_c[0:z_num], OSR_proj_c[0:z_num], sample_weight=zonal_weights),
                                                   var_weights_dict['OLR']*mae(OLR_obs_c[0:z_num], OLR_proj_c[0:z_num], sample_weight=zonal_weights)])
    DY2_zonal_cost = zrg_weights_dict['zonal']*np.mean([var_weights_dict['PCP']*mae(PCP_obs_c[(all_num):(all_num+z_num)], PCP_proj_c[(all_num):(all_num+z_num)], sample_weight=zonal_weights),
                                                   var_weights_dict['TLWP']*mae(TLWP_obs_c[(all_num):(all_num+z_num)], TLWP_proj_c[(all_num):(all_num+z_num)], sample_weight=zonal_weights),
                                                   var_weights_dict['OSR']*mae(OSR_obs_c[(all_num):(all_num+z_num)], OSR_proj_c[(all_num):(all_num+z_num)], sample_weight=zonal_weights),
                                                   var_weights_dict['OLR']*mae(OLR_obs_c[(all_num):(all_num+z_num)], OLR_proj_c[(all_num):(all_num+z_num)], sample_weight=zonal_weights)])
    
    DY1_regional_cost = zrg_weights_dict['regional']*np.mean([var_weights_dict['PCP']*mae(PCP_obs_c[(z_num):(z_num+r_num)], PCP_proj_c[(z_num):(z_num+r_num)], sample_weight=regional_weights),
                                                         var_weights_dict['TLWP']*mae(TLWP_obs_c[(z_num):(z_num+r_num)], TLWP_proj_c[(z_num):(z_num+r_num)], sample_weight=regional_weights),
                                                         var_weights_dict['OSR']*mae(OSR_obs_c[(z_num):(z_num+r_num)], OSR_proj_c[(z_num):(z_num+r_num)], sample_weight=regional_weights),
                                                         var_weights_dict['OLR']*mae(OLR_obs_c[(z_num):(z_num+r_num)], OLR_proj_c[(z_num):(z_num+r_num)], sample_weight=regional_weights)])
    DY2_regional_cost = zrg_weights_dict['regional']*np.mean([var_weights_dict['PCP']*mae(PCP_obs_c[(all_num+z_num):(all_num+z_num+r_num)], PCP_proj_c[(all_num+z_num):(all_num+z_num+r_num)], sample_weight=regional_weights),
                                                         var_weights_dict['TLWP']*mae(TLWP_obs_c[(all_num+z_num):(all_num+z_num+r_num)], TLWP_proj_c[(all_num+z_num):(all_num+z_num+r_num)], sample_weight=regional_weights),
                                                         var_weights_dict['OSR']*mae(OSR_obs_c[(all_num+z_num):(all_num+z_num+r_num)], OSR_proj_c[(all_num+z_num):(all_num+z_num+r_num)], sample_weight=regional_weights),
                                                         var_weights_dict['OLR']*mae(OLR_obs_c[(all_num+z_num):(all_num+z_num+r_num)], OLR_proj_c[(all_num+z_num):(all_num+z_num+r_num)], sample_weight=regional_weights)])
    
    DY1_global_cost = zrg_weights_dict['global']*np.mean([var_weights_dict['PCP']*abs(PCP_obs_c[all_num-1] - PCP_proj_c[all_num-1]),
                                                     var_weights_dict['TLWP']*abs(TLWP_obs_c[all_num-1] - TLWP_proj_c[all_num-1]),
                                                     var_weights_dict['OSR']*abs(OSR_obs_c[all_num-1] - OSR_proj_c[all_num-1]),
                                                     var_weights_dict['OLR']*abs(OLR_obs_c[all_num-1] - OLR_proj_c[all_num-1])])
    DY2_global_cost = zrg_weights_dict['global']*np.mean([var_weights_dict['PCP']*abs(PCP_obs_c[-1] - PCP_proj_c[-1]),
                                                     var_weights_dict['TLWP']*abs(TLWP_obs_c[-1] - TLWP_proj_c[-1]),
                                                     var_weights_dict['OSR']*abs(OSR_obs_c[-1] - OSR_proj_c[-1]),
                                                     var_weights_dict['OLR']*abs(OLR_obs_c[-1] - OLR_proj_c[-1])])

    cost = DY_weights_dict['DY1']*(DY1_zonal_cost + DY1_regional_cost + DY1_global_cost) + DY_weights_dict['DY2']*(DY2_zonal_cost + DY2_regional_cost + DY2_global_cost)
    
    return cost

PCP_default_cost = PCP_train_norm[0] - PCP_obs_norm
TLWP_default_cost = TLWP_train_norm[0] - TLWP_obs_norm
OSR_default_cost = OSR_train_norm[0] - OSR_obs_norm
OLR_default_cost = OLR_train_norm[0] - OLR_obs_norm

default_cost = ZRG_cost_function_mae_weighted(Y_train_norm[0][np.newaxis, :], obs_norm, var_weights_dict, DY_weights_dict, zrg_weights_dict, zonal_weights, regional_weights)

print('default direct:')
print(default_cost)

print('default function:')
print(params_to_cost(X_train_norm[0]))

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
        seed = seed,
        niter=1
    )
    print('result', result)
    end = timer()
    print(f"time: {end - start:.4f} seconds")
    return np.hstack((result.x, result.fun))

# List of starting points
rn = np.random.RandomState(seed)
xstarts = rn.rand(N_xstarts, 16)
print(xstarts)

with ThreadPoolExecutor() as executor:
    results = list(executor.map(run_bh, xstarts))
    results = np.vstack(results)
    #top10_rows = np.argsort(abs(results[:, -1]))[:10]
    top10_rows = np.argsort(abs(results[:, -1])) #[:N_xstarts]

    # Save results
    date_str = datetime.now().strftime("%Y-%m-%d")
    csv_filename = f"/global/cfs/cdirs/e3sm/jpaige3/optimizing/Optimizing_results/{costname}_cost_fun/results{N_xstarts}_{seed}_{date_str}_{costname}.csv"
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
            #writer.writerow([idx] + list(params) + [cost])


end_1 = timer()
print(f"all time: {end_1 - start_1:.4f} seconds")

# results now contains the output from each run_bh(xstart)
