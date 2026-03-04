#!/usr/bin/env python
# coding: utf-8

# # Abbreviated surrogate training workflow
# 
# The below is implimented in a python file '/global/cfs/cdirs/e3sm/jpaige3/optimizing/run_GPsurrogate_fromsave.py'
# 
# The documentation for ESEm is at https://esem.readthedocs.io/en/latest/

# Resolving the warnings thrown about installs, especially the tensorflow, might help with optimization functions, like the ABC sampler. Watson-Parris says the tenserflow install is very finicky.

# In[1]:


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

#import gpflow
#import tensorflow as tf
from datetime import date
from datetime import datetime
import timeit
from timeit import default_timer as timer

from concurrent.futures import ThreadPoolExecutor

from scipy.optimize import basinhopping
from scipy.optimize import minimize
import csv

#address warning messages: pip install tensorflow[and-cuda]


# In[ ]:


DY1_obs_dir = '/global/cfs/cdirs/e3smdata/simulations/ecp-autotune/obs/'
DY1_precip_obs_file = DY1_obs_dir + 'IMERG.precip_total_surf_mass_flux.daily_AVERAGE.ne30pg2.20160807_mahf708.nc' #maybe check this
DY1_LW_obs_file = DY1_obs_dir + 'CERES.LW_flux_up_at_model_top.daily_AVERAGE.ne30pg2.20160807_mahf708.nc'
#DY1_SW_dn_obs_file = DY1_obs_dir + 'CERES.SW_flux_dn_at_model_top.daily_AVERAGE.ne30pg2.20160807_mahf708.nc'
DY1_SW_up_obs_file = DY1_obs_dir + 'CERES.SW_flux_up_at_model_top.daily_AVERAGE.ne30pg2.20160807_mahf708.nc'
DY1_LWP_file = DY1_obs_dir + 'mac.clwp-tlwp-wvp.20160807.ne30pg2.nc'

DY1_PCP_obs = xr.open_dataset(DY1_precip_obs_file).variables['precip_total_surf_mass_flux'].squeeze('time')
DY1_TLWP_obs = (xr.open_dataset(DY1_LWP_file).variables['tlwp']*1e-3).squeeze('time')
DY1_OSR_obs = xr.open_dataset(DY1_SW_up_obs_file).variables['SW_flux_up_at_model_top'].squeeze('time')
DY1_OLR_obs = xr.open_dataset(DY1_LW_obs_file).variables['LW_flux_up_at_model_top'].squeeze('time')


# In[ ]:


DY2_obs_dir = '/global/cfs/projectdirs/e3smdata/simulations/SCREAM.2024-autocal-00.ne1024pg2/obs/'
DY2_precip_obs_file = DY2_obs_dir + 'IMERG.precip_total_surf_mass_flux.AVERAGE.ne30pg2.20200126.nc'
DY2_LW_obs_file = DY2_obs_dir + 'CERES.LW_flux_up_at_model_top.AVERAGE.ne30pg2.20200126.nc'
#DY2_SW_dn_obs_file = DY2_obs_dir + 'CERES.SW_flux_dn_at_model_top.AVERAGE.ne30pg2.20200126.nc'
DY2_SW_up_obs_file = DY2_obs_dir + 'CERES.SW_flux_up_at_model_top.AVERAGE.ne30pg2.20200126.nc'
DY2_LWP_file = DY2_obs_dir + 'mac.clwp-tlwp-wvp.20200126.ne30pg2.nc'

DY2_PCP_obs = xr.open_dataset(DY2_precip_obs_file).variables['precip_total_surf_mass_flux'].squeeze('time')
DY2_TLWP_obs = (xr.open_dataset(DY2_LWP_file).variables['tlwp']*1e-3).squeeze('time')
DY2_OSR_obs = xr.open_dataset(DY2_SW_up_obs_file).variables['SW_flux_up_at_model_top'].squeeze('time')
DY2_OLR_obs = xr.open_dataset(DY2_LW_obs_file).variables['LW_flux_up_at_model_top'].squeeze('time')

Units
SCREAM
float precip_total_surf_mass_flux(time, ncol) ;
		precip_total_surf_mass_flux:units = "m s^-1" ;  #*1e3*24*3600 --- this is a conversion Hassan had; 1000 meters / day --- should maybe be kg 1e-3?
		precip_total_surf_mass_flux:long_name = "precip_total_surf_mass_flux" ;
		precip_total_surf_mass_flux:_FillValue = 3.402824e+33f ;
		precip_total_surf_mass_flux:averaging_count_tracker = "avg_count_ncol" ;
float LiqWaterPath(time, ncol) ;
		LiqWaterPath:units = "m^-2 kg" ;
		LiqWaterPath:long_name = "LiqWaterPath" ;
		LiqWaterPath:_FillValue = 3.402824e+33f ;
		LiqWaterPath:averaging_count_tracker = "avg_count_ncol" ;
float RainWaterPath(time, ncol) ;
		RainWaterPath:units = "m^-2 kg" ;
		RainWaterPath:long_name = "RainWaterPath" ;
		RainWaterPath:_FillValue = 3.402824e+33f ;
		RainWaterPath:averaging_count_tracker = "avg_count_ncol" ;
float SW_flux_up_at_model_top(time, ncol) ;
		SW_flux_up_at_model_top:units = "W/m2" ;
		SW_flux_up_at_model_top:long_name = "SW_flux_up_at_model_top" ;
		SW_flux_up_at_model_top:_FillValue = 3.402824e+33f ;
		SW_flux_up_at_model_top:averaging_count_tracker = "avg_count_ncol" ;
float LW_flux_up_at_model_top(time, ncol) ;
		LW_flux_up_at_model_top:units = "W/m2" ;
		LW_flux_up_at_model_top:long_name = "LW_flux_up_at_model_top" ;
		LW_flux_up_at_model_top:_FillValue = 3.402824e+33f ;
		LW_flux_up_at_model_top:averaging_count_tracker = "avg_count_ncol" ;Units
Obs
float precip_total_surf_mass_flux(time, ncol) ;
		precip_total_surf_mass_flux:units = "m s^-1" ;
		precip_total_surf_mass_flux:DimensionNames = "time,lon,lat" ;
		precip_total_surf_mass_flux:Units = "mm/hr" ;
		precip_total_surf_mass_flux:CodeMissingValue = "-9999.9" ;
		precip_total_surf_mass_flux:cell_methods = "time: mean" ;
		precip_total_surf_mass_flux:missing_value = -9999.9f ;
		precip_total_surf_mass_flux:_FillValue = -9999.9f ;
		precip_total_surf_mass_flux:coordinates = "lat lon" ;
		precip_total_surf_mass_flux:cell_measures = "area: area" ;
float tlwp(time, ncol) ;
		tlwp:units = "g/m2" ; #*1e-3 --- THIS IS THE NEEDED COEFFICIENT!!
		tlwp:long_name = "Total Liquid (Cloud+Rain) Water Path (TLWP)" ;
		tlwp:_FillValue = -9999.99f ;
		tlwp:cell_methods = "time: mean" ;
		tlwp:coordinates = "lat lon" ;
		tlwp:cell_measures = "area: area" ;
float SW_flux_up_at_model_top(time, ncol) ;
		SW_flux_up_at_model_top:long_name = "Observed Top of the Atmosphere Shortwave Flux, All-sky conditions, Hourly Daily Means" ;
		SW_flux_up_at_model_top:standard_name = "Observed TOA Shortwave Flux - All-sky" ;
		SW_flux_up_at_model_top:units = "W m-2" ;
		SW_flux_up_at_model_top:valid_min = "       0" ;
		SW_flux_up_at_model_top:valid_max = "    1400" ;
		SW_flux_up_at_model_top:cell_methods = "time: mean" ;
		SW_flux_up_at_model_top:missing_value = -999.f ;
		SW_flux_up_at_model_top:_FillValue = -999.f ;
		SW_flux_up_at_model_top:coordinates = "lat lon" ;
		SW_flux_up_at_model_top:cell_measures = "area: area" ;
float LW_flux_up_at_model_top(time, ncol) ;
		LW_flux_up_at_model_top:long_name = "Observed Top of the Atmosphere Longwave Flux, All-sky conditions, Hourly Daily Means" ;
		LW_flux_up_at_model_top:standard_name = "Observed TOA Longwave Flux - All-sky" ;
		LW_flux_up_at_model_top:units = "W m-2" ;
		LW_flux_up_at_model_top:valid_min = "       0" ;
		LW_flux_up_at_model_top:valid_max = "     500" ;
		LW_flux_up_at_model_top:cell_methods = "time: mean" ;
		LW_flux_up_at_model_top:missing_value = -999.f ;
		LW_flux_up_at_model_top:_FillValue = -999.f ;
		LW_flux_up_at_model_top:coordinates = "lat lon" ;
		LW_flux_up_at_model_top:cell_measures = "area: area" ;
# ### Preprocessing the data
# In the longer file on surrogate training, there are more details about loading in the data using xarray, masking to where observations are available, taking geographic averages, and normalizing the data to reduce impact of scale of variables. It also covers k-fold cross validation and different model implimentations.

# ### Trying to save the model
# I have tried a number of ways to save the GP model: using pickle, checkpoint, and saved model. Have yet to find a successful way that works with the ESEm wrapper
# 
# In the absence of a saved model, I saved preprocessing transformations and processes/unprocessed data for quick retraining. (The dataset is quite small so this is a sufficient solution for the moment.)

# In[2]:


model_gp.model.model


# ##### Saving with pickle

# In[37]:


# The sklearn model is held internally in the esem model
with open("trying_to_save_this_model.pkl","wb") as f:
    pickle.dump(model_gp.model.model, f)


# In[38]:


with open("trying_to_save_this_model.pkl","rb") as f:
    skmodel=pickle.load(f)


# ##### Saving with Saved Model

# In[44]:


save_dir = '/global/cfs/cdirs/e3sm/jpaige3/ESEm/TF_saving/'
save_name = save_dir + 'GPFlow_save' + str(date.today()) + '/'
tf.saved_model.save(model_gp.model.model, save_name)


# In[46]:


loaded_model = tf.saved_model.load(save_dir+save_name)


# In[79]:


#recreating
loaded_model = tf.saved_model.load(save_name)
model_gp = loaded_model


# ##### Saving with Checkpoint

# In[48]:


#creates a checkpoint at the final trained stage
checkpoint = tf.train.Checkpoint(model=model_gp.model.model)

#saves this checkpoint to GPFlow_save_ckpt
checkpoint_dir = '/global/cfs/cdirs/e3sm/jpaige3/ESEm/TF_checkpoint_saving/'
save_date = str(date.today())
checkpoint_savename = checkpoint_dir + 'GPFlow_save_ckpt' + save_date + 'Y_ss_wallrts'
checkpoint.save(file_prefix=checkpoint_savename)


# In[5]:


#recreating
#model_gp_recreate = gp_model(X_train_norm, Y_train_norm) #create an instance of the model
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)) #recreate


# #### Saving the transforms

# In[66]:


### Saving projections both normed and not and R2s
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#path - /global/cfs/cdirs/e3sm/jpaige3/ESEm/TF_saving/ --- REPLACE THIS PATH TO SAVE THE PIPELINE WITH THE MODEL
GP_proj_filename = f"/global/cfs/cdirs/e3sm/jpaige3/ESEm/TF_saving/GPFlow_save2025-06-12/GP_ZRG_proj_{timestamp}.pkl"

#make dictionary to save
GP_data_to_save = {
    'X_pipeline': X_pipe_sk_minmax, 
    'Y_pipeline_PCP': Y_pipe_sk_ss_PCP, 
    'Y_pipeline_TLWP': Y_pipe_sk_ss_TLWP, 
    'Y_pipeline_OSR': Y_pipe_sk_ss_OSR, 
    'Y_pipeline_OLR': Y_pipe_sk_ss_OLR,
    ####
    'X_train_index': train_run_labels,
    ### normalized/transformed
    'X_train_norm': X_train_norm, 
    'proj_norm_gp': m_gp,
    'proj_v_norm_gp': v_gp,
    ### unnormalized/untransformed
    'X_train': X_train.to_numpy(), 
    'Y_train': Y_train_ZRG,
    'proj_gp': proj_gp,
    'proj_v_gp': proj_v_gp,
}

#save using pickle
with open(GP_proj_filename, 'wb') as f:
    pickle.dump(GP_data_to_save, f)


# ## Restarting from saved model/data

# Load in regions and define zones to take geographical averages

# In[2]:


regions_file = xr.open_dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/regions.nc')
regions_list = ['poles','extratropical_land','extratropical_ocean','tropical_land','ascending_tropical_ocean','descending_tropical_ocean']
#area = ppe_dataset.area[1,:] #only taking the first row, because all rows should have the same values
control = xr.open_dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/SCREAM.2024-autocal-00.ne1024pg2/m0000/SCREAM.2024-autocal-00.ne1024pg2/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc')
area = control.variables['area'][:]
lat = control.variables['lat'][:]
lon = control.variables['lon'][:]


# In[3]:


def zonal_means_native(data, area, lat, lon): #averages across zones given the unstructured data files
    lat_bands = np.linspace(-90,90,19) #currently dividing globe in 18 zones - 10 degree bands
    zonal_means = dict()
    for i in range(len(lat_bands) - 1):
        mask_zone = (lat > lat_bands[i]) & (lat < lat_bands[i+1]).squeeze()
        data_zone = np.where( mask_zone>0, data.squeeze(), np.nan)
        area_zone = np.where( mask_zone>0, area.squeeze(), np.nan)
        zone_mean = np.nansum(data_zone*area_zone) / np.nansum(area_zone) #note, they are area weighted
        zone_center = abs(lat_bands[i] - lat_bands[i+1])/2 + lat_bands[i]
        zonal_means[zone_center] = zone_mean
    return zonal_means

def regional_means_native(data, area): #averages across regions 
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

def global_means_native(data, area): #takes global averages
    global_mean = np.nanmean(data*area)/np.nanmean(area)
    return global_mean


# In[4]:


#Load back obs
obs_filename = "/global/cfs/cdirs/e3sm/jpaige3/ESEm/TF_saving/obs_2025-07-30_09-40-13.pkl"

with open(obs_filename, 'rb') as f:
    loaded_obs = pickle.load(f)
    
zrg_obs = loaded_obs['zrg_obs']


# In[5]:


n_cols_per_df = zrg_obs.shape[1] // 4 #50 geographic areas for DY1 and DY2
#creates observational datasets for precipitation, total liquid water path, outgoing shortwave radiation, and outgoing longwave radiation
PCP_zrg_obs  = zrg_obs.iloc[:, 0 : n_cols_per_df]
TLWP_zrg_obs = zrg_obs.iloc[:, n_cols_per_df : 2 * n_cols_per_df]
OSR_zrg_obs  = zrg_obs.iloc[:, 2 * n_cols_per_df : 3 * n_cols_per_df]
OLR_zrg_obs  = zrg_obs.iloc[:, 3 * n_cols_per_df : 4 * n_cols_per_df]


# In[6]:


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


# In[7]:


PCP_train = Y_train_ZRG[:, :, 0]
TLWP_train = Y_train_ZRG[:, :, 1]
OSR_train = Y_train_ZRG[:, :, 2]
OLR_train = Y_train_ZRG[:, :, 3]


# The observational dataset and the training dataset (the full ppe in this case) have been loaded in. They are composed of DY1 and DY2 data with zonal, regional, and global data. They are separated by target variable so (for independent normalization).

# In[8]:


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


# In[9]:


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

#note typically these columns are named, but were not saved with names so can throw errors when using the saved data


# ## Training the model

# In[10]:


print(X_train_norm.shape, Y_train_norm.shape) #the first dimension of these must match


# In[11]:


model_gp = gp_model(X_train_norm, Y_train_norm) #creates the model form, but is not trained yet
#this default kernel has been useful (combination of linear, polynomial, and RBF) and outperformed other options individually


# In[12]:


model_gp.train() #training the model--should take only a few seconds with the current data size, could be improved with GPU usage


# In[49]:


range_thl2tune = [0.1, 10]
range_qw2tune = [0.1, 10]
range_length_fac = [0.1, 10]
range_c_diag_3rd_mom = [0.1, 10]
range_Ckh = [0.1, 1]
range_Ckm = [0.1, 1]
range_lambda_low = [0.0001, 0.1]
range_lambda_high = [0.0001, 0.1]
range_spa_to_nc = [0.1, 10]
range_p3_eci = [0.1, 1]
range_p3_eri = [0.1, 1]
range_k_acc = [0.01, 100]
range_p3_dep_nucleation_exponent = [0.2, 0.304]
range_max_total_ni = [5e5, 1e7]
#range_ice_sed_knob = [0.1, 2]
range_ice_sed_knob = [1, 2]
range_p3_d_breakup_cutoff = [0, 500e-6]

dict_range_pars = dict()
dict_range_pars['length_fac'] = range_length_fac
dict_range_pars['p3_spa_to_nc'] = range_spa_to_nc
dict_range_pars['p3_k_accretion'] = range_k_acc
dict_range_pars['p3_ice_sed_knob'] = range_ice_sed_knob
dict_range_pars['thl2tune'] = range_thl2tune
dict_range_pars['qw2tune'] = range_qw2tune
dict_range_pars['c_diag_3rd_mom'] = range_c_diag_3rd_mom
dict_range_pars['Ckh'] = range_Ckh
dict_range_pars['Ckm'] = range_Ckm
dict_range_pars['lambda_low'] = range_lambda_low
dict_range_pars['lambda_high'] = range_lambda_high
dict_range_pars['p3_eci'] = range_p3_eci
dict_range_pars['p3_eri'] = range_p3_eri
dict_range_pars['p3_dep_nucleation_exponent'] = range_p3_dep_nucleation_exponent
dict_range_pars['p3_d_breakup_cutoff'] = range_p3_d_breakup_cutoff
dict_range_pars['max_total_ni'] = range_max_total_ni


# In[100]:


default_cost, params_to_cost(X_pipe_sk_minmax.transform(X_train[0].reshape(1,-1)))


# In[117]:


sample_iter = 50
param_name_1 = 'p3_k_accretion' 
param_name_2 = 'p3_ice_sed_knob'

index_1 = 11
index_2 = 14

p1_range = np.linspace(dict_range_pars[param_name_1][0], dict_range_pars[param_name_1][1], sample_iter)
p2_range = np.linspace(dict_range_pars[param_name_2][0], dict_range_pars[param_name_2][1], sample_iter)

# Fix the other 14 parameters
defaulted_params = ppe_params.iloc[0]

# Prepare grid for plotting
P1, P2 = np.meshgrid(p1_range, p2_range)
Z = np.zeros_like(P1)

# Evaluate model on the grid
for i in range(P1.shape[0]):
    for j in range(P1.shape[1]):
        params = defaulted_params.copy()
        params[param_name_1] = P1[i, j]  # Vary parameter 1
        params[param_name_2] = P2[i, j]  # Vary parameter 2
        Z[i,j] = params_to_cost(X_pipe_sk_minmax.transform(params.to_numpy().reshape(1,-1)))

# Plot as heatmap
plt.figure(figsize=(8, 6))
plt.contourf(P1, P2, Z, levels=50, cmap='bwr')
plt.colorbar(label='Cost')
plt.scatter(X_train[0][index_1], X_train[0][index_2], color = 'black', s=50, label='Default') #color='black', s=50, label='My Point')
plt.xlabel(param_name_1)
plt.ylabel(param_name_2)
plt.title('Cost surface visualized for two parameters')
plt.legend()
plt.show()

# Or as a 3D surface plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[0][index_1], X_train[0][index_2], default_cost, color = 'black', s=50, label='Default')
surf = ax.plot_surface(P1, P2, Z, cmap='bwr')
fig.colorbar(surf, ax=ax, label='Cost')
ax.set_xlabel(param_name_1)
ax.set_ylabel(param_name_2)
ax.set_zlabel('Cost')
plt.title('3D Cost Surface Plot')
plt.legend()
plt.tight_layout()
plt.show()


# ## Optimizing

# ##### All mostly on the same range -- should help with optimizing

# In[13]:


np.min(PCP_train_norm), np.max(PCP_train_norm)


# In[14]:


np.min(TLWP_train_norm), np.max(TLWP_train_norm)


# In[15]:


np.min(OSR_train_norm), np.max(OSR_train_norm)


# In[16]:


np.min(OLR_train_norm), np.max(OLR_train_norm)


# ### Cost function - weights on regions and variables

# In[17]:


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


# In[18]:


lat_bands = np.linspace(-90,90,18) 


# In[19]:


def params_to_cost(params_guess): #, obs, area_weights, var_weights_dict, zrg_weights_dict):
    m_gp_guess, v_gp_guess = model_gp.predict(params_guess.reshape(1, -1))
    cost = ZRG_cost_function_rmse(m_gp_guess, obs_norm, var_weights_dict, zrg_weights_dict) #area_weights,
    return cost

#this function just prints the breakdowns of the cost function
def params_to_cost_print(params_guess): #, obs, area_weights, var_weights_dict, zrg_weights_dict):
    m_gp_guess, v_gp_guess = model_gp.predict(params_guess.reshape(1, -1))
    cost = ZRG_cost_function_rmse_print(m_gp_guess, obs_norm, var_weights_dict, zrg_weights_dict) #area_weights,
    return cost


# In[33]:


#This cost function strictly enforces the constraints on the lambda high and low, seems to be unnecessary as the GP doesn't tend to this space and adds computation
def params_to_cost_full_precision(params_guess): #, obs, area_weights, var_weights_dict, zrg_weights_dict):
    # Enforce parameter constraints (hard-coded for now)
    #if the lambda high and lambda low are not feasible
    
    #CHECK ON IF UNTRANSFORM IS NEEDED OR IF TRANSFORM IS THE SAME FOR BOTH!
    untransformed_params_guess = X_pipe_sk_minmax.inverse_transform(params_guess.reshape(1, -1))[0]
    ### NEED these indices to match to lambda low and lambda high
    if untransformed_params_guess[6] > untransformed_params_guess[7]:
        #if params_guess['lambda_low'] > params_guess['lambda_high']
        return 1e100 #upweight this result
    m_gp_guess, v_gp_guess = model_gp.predict(params_guess.reshape(1, -1))
    cost = ZRG_cost_function_rmse(m_gp_guess, obs_norm, var_weights_dict, zrg_weights_dict) #area_weights,
    return cost

def params_to_cost_explicit(params_guess, obs_norm, var_weights_dict, zrg_weights_dict):
    m_gp_guess, v_gp_guess = model_gp.predict(params_guess.reshape(1, -1))
    cost = ZRG_cost_function_rmse(m_gp_guess, obs_norm, var_weights_dict, zrg_weights_dict) #area_weights,
    return cost
    
def params_to_cost_abs_print(params_guess): #, obs, area_weights, var_weights_dict, zrg_weights_dict):
    m_gp_guess, v_gp_guess = model_gp.predict(params_guess.reshape(1, -1))
    cost = ZRG_cost_function_print(m_gp_guess, obs_norm, var_weights_dict, zrg_weights_dict) #area_weights,
    return cost

def params_to_cost_untransform(params_guess): #, obs, area_weights, var_weights_dict, zrg_weights_dict):
    m_gp_guess, v_gp_guess = model_gp.predict(params_guess.reshape(1, -1))
    
    PCP_proj_untransform_c = Y_pipe_sk_ss_PCP.inverse_transform(m_gp_guess[:, :, 0]) 
    TLWP_proj_untransform_c = Y_pipe_sk_ss_TLWP.inverse_transform(m_gp_guess[:, :, 1])
    OSR_proj_untransform_c = Y_pipe_sk_ss_OSR.inverse_transform(m_gp_guess[:, :, 2])
    OLR_proj_untransform_c = Y_pipe_sk_ss_OLR.inverse_transform(m_gp_guess[:, :, 3])

    proj_untransform = np.stack([PCP_proj_untransform_c, TLWP_proj_untransform_c, OSR_proj_untransform_c, OLR_proj_untransform_c])
    proj_untransform = proj_untransform.transpose(1, 2, 0)

    cost = ZRG_cost_function_untransform(proj_untransform, obs_untransform, var_weights_dict, zrg_weights_dict) #area_weights,
    return cost


# In[20]:


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


# In[32]:


def ZRG_cost_function_rmse_print(preds, obs, var_weights_dict, zrg_weights_dict): #area_weights,
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


    #print('PCP cost', np.mean(total_PCP_cost), 'TLWP cost', np.mean(total_TLWP_cost), 'OSR cost', np.mean(total_OSR_cost), 'OLR cost', np.mean(total_OLR_cost))                       
    print('zonal_cost', (DY1_zonal_cost+DY2_zonal_cost), 'regional_cost', (DY1_regional_cost+DY2_regional_cost), 'global_cost', (DY1_global_cost+DY2_global_cost))
    print('DY1 cost', (DY1_zonal_cost + DY1_regional_cost + DY1_global_cost), 'DY2 cost', (DY2_zonal_cost + DY2_regional_cost + DY2_global_cost))

    cost = DY_weights_dict['DY1']*(DY1_zonal_cost + DY1_regional_cost + DY1_global_cost) + DY_weights_dict['DY2']*(DY2_zonal_cost + DY2_regional_cost + DY2_global_cost)
    print('cost:', cost)
    return cost


# In[34]:


default_cost = ZRG_cost_function_rmse(Y_train_norm[0][np.newaxis, :], obs_norm, var_weights_dict, zrg_weights_dict)
print('True default cost:', default_cost)

print('Predictions:')
pred_default_cost = params_to_cost_print(X_train_norm[0])


# ### Minimizing using Basinhopping
# We tried several techniques for minimizing, and are still interested in ESEm's integrated MCMC and ABC sampler. Currently, running into issues with implimenting it, possibly due to Tensorflow instillation? Several built in features to ESEm might be useful moving forward.

# In[ ]:


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
seed = 42
N_xstarts = 1
rn = np.random.RandomState(seed)
xstarts = rn.rand(N_xstarts, 16)
print(xstarts)

with ThreadPoolExecutor() as executor:
    results = list(executor.map(run_bh, xstarts))
    results = np.vstack(results)

    top10_rows = np.argsort(abs(results[:, -1]))[:10]

    # Save results
    date_str = datetime.now().strftime("%Y-%m-%d")
    csv_filename = f"/global/cfs/cdirs/e3sm/jpaige3/optimizing/Optimizing_results/results{N_xstarts}_{seed}_{date_str}.csv"
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


# This is an example of a successful run with Basinhopping

# In[27]:


from concurrent.futures import ThreadPoolExecutor
print('all start')
start_1 = timer()

def run_scimin(xstart):
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

    minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
    
    print('result', result)
    end = timer()
    print(f"time: {end - start:.4f} seconds")
    return np.hstack((result.x, result.fun))

# List of starting points
seed = 54
N_xstarts = 10
rn = np.random.RandomState(seed)
xstarts = rn.rand(N_xstarts, 16)
print(xstarts)

with ThreadPoolExecutor() as executor:
    results = list(executor.map(run_bh, xstarts))
    results = np.vstack(results)

    top10_rows = np.argsort(abs(results[:, -1]))[:10]

    # Save results
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = f"/global/cfs/cdirs/e3sm/jpaige3/optimizing/Optimizing_results/results{N_xstarts}_{seed}_{date_str}.csv"
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

Example Output:
start
result                     message: ['requested number of basinhopping iterations completed successfully']
                    success: True
                        fun: 1.6527049790005854
                          x: [ 0.000e+00  6.142e-01 ...  6.157e-01
                               2.511e-01]
                        nit: 1
      minimization_failures: 0
                       nfev: 2958
                       njev: 174
 lowest_optimization_result:  message: CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH
                              success: True
                               status: 0
                                  fun: 1.6527049790005854
                                    x: [ 0.000e+00  6.142e-01 ...
                                         6.157e-01  2.511e-01]
                                  nit: 53
                                  jac: [ 5.022e-03 -3.489e-02 ...
                                         5.905e-02  1.793e-03]
                                 nfev: 1887
                                 njev: 111
                             hess_inv: <16x16 LbfgsInvHessProduct with dtype=float64>
time: 2508.1813 seconds
# ## Validating optimal runs

# In[55]:


### These are the first optimal params run in SCREAM in Oct 2025
optimal_params_final = np.array( [1.00000000e-01, 8.72596066e+00, 5.81553407e-01, 1.00000000e-01,
                                  3.97719941e-01, 1.00000000e+00, 4.20402644e-02, 4.36854065e-02,
                                  1.00000000e-01, 1.00000000e+00, 1.00000000e-01, 1.00000000e-02,
                                  2.93051240e-01, 9.36429137e+06, 1.65530607e+00, 7.70072093e-11, 1])


# In[50]:


#when output from basing hopping, they will be normalized/transformed
optimal_params_norm = X_pipe_sk_minmax.transform(optimal_params_final[0:-1].reshape(1,-1))


# In[51]:


print('default:')
print(params_to_cost_print(X_train_norm[0]))
print('optimal:')
print(params_to_cost_print(optimal_params_norm)) #


# ##### Must transform back before using in run

# In[53]:


final_preds_norm = model_gp.predict(optimal_params_norm.reshape(1,-1))[0]
PCP_final_preds_norm = final_preds_norm[:, :, 0] #this is a numpy array
TLWP_final_preds_norm = final_preds_norm[:, :, 1]
OSR_final_preds_norm = final_preds_norm[:, :, 2]
OLR_final_preds_norm = final_preds_norm[:, :, 3]

PCP_final_preds = Y_pipe_sk_ss_PCP.inverse_transform(PCP_final_preds_norm)[0]
TLWP_final_preds = Y_pipe_sk_ss_TLWP.inverse_transform(TLWP_final_preds_norm)[0]
OSR_final_preds = Y_pipe_sk_ss_OSR.inverse_transform(OSR_final_preds_norm)[0]
OLR_final_preds = Y_pipe_sk_ss_OLR.inverse_transform(OLR_final_preds_norm)[0]

final_params = X_pipe_sk_minmax.inverse_transform(optimal_params_final[0:-1].reshape(1,-1))


# ### Below likely less useful to immediate workflow
# Gathering the saved results from basinhopping output, analyzing the new SCREAM runs, plotting examples

# #### Gathering basinhopping results

# In[26]:


# gather CSV files
opts_file_list = glob.glob("/global/cfs/cdirs/e3sm/jpaige3/optimizing/Optimizing_results/results*.csv")
# read into dataframe
combined_df = pd.concat(
    (pd.read_csv(fname).assign(source_file=fname) for fname in sorted(opts_file_list)),
    ignore_index=True
)
optimals_bh = combined_df.drop(['source_file', 'Rank'], axis=1)
sort_opts_bh = optimals_bh.sort_values(by='cost')
sort_opts_bh['basinhopping'] = np.ones(len(sort_opts_bh))
sort_opts_bh


# In[29]:


# gather CSV files
optscimin_file_list = glob.glob("/global/cfs/cdirs/e3sm/jpaige3/optimizing/Optimizing_results/scimin_results*.csv")
# read into dataframe
combined_df = pd.concat(
    (pd.read_csv(fname).assign(source_file=fname) for fname in sorted(optscimin_file_list)),
    ignore_index=True
)
optimals_sm = combined_df.drop(['source_file', 'Rank'], axis=1)
sort_opts_sm = optimals_sm.sort_values(by='cost')
sort_opts_sm['basinhopping'] = np.zeros(len(sort_opts_sm))
sort_opts_sm


# In[30]:


overall_opts = pd.concat( (sort_opts_bh, sort_opts_sm), axis = 0)
sort_overall_opts = overall_opts.sort_values(by='cost')


# In[200]:


sort_overall_opts


# In[240]:


# gather CSV files
maxopts_file_list = glob.glob("/global/cfs/cdirs/e3sm/jpaige3/optimizing/Optimizing_results/maximizing_results*.csv")
# read into dataframe
combined_df = pd.concat(
    (pd.read_csv(fname).assign(source_file=fname) for fname in sorted(maxopts_file_list)),
    ignore_index=True
)
maxopts_bh = combined_df.drop(['source_file', 'Rank'], axis=1)
maxopts_bh['cost'] = - maxopts_bh['cost']
sort_maxopts_bh = maxopts_bh.sort_values(by='cost')
#sort_maxopts_bh['basinhopping'] = np.ones(len(sort_maxopts_bh))
sort_maxopts_bh.head()


# In[245]:


#combine max and min
max_min_bh = pd.concat((sort_opts_bh, sort_maxopts_bh), axis = 0)
len(sort_maxopts_bh)


# #### Initial optimals

# In[26]:


parameters_names_list = ['thl2tune', 'qw2tune', 'length_fac', 'c_diag_3rd_mom', 'Ckh', 'Ckm',
       'lambda_low', 'lambda_high', 'p3_spa_to_nc', 'p3_eci', 'p3_eri',
       'p3_k_accretion', 'p3_dep_nucleation_exponent', 'max_total_ni',
       'p3_ice_sed_knob', 'p3_d_breakup_cutoff']


# In[374]:


new_default_data = xr.open_dataset('/global/cfs/cdirs/e3smdata/simulations/ecp-autotune/sims-dec3-2024/hh1024/m0000/SCREAM.2024-autocal-00.ne1024pg2/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc')
#/global/cfs/cdirs/e3smdata/simulations/ecp-autotune/sims-dec3-2024/hh1024/m0000/SCREAM.2024-autocal-00.ne1024pg2/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc')
new_default_data_small = new_default_data.drop_vars(to_leave)
new_default_data_small['TotalLiqWaterPath'] = (new_default_data_small.LiqWaterPath + new_default_data_small.RainWaterPath)
new_default_data_small.squeeze('time')


# In[417]:


ppe_dataset_small


# ##### New default analysis

# In[440]:


### Grid cell rmse
DY2_PCP_default_error = np.ones(len(sim_names))
DY2_TLWP_default_error = np.ones(len(sim_names))
DY2_OSR_default_error = np.ones(len(sim_names))
DY2_OLR_default_error = np.ones(len(sim_names))

for runn in range(len(sim_names)):
    run_lab = sim_names[runn]
    DY2_PCP_default_error[runn] = rmse(new_default_data_small.precip_total_surf_mass_flux[0], DY2_ppe_dataset_small.sel(run_label=run_lab).DY2_precip_total_surf_mass_flux)
    DY2_TLWP_default_error[runn] = rmse(new_default_data_small.TotalLiqWaterPath[0], DY2_ppe_dataset_small.sel(run_label=run_lab).DY2_TotalLiqWaterPath)
    DY2_OSR_default_error[runn] = rmse(new_default_data_small.SW_flux_up_at_model_top[0], DY2_ppe_dataset_small.sel(run_label=run_lab).DY2_SW_flux_up_at_model_top)
    DY2_OLR_default_error[runn] = rmse(new_default_data_small.LW_flux_up_at_model_top[0], DY2_ppe_dataset_small.sel(run_label=run_lab).DY2_LW_flux_up_at_model_top)
    
DY2_final_default_error = DY2_PCP_default_error+DY2_TLWP_default_error+DY2_OSR_default_error+DY2_OLR_default_error

print('Argmin PCP:', sim_names[np.argmin(DY2_PCP_default_error)])
print('Argmin TLWP:', sim_names[np.argmin(DY2_TLWP_default_error)])
print('Argmin OSR:', sim_names[np.argmin(DY2_OSR_default_error)])
print('Argmin OLR:', sim_names[np.argmin(DY2_OLR_default_error)])

print('Final Argmin:', sim_names[np.argmin(DY2_final_default_error)])


# In[ ]:


dict1 = zonal_means_native(optimal_run_data_small.precip_total_surf_mass_flux, area, lat, lon)
dict2 = regional_means_native(optimal_run_data_small.precip_total_surf_mass_flux, area)
array = np.array([global_means_native(optimal_run_data_small.precip_total_surf_mass_flux, area)])


# In[ ]:


### ZRG abs error
DY2_PCP_default_zrg_error = np.ones(len(sim_names))
DY2_TLWP_default_zrg_error = np.ones(len(sim_names))
DY2_OSR_default_zrg_error = np.ones(len(sim_names))
DY2_OLR_default_zrg_error = np.ones(len(sim_names))

for runn in range(len(sim_names)):
    run_lab = sim_names[runn]
    DY2_PCP_default_zrg_error[runn] = rmse(new_default_data_small.precip_total_surf_mass_flux[0], DY2_ppe_dataset_small.sel(run_label=run_lab).DY2_precip_total_surf_mass_flux)
    DY2_TLWP_default_zrg_error[runn] = rmse(new_default_data_small.TotalLiqWaterPath[0], DY2_ppe_dataset_small.sel(run_label=run_lab).DY2_TotalLiqWaterPath)
    DY2_OSR_default_zrg_error[runn] = rmse(new_default_data_small.SW_flux_up_at_model_top[0], DY2_ppe_dataset_small.sel(run_label=run_lab).DY2_SW_flux_up_at_model_top)
    DY2_OLR_default_zrg_error[runn] = rmse(new_default_data_small.LW_flux_up_at_model_top[0], DY2_ppe_dataset_small.sel(run_label=run_lab).DY2_LW_flux_up_at_model_top)
    
DY2_final_default_error = DY2_PCP_default_error+DY2_TLWP_default_error+DY2_OSR_default_error+DY2_OLR_default_error

print('Argmin PCP:', sim_names[np.argmin(DY2_PCP_default_error)])
print('Argmin TLWP:', sim_names[np.argmin(DY2_TLWP_default_error)])
print('Argmin OSR:', sim_names[np.argmin(DY2_OSR_default_error)])
print('Argmin OLR:', sim_names[np.argmin(DY2_OLR_default_error)])

print('Final Argmin:', sim_names[np.argmin(DY2_final_default_error)])


# In[56]:


#This is just DY2 data
optimal_run_data = xr.open_dataset('/global/cfs/cdirs/e3smdata/simulations/ecp-autotune/sims-dec3-2024/hh1024/optdec3/SCREAM.2024-autocal-00.ne1024pg2/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc')


# In[58]:


#filter variables in output - keep only the 4 of interest
to_keep = ['precip_total_surf_mass_flux','LiqWaterPath','RainWaterPath','SW_flux_up_at_model_top','LW_flux_up_at_model_top']
           
to_leave = ['SW_flux_dn','SW_flux_dn_at_model_bot','SW_flux_up','SW_flux_up_at_model_bot','SW_flux_dn_at_model_top', 'T_2m',
'T_mid', 'precip_ice_surf_mass_flux','precip_liq_surf_mass_flux','ps','qc','qi','qm','qr','qv','qv_2m','LW_flux_up','LW_flux_up_at_model_bot',
'IceWaterPath', 'LW_flux_dn','LW_flux_dn_at_model_bot','time_bnds', 'LongwaveCloudForcing', 'MeridionalVapFlux','ShortwaveCloudForcing','U','V', 
'VapWaterPath', 'ZonalVapFlux','bm','eddy_diff_mom', 'eff_radius_qc_at_cldtop','eff_radius_qi_at_cldtop', 'homme_T_mid_tend', 'homme_qv_tend',
'horiz_winds_at_model_bot', 'nc', 'ni', 'nr', 'omega', 'p3_T_mid_tend', 'p3_qv_tend', 'rrtmgp_T_mid_tend', 'sgs_buoy_flux', 'shoc_T_mid_tend',
'shoc_qv_tend', 'surf_evap', 'surf_mom_flux', 'surf_radiative_T','surf_sens_flux', 'surface_upward_latent_heat_flux', 'avg_count_ncol', 
'avg_count_ncol_lev', 'avg_count_ncol_dim', 'area', 'lat', 'lon']


# In[57]:


#This is just DY2
optimal_run_data
optimal_run_data_small = optimal_run_data.drop_vars(to_leave)
optimal_run_data_small['TotalLiqWaterPath'] = (optimal_run_data_small.LiqWaterPath + optimal_run_data_small.RainWaterPath)
optimal_run_data_small.squeeze('time')


# #### Plotting

# ##### Comparison plots

# In[456]:


plt.figure(figsize=(8, 4))
point_size = 30

dict1 = zonal_means_native(TLWP_opt, area, lat, lon)
dict2 = regional_means_native(TLWP_opt, area)
array = np.array([global_means_native(TLWP_opt, area)])

# Extract keys and values from dictionaries
keys1, values1 = list(dict1.keys()), list(dict1.values())
keys2, values2 = list(dict2.keys()), list(dict2.values())

# Create x positions
x1 = range(len(keys1))  # x positions for dict1
x2 = range(len(keys1), len(keys1) + len(keys2))  # x positions for dict2
x3 = range(len(keys1) + len(keys2), len(keys1) + len(keys2) + len(array))  # x positions for array
array_labels = ['global'] # label for global array

#Scatter each row of zrg ppe dataset
for row_name, row_values in DY2_TLWP_zrg_ppedataset.iterrows():
    plt.scatter(range(all_num), (row_values-DY2_TLWP_zrg_obs), color='gray', alpha=0.25, s=point_size-10)
    
plt.axhline(y=0, color="black", linewidth=1, zorder=1, alpha=0.7) #linestyle=":",

#plt.scatter(range(all_num), (TLWP_default_values-DY2_TLWP_zrg_obs.iloc[:,0:all_num]), label="Default", marker = '^', color='orange', s=point_size)
plt.plot(range(all_num), (TLWP_default_values-DY2_TLWP_zrg_obs.iloc[:,0:all_num]).to_numpy()[0], '-o', color='orange', label="Default")

# Plot scatter points - simulation (we want DY2, so the second half of the dataset
#plt.scatter(x1, (values1-DY2_TLWP_zrg_obs.iloc[:,0:z_num]), label="Opt simulation", edgecolor="green", facecolors="green", s=point_size)
#plt.scatter(x2, (values2-DY2_TLWP_zrg_obs.iloc[:,(z_num):(z_num+r_num)]), edgecolor="green", facecolors="green", s=point_size)
#plt.scatter(x3, (array-DY2_TLWP_zrg_obs.iloc[:,-1]), edgecolor="green", facecolors="green", s=point_size)
plt.plot(x1, (values1-DY2_TLWP_zrg_obs.iloc[:,0:z_num]).to_numpy()[0], '-^', label="Opt simulation", color="green")
plt.plot(x2, (values2-DY2_TLWP_zrg_obs.iloc[:,(z_num):(z_num+r_num)]).to_numpy()[0], '-^', color="green")
plt.plot(x3, (array-DY2_TLWP_zrg_obs.iloc[:,-1]).to_numpy()[0], '-^', color="green")

#plt.scatter(range(len(TLWP_final_preds)), TLWP_final_preds, label="Opt ZRG pred", marker = 's', edgecolors="blue", facecolors="none", s=point_size)
#plt.scatter(range(len(TLWP_final_preds)), TLWP_gridcell, label="Opt gridcell pred", color='purple', s=point_size)
#plt.scatter(range(all_num), TLWP_default_values, label="Default", marker = 'x', color="orange", s=point_size)
#plt.scatter(range(all_num), DY2_TLWP_zrg_obs, label="Obs", marker = '^', edgecolor='red', facecolors="none", s=point_size)


# Create combined x-tick labels and positions
all_keys = keys1 + keys2 + array_labels
all_positions = list(x1) + list(x2) + list(x3)

# Customize x-axis
plt.xticks(all_positions, all_keys, rotation=45, ha='right')

plt.xlabel("           Center of lattitude band                             Region               Global")
plt.ylabel("Target variable error")
plt.title("TLWP comparison plot")
plt.legend(fontsize=8)
#plt.tight_layout()
plt.show()


# In[190]:


plt.figure(figsize=(8, 4))
point_size = 20

dict1 = zonal_means_native(TLWP_opt, area, lat, lon)
dict2 = regional_means_native(TLWP_opt, area)
array = np.array([global_means_native(TLWP_opt, area)])

# Extract keys and values from dictionaries
keys1, values1 = list(dict1.keys()), list(dict1.values())
keys2, values2 = list(dict2.keys()), list(dict2.values())

# Create x positions
x1 = range(len(keys1))  # x positions for dict1
x2 = range(len(keys1), len(keys1) + len(keys2))  # x positions for dict2
x3 = range(len(keys1) + len(keys2), len(keys1) + len(keys2) + len(array))  # x positions for array
array_labels = ['global'] # label for global array

# Plot scatter points - simulation 
plt.scatter(x1, values1, label="Opt simulation", edgecolor="green", facecolors="none", s=point_size)
plt.scatter(x2, values2, edgecolor="green", facecolors="none", s=point_size)
plt.scatter(x3, array, edgecolor="green", facecolors="none", s=point_size)

plt.scatter(range(len(TLWP_final_preds)), TLWP_final_preds, label="Opt ZRG pred", marker = 's', edgecolors="blue", facecolors="none", s=point_size)
#plt.scatter(range(len(TLWP_final_preds)), TLWP_gridcell, label="Opt gridcell pred", color='purple', s=point_size)
#plt.scatter(range(len(TLWP_final_preds)), TLWP_default_values, label="Default", marker = 'x', color="orange", s=point_size)

plt.scatter(range(len(TLWP_final_preds)), zrg_obs.iloc[:, 25:50], label="Obs", marker = '^', edgecolor='red', facecolors="none", s=point_size)

# Create combined x-tick labels and positions
all_keys = keys1 + keys2 + array_labels
all_positions = list(x1) + list(x2) + list(x3)

# Customize x-axis
plt.xticks(all_positions, all_keys, rotation=45, ha='right')

plt.xlabel("            Center of lattitude band                             Region               Global")
plt.ylabel("Target variable value")
plt.title("TLWP comparison plot")
plt.legend(fontsize=8)
#plt.tight_layout()
plt.show()


# In[194]:


plt.figure(figsize=(8, 4))
point_size = 20

dict1 = zonal_means_native(OLR_opt, area, lat, lon)
dict2 = regional_means_native(OLR_opt, area)
array = np.array([global_means_native(OLR_opt, area)])

# Extract keys and values from dictionaries
keys1, values1 = list(dict1.keys()), list(dict1.values())
keys2, values2 = list(dict2.keys()), list(dict2.values())

# Create x positions
x1 = range(len(keys1))  # x positions for dict1
x2 = range(len(keys1), len(keys1) + len(keys2))  # x positions for dict2
x3 = range(len(keys1) + len(keys2), len(keys1) + len(keys2) + len(array))  # x positions for array
array_labels = ['global'] # label for global array

# Plot scatter points - simulation 
plt.scatter(x1, values1, label="Opt simulation", edgecolor="green", facecolors="none", s=point_size)
plt.scatter(x2, values2, edgecolor="green", facecolors="none", s=point_size)
plt.scatter(x3, array, edgecolor="green", facecolors="none", s=point_size)

#plt.scatter(range(len(OLR_final_preds)), OLR_final_preds, label="Opt ZRG pred", marker = 's', edgecolors="blue", facecolors="none", s=point_size)
#plt.scatter(range(len(OLR_final_preds)), OLR_gridcell, label="Opt gridcell pred", color='purple', s=point_size)
plt.scatter(range(len(OLR_final_preds)), OLR_default_values, label="Default", marker = 'x', color="orange", s=point_size)

plt.scatter(range(len(OLR_final_preds)), zrg_obs.iloc[:, 75:100], label="Obs", marker = '^', edgecolor='red', facecolors="none", s=point_size)

# Create combined x-tick labels and positions
all_keys = keys1 + keys2 + array_labels
all_positions = list(x1) + list(x2) + list(x3)

# Customize x-axis
plt.xticks(all_positions, all_keys, rotation=45, ha='right')

plt.xlabel("            Center of lattitude band                             Region               Global")
plt.ylabel("Target variable value")
plt.title("OLR comparison plot")
plt.legend(fontsize=8)
#plt.tight_layout()
plt.show()

