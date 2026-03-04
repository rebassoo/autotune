#!/usr/bin/env python
# coding: utf-8

# # Abbreviated surrogate training workflow
# 
# The below is implimented in a python file '/global/cfs/cdirs/e3sm/jpaige3/optimizing/run_GPsurrogate_fromsave.py'
# 
# The documentation for ESEm is at https://esem.readthedocs.io/en/latest/

# In[1]:


import os
import json
import math
import pandas as pd
import numpy as np
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
from sklearn.metrics import mean_absolute_error as mae
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


# In[2]:


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


# In[3]:


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
# ## Restarting from saved model/data

# Load in regions and define zones to take geographical averages

# In[4]:


regions_file = xr.open_dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/regions.nc')
regions_list = ['poles','extratropical_land','extratropical_ocean','tropical_land','ascending_tropical_ocean','descending_tropical_ocean']
#area = ppe_dataset.area[1,:] #only taking the first row, because all rows should have the same values
control = xr.open_dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/SCREAM.2024-autocal-00.ne1024pg2/m0000/SCREAM.2024-autocal-00.ne1024pg2/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc')
area = control.variables['area'][:]
lat = control.variables['lat'][:]
lon = control.variables['lon'][:]


# In[5]:


def zonal_means_native(data, area, lat, lon):
    data = data.squeeze()
    area = area.squeeze()
    lat = lat.squeeze()
    lat_bands = np.linspace(-90, 90, 19) #currently dividing globe in 18 zones via 19 borders - 10 degree bands
    zonal_means = dict()
    for i in range(len(lat_bands) - 1):
        mask_zone = (lat >= lat_bands[i]) & (lat < lat_bands[i+1]) #includes lower bound in the band
        data_zone = np.where(mask_zone, data, np.nan)
        area_zone = np.where(mask_zone, area, np.nan)
        zone_mean = np.nansum(data_zone * area_zone) / np.nansum(area_zone)
        zone_center = lat_bands[i] + (lat_bands[i+1] - lat_bands[i])/2
        zonal_means[zone_center] = zone_mean
    return zonal_means

def regional_means_native(data, area, region_data):
    #region_data = xr.open_dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/regions.nc') #for performance improvement move region_data out as argument
    data = data.squeeze()
    area = area.squeeze()
    regions_list = ['poles', 'extratropical_land', 'extratropical_ocean', 
                    'tropical_land', 'ascending_tropical_ocean', 'descending_tropical_ocean']
    region_means = dict()
    for reg_name in regions_list:
        mask_reg = region_data[reg_name].squeeze()
        data_reg = np.where(mask_reg > 0, data, np.nan)
        area_reg = np.where(mask_reg > 0, area, np.nan)
        reg_mean = np.nansum(data_reg * area_reg) / np.nansum(area_reg)
        region_means[reg_name] = reg_mean
    return region_means

def global_means_native(data, area): #takes global averages
    global_mean = np.nansum(data*area)/np.nansum(area)
    return global_mean


# In[6]:


#load back observations
obs_filename = "/global/cfs/cdirs/e3sm/jpaige3/ESEm/GP_Saved_Model_Data/obs_2026-03-01_13-10-41.pkl"

with open(obs_filename, 'rb') as f:
    loaded_obs = pickle.load(f)
    
zrg_obs = loaded_obs['zrg_obs']
PCP_zrg_obs = loaded_obs['PCP_zrg_obs']
TLWP_zrg_obs = loaded_obs['TLWP_zrg_obs']
OSR_zrg_obs = loaded_obs['OSR_zrg_obs']
OLR_zrg_obs = loaded_obs['OLR_zrg_obs']
n_cols_per_df = zrg_obs.shape[1] // 4 #50 DY1 and DY2


# In[7]:


# Load back GP proj
GP_proj_filename = "/global/cfs/cdirs/e3sm/jpaige3/ESEm/GP_Saved_Model_Data/GP_ZRG_masked_proj_2026-03-01_13-10-41.pkl" #has masks properly and norms to full parameter ranges

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


# In[8]:


print(X_train_norm.shape, Y_train_norm.shape)


# The observational dataset and the training dataset (the full ppe in this case) have been loaded in. They are composed of DY1 and DY2 data with zonal, regional, and global data. They are separated by target variable so (for independent normalization).

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

obs_untransform = np.stack([PCP_zrg_obs, TLWP_zrg_obs, OSR_zrg_obs, OLR_zrg_obs])
obs_untransform = obs_untransform.transpose(1, 2, 0)
obs_untransform.shape


# ## Training the model

# In[10]:


print(X_train_norm.shape, Y_train_norm.shape) #the first dimension of these must match


# In[11]:


model_gp = gp_model(X_train_norm, Y_train_norm) #creates the model form, but is not trained yet
#this default kernel has been useful (combination of linear, polynomial, and RBF) and outperformed other options individually


# In[12]:


model_gp.train() #training the model--should take only a few seconds with the current data size, could be improved with GPU usage


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

zonal_weights = [1,1,1,1,1,1,1,1,1,
                 1,1,1,1,1,1,1,1,1] #18 regions, 10 degrees latititude each
regional_weights = [1,1,1,
                   1,1,1] #6 regions


# In[18]:


lat_bands = np.linspace(-90,90,19) 


# In[19]:


lamlow_index = X_train.columns.get_loc('lambda_low')
lamhigh_index = X_train.columns.get_loc('lambda_high')


# In[20]:


def params_to_cost(params_guess): #, obs, area_weights, var_weights_dict, zrg_weights_dict):
    #if params_guess[6] > params_guess[7]:
    #if params_guess['lambda_low'] > params_guess['lambda_high']
    violation = params_guess[lamlow_index] > params_guess[lamhigh_index]  # lambda_low - lambda_high
    if violation > 0:
        return 1e2 + 1e2 * violation  # linear penalty: gets worse the more you violate
    m_gp_guess, v_gp_guess = model_gp.predict(params_guess.reshape(1, -1))
    cost = ZRG_cost_function_mae_weighted(m_gp_guess, obs_norm, var_weights_dict, DY_weights_dict, zrg_weights_dict, zonal_weights, regional_weights)
    return cost

def params_to_cost_print(params_guess): #, obs, area_weights, var_weights_dict, zrg_weights_dict):
    #if params_guess[6] > params_guess[7]:
    #if params_guess['lambda_low'] > params_guess['lambda_high']
    violation = params_guess[lamlow_index] > params_guess[lamhigh_index]
    #violation = params_guess[6] - params_guess[7]  # lambda_low - lambda_high
    if violation > 0:
        return 1e2 + 1e2 * violation  # linear penalty: gets worse the more you violate
    m_gp_guess, v_gp_guess = model_gp.predict(params_guess.reshape(1, -1))
    cost = ZRG_cost_function_mae_weighted_print(m_gp_guess, obs_norm, var_weights_dict, DY_weights_dict, zrg_weights_dict, zonal_weights, regional_weights)
    return cost


# In[115]:


def params_to_cost_print_rmse(params_guess): #, obs, area_weights, var_weights_dict, zrg_weights_dict):
    #if params_guess[6] > params_guess[7]:
    #if params_guess['lambda_low'] > params_guess['lambda_high']
    violation = params_guess[lamlow_index] > params_guess[lamhigh_index]
    #violation = params_guess[6] - params_guess[7]  # lambda_low - lambda_high
    if violation > 0:
        return 1e2 + 1e2 * violation  # linear penalty: gets worse the more you violate
    m_gp_guess, v_gp_guess = model_gp.predict(params_guess.reshape(1, -1))
    cost = ZRG_cost_function_rmse_weighted_print(m_gp_guess, obs_norm, var_weights_dict, DY_weights_dict, zrg_weights_dict, zonal_weights, regional_weights)
    return cost


# In[ ]:


ZRG_cost_function_rmse_weighted_print


# #### Cost functions
# There are several versions of this. The original implimentation of rmse was treating all inputs as outputs from the same sample, leading to it producing an mae output. There is a corrected rmse implimentation. There is also a weighted mae version

# In[21]:


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


# In[22]:


def ZRG_cost_function_mae_weighted_print(preds, obs, var_weights_dict, DY_weights_dict, zrg_weights_dict, zonal_weights = None, regional_weights = None): #area_weights,  
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

    #print('PCP cost', np.mean(total_PCP_cost), 'TLWP cost', np.mean(total_TLWP_cost), 'OSR cost', np.mean(total_OSR_cost), 'OLR cost', np.mean(total_OLR_cost))                       
    print('zonal_cost', (DY1_zonal_cost+DY2_zonal_cost), 'regional_cost', (DY1_regional_cost+DY2_regional_cost), 'global_cost', (DY1_global_cost+DY2_global_cost))
    print('DY1 cost', (DY1_zonal_cost + DY1_regional_cost + DY1_global_cost), 'DY2 cost', (DY2_zonal_cost + DY2_regional_cost + DY2_global_cost))

    cost = DY_weights_dict['DY1']*(DY1_zonal_cost + DY1_regional_cost + DY1_global_cost) + DY_weights_dict['DY2']*(DY2_zonal_cost + DY2_regional_cost + DY2_global_cost)
    print('cost:', cost)
    
    return cost


# In[109]:


## computing RMSE
def ZRG_cost_function_rmse_weighted(preds, obs, var_weights_dict, DY_weights_dict, zrg_weights_dict, zonal_weights = None, regional_weights = None): #area_weights,
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

    DY1_zonal_cost = zrg_weights_dict['zonal']*np.mean([var_weights_dict['PCP']*rmse(PCP_obs_c[0:z_num], PCP_proj_c[0:z_num], sample_weight=zonal_weights),                            
                                                   var_weights_dict['TLWP']*rmse(TLWP_obs_c[0:z_num], TLWP_proj_c[0:z_num], sample_weight=zonal_weights),
                                                   var_weights_dict['OSR']*rmse(OSR_obs_c[0:z_num], OSR_proj_c[0:z_num], sample_weight=zonal_weights),
                                                   var_weights_dict['OLR']*rmse(OLR_obs_c[0:z_num], OLR_proj_c[0:z_num], sample_weight=zonal_weights)])
    DY2_zonal_cost = zrg_weights_dict['zonal']*np.mean([var_weights_dict['PCP']*rmse(PCP_obs_c[(all_num):(all_num+z_num)], PCP_proj_c[(all_num):(all_num+z_num)], sample_weight=zonal_weights),
                                                   var_weights_dict['TLWP']*rmse(TLWP_obs_c[(all_num):(all_num+z_num)], TLWP_proj_c[(all_num):(all_num+z_num)], sample_weight=zonal_weights),
                                                   var_weights_dict['OSR']*rmse(OSR_obs_c[(all_num):(all_num+z_num)], OSR_proj_c[(all_num):(all_num+z_num)], sample_weight=zonal_weights),
                                                   var_weights_dict['OLR']*rmse(OLR_obs_c[(all_num):(all_num+z_num)], OLR_proj_c[(all_num):(all_num+z_num)], sample_weight=zonal_weights)])
    
    DY1_regional_cost = zrg_weights_dict['regional']*np.mean([var_weights_dict['PCP']*rmse(PCP_obs_c[(z_num):(z_num+r_num)], PCP_proj_c[(z_num):(z_num+r_num)], sample_weight=regional_weights),
                                                         var_weights_dict['TLWP']*rmse(TLWP_obs_c[(z_num):(z_num+r_num)], TLWP_proj_c[(z_num):(z_num+r_num)], sample_weight=regional_weights),
                                                         var_weights_dict['OSR']*rmse(OSR_obs_c[(z_num):(z_num+r_num)], OSR_proj_c[(z_num):(z_num+r_num)], sample_weight=regional_weights),
                                                         var_weights_dict['OLR']*rmse(OLR_obs_c[(z_num):(z_num+r_num)], OLR_proj_c[(z_num):(z_num+r_num)], sample_weight=regional_weights)])
    DY2_regional_cost = zrg_weights_dict['regional']*np.mean([var_weights_dict['PCP']*rmse(PCP_obs_c[(all_num+z_num):(all_num+z_num+r_num)], PCP_proj_c[(all_num+z_num):(all_num+z_num+r_num)], sample_weight=regional_weights),
                                                         var_weights_dict['TLWP']*rmse(TLWP_obs_c[(all_num+z_num):(all_num+z_num+r_num)], TLWP_proj_c[(all_num+z_num):(all_num+z_num+r_num)], sample_weight=regional_weights),
                                                         var_weights_dict['OSR']*rmse(OSR_obs_c[(all_num+z_num):(all_num+z_num+r_num)], OSR_proj_c[(all_num+z_num):(all_num+z_num+r_num)], sample_weight=regional_weights),
                                                         var_weights_dict['OLR']*rmse(OLR_obs_c[(all_num+z_num):(all_num+z_num+r_num)], OLR_proj_c[(all_num+z_num):(all_num+z_num+r_num)], sample_weight=regional_weights)])
    
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


# In[110]:


## computing RMSE
def ZRG_cost_function_rmse_weighted_print(preds, obs, var_weights_dict, DY_weights_dict, zrg_weights_dict, zonal_weights = None, regional_weights = None): #area_weights,
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

    DY1_zonal_cost = zrg_weights_dict['zonal']*np.mean([var_weights_dict['PCP']*rmse(PCP_obs_c[0:z_num], PCP_proj_c[0:z_num], sample_weight=zonal_weights),                            
                                                   var_weights_dict['TLWP']*rmse(TLWP_obs_c[0:z_num], TLWP_proj_c[0:z_num], sample_weight=zonal_weights),
                                                   var_weights_dict['OSR']*rmse(OSR_obs_c[0:z_num], OSR_proj_c[0:z_num], sample_weight=zonal_weights),
                                                   var_weights_dict['OLR']*rmse(OLR_obs_c[0:z_num], OLR_proj_c[0:z_num], sample_weight=zonal_weights)])
    DY2_zonal_cost = zrg_weights_dict['zonal']*np.mean([var_weights_dict['PCP']*rmse(PCP_obs_c[(all_num):(all_num+z_num)], PCP_proj_c[(all_num):(all_num+z_num)], sample_weight=zonal_weights),
                                                   var_weights_dict['TLWP']*rmse(TLWP_obs_c[(all_num):(all_num+z_num)], TLWP_proj_c[(all_num):(all_num+z_num)], sample_weight=zonal_weights),
                                                   var_weights_dict['OSR']*rmse(OSR_obs_c[(all_num):(all_num+z_num)], OSR_proj_c[(all_num):(all_num+z_num)], sample_weight=zonal_weights),
                                                   var_weights_dict['OLR']*rmse(OLR_obs_c[(all_num):(all_num+z_num)], OLR_proj_c[(all_num):(all_num+z_num)], sample_weight=zonal_weights)])
    
    DY1_regional_cost = zrg_weights_dict['regional']*np.mean([var_weights_dict['PCP']*rmse(PCP_obs_c[(z_num):(z_num+r_num)], PCP_proj_c[(z_num):(z_num+r_num)], sample_weight=regional_weights),
                                                         var_weights_dict['TLWP']*rmse(TLWP_obs_c[(z_num):(z_num+r_num)], TLWP_proj_c[(z_num):(z_num+r_num)], sample_weight=regional_weights),
                                                         var_weights_dict['OSR']*rmse(OSR_obs_c[(z_num):(z_num+r_num)], OSR_proj_c[(z_num):(z_num+r_num)], sample_weight=regional_weights),
                                                         var_weights_dict['OLR']*rmse(OLR_obs_c[(z_num):(z_num+r_num)], OLR_proj_c[(z_num):(z_num+r_num)], sample_weight=regional_weights)])
    DY2_regional_cost = zrg_weights_dict['regional']*np.mean([var_weights_dict['PCP']*rmse(PCP_obs_c[(all_num+z_num):(all_num+z_num+r_num)], PCP_proj_c[(all_num+z_num):(all_num+z_num+r_num)], sample_weight=regional_weights),
                                                         var_weights_dict['TLWP']*rmse(TLWP_obs_c[(all_num+z_num):(all_num+z_num+r_num)], TLWP_proj_c[(all_num+z_num):(all_num+z_num+r_num)], sample_weight=regional_weights),
                                                         var_weights_dict['OSR']*rmse(OSR_obs_c[(all_num+z_num):(all_num+z_num+r_num)], OSR_proj_c[(all_num+z_num):(all_num+z_num+r_num)], sample_weight=regional_weights),
                                                         var_weights_dict['OLR']*rmse(OLR_obs_c[(all_num+z_num):(all_num+z_num+r_num)], OLR_proj_c[(all_num+z_num):(all_num+z_num+r_num)], sample_weight=regional_weights)])
    
    DY1_global_cost = zrg_weights_dict['global']*np.mean([var_weights_dict['PCP']*abs(PCP_obs_c[all_num-1] - PCP_proj_c[all_num-1]),
                                                     var_weights_dict['TLWP']*abs(TLWP_obs_c[all_num-1] - TLWP_proj_c[all_num-1]),
                                                     var_weights_dict['OSR']*abs(OSR_obs_c[all_num-1] - OSR_proj_c[all_num-1]),
                                                     var_weights_dict['OLR']*abs(OLR_obs_c[all_num-1] - OLR_proj_c[all_num-1])])
    DY2_global_cost = zrg_weights_dict['global']*np.mean([var_weights_dict['PCP']*abs(PCP_obs_c[-1] - PCP_proj_c[-1]),
                                                     var_weights_dict['TLWP']*abs(TLWP_obs_c[-1] - TLWP_proj_c[-1]),
                                                     var_weights_dict['OSR']*abs(OSR_obs_c[-1] - OSR_proj_c[-1]),
                                                     var_weights_dict['OLR']*abs(OLR_obs_c[-1] - OLR_proj_c[-1])])

    #print('PCP cost', np.mean(total_PCP_cost), 'TLWP cost', np.mean(total_TLWP_cost), 'OSR cost', np.mean(total_OSR_cost), 'OLR cost', np.mean(total_OLR_cost))                       
    print('zonal_cost', (DY1_zonal_cost+DY2_zonal_cost), 'regional_cost', (DY1_regional_cost+DY2_regional_cost), 'global_cost', (DY1_global_cost+DY2_global_cost))
    print('DY1 cost', (DY1_zonal_cost + DY1_regional_cost + DY1_global_cost), 'DY2 cost', (DY2_zonal_cost + DY2_regional_cost + DY2_global_cost))

    cost = DY_weights_dict['DY1']*(DY1_zonal_cost + DY1_regional_cost + DY1_global_cost) + DY_weights_dict['DY2']*(DY2_zonal_cost + DY2_regional_cost + DY2_global_cost)
    print('cost:', cost)
    
    return cost


# In[69]:


default_cost = ZRG_cost_function_mae_weighted_print(Y_train_norm[0], obs_norm, var_weights_dict, DY_weights_dict, zrg_weights_dict)
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

    #minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
    
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

# In[25]:


### These are the first optimal params run in SCREAM in Oct 2025
optimal_params_final = np.array( [1.00000000e-01, 8.72596066e+00, 5.81553407e-01, 1.00000000e-01,
                                  3.97719941e-01, 1.00000000e+00, 4.20402644e-02, 4.36854065e-02,
                                  1.00000000e-01, 1.00000000e+00, 1.00000000e-01, 1.00000000e-02,
                                  2.93051240e-01, 9.36429137e+06, 1.65530607e+00, 7.70072093e-11, 1.4575680588303337])


# In[26]:


#typically when output from basing hopping, they will be normalized/transformed
optimal_params_norm = X_pipe_sk_minmax.transform(optimal_params_final[0:-1].reshape(1,-1))


# In[27]:


print('default:')
print(params_to_cost_print(X_train_norm[0]))
print('optimal:')
print(params_to_cost_print(optimal_params_norm[0])) #


# ##### Must transform back before using in run

# In[28]:


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


# ### Optimizing results

# #### Gathering basinhopping results

# In[35]:


parameters_names_list = ['thl2tune', 'qw2tune', 'length_fac', 'c_diag_3rd_mom', 'Ckh', 'Ckm',
       'lambda_low', 'lambda_high', 'p3_spa_to_nc', 'p3_eci', 'p3_eri',
       'p3_k_accretion', 'p3_dep_nucleation_exponent', 'max_total_ni',
       'p3_ice_sed_knob', 'p3_d_breakup_cutoff']


# In[29]:


def gather_opt_output(file_list_name):
    # gather CSV files
    opts_file_list = glob.glob(file_list_name)
    # read into dataframe
    combined_df = pd.concat(
        (pd.read_csv(fname).assign(source_file=fname) for fname in sorted(opts_file_list)),
        ignore_index=True
    )
    optimals = combined_df.drop(['source_file', 'Rank'], axis=1)
    sort_opts = optimals.sort_values(by='cost')
    return sort_opts


# In[78]:


def parse_param_string(s):
    # Remove brackets, strip spaces, then split and convert
    clean = s.replace('[','').replace(']','').strip()
    return np.array(clean.split(), dtype=float)

def make_barcode_plot_opts(datset, title):
    param_arrays = datset.iloc[:,0].apply(parse_param_string).to_list()
    cost_values = datset.iloc[:,1].to_numpy().reshape(-1, 1)
    
    # Stack into single array [params | cost]
    initial_opt_params = np.hstack([np.vstack(param_arrays), cost_values])
    
    main_params = initial_opt_params[:, :16]  # first 16 numbers
    cost = initial_opt_params[:, 16:]         # last value (cost column)
    cost = np.flipud(cost)
    
    n_param_sets = initial_opt_params.shape[0]
    
    fig, ax = plt.subplots(figsize=(11, 0.1 * n_param_sets + 1))
    
    # Plot main parameters
    im1 = ax.imshow(np.clip(main_params, 0, 1),
                    aspect='auto', cmap='coolwarm', vmin=0, vmax=1)
    
    # Plot last parameter with a different colormap
    im2 = ax.imshow(cost, aspect='auto', cmap='Greens',
                    vmin=np.min(cost), vmax=np.max(cost),
                    extent=[16-0.5, 16+0.5, -0.5, n_param_sets-0.5])
    
    # Ticks and labels — remove parameterization names
    ax.set_yticks(np.arange(n_param_sets))
    ax.set_yticklabels([])  # no names
    
    ax.set_xticks(list(range(16)) + [16])
    ax.set_xticklabels(parameters_names_list + ['Cost'], rotation=90)
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Rank')
    ax.set_title(title)
    
    # Grid lines
    ax.set_xticks(np.arange(-.5, 17, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_param_sets, 1), minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(which='minor', bottom=False, left=False)
    
    ax.set_ylim(n_param_sets - 0.5, -0.5)
    
    #plt.axhline(y=n_param_sets-30.5, color='black', linestyle='-', linewidth=2)
    
    # Colorbars
    cbar1 = fig.colorbar(im1, ax=ax, fraction=0.025, pad=0.065, label='Normalized Parameter Value')
    cbar2 = fig.colorbar(im2, ax=ax, fraction=0.025, pad=0.01, label=' Cost')
    
    plt.tight_layout()
    plt.show()


# In[113]:


paramsss = np.array([0.24926626253817194, 0.9999996845376129, 0.059038938101014536, 0.0, 0.3756676839762469, 1.0, 0.4296164658921598, 0.99999999988307, 0.0004523068137208853, 1.0, 0.28399191283083763, 0.0, 0.8298709167971713, \
0.9997931509075013, 0.7572273305844853, 0.0])


# In[117]:


params_to_cost_print(paramsss)


# In[116]:


params_to_cost_print_rmse(paramsss)


# In[102]:


default_opts = gather_opt_output("/global/cfs/cdirs/e3sm/jpaige3/optimizing/Optimizing_results/default_cost_fun/results*.csv")
default_opts


# In[40]:


precip_opts = gather_opt_output("/global/cfs/cdirs/e3sm/jpaige3/optimizing/Optimizing_results/precip_cost_fun/results*.csv")
precip_opts


# In[41]:


precip4_opts = gather_opt_output("/global/cfs/cdirs/e3sm/jpaige3/optimizing/Optimizing_results/precip4_cost_fun/results*.csv")
precip4_opts


# In[42]:


no_region_opts = gather_opt_output("/global/cfs/cdirs/e3sm/jpaige3/optimizing/Optimizing_results/no_region_cost_fun/results*.csv")
no_region_opts


# In[43]:


tropics_weighted_opts = gather_opt_output("/global/cfs/cdirs/e3sm/jpaige3/optimizing/Optimizing_results/tropics_weighted_cost_fun/results*.csv")
tropics_weighted_opts


# In[79]:


make_barcode_plot_opts(default_opts, 'Default cost function')


# In[36]:


make_barcode_plot_opts(precip_opts, 'Precipitation Upweighted by 2 Cost Function')


# In[37]:


make_barcode_plot_opts(precip4_opts, 'Precipitation Upweighted by 4 Cost Function')


# In[294]:


make_barcode_plot_opts(no_region_opts, 'No Regional Contribution to Cost')


# In[428]:


make_barcode_plot_opts(tropics_weighted_opts, 'Tropics Upweighted')


# In[89]:


default_df = default_opts.iloc[[0]]
type(default_df.iloc[0,0]) 
default_df.iloc[0,0] = '[0.09090909090909091 0.09090909090909091 0.0404040404040404 0.696969696969697 0.0 0.0 0.009009009009009009 \n 0.7997997997997999 0.09090909090909091 0.4444444444444445 1.0 0.66996699669967 1.0 0.02526315789473684 0.0 0.5599999999999999]'
#str(X_train_norm[0].tolist())
default_df.iloc[0,1] = params_to_cost(X_train_norm[0])


# In[90]:


top_opts_across_costfuns = pd.concat([default_df, default_opts.iloc[[0]], precip_opts.iloc[[0]], precip4_opts.iloc[[0]], no_region_opts.iloc[[0]], tropics_weighted_opts.iloc[[0]]], axis=0)
top_opts_across_costfuns


# In[74]:


make_barcode_plot_opts(top_opts_across_costfuns, 'Optimal Parameters across cost functions')


# In[101]:


title = 'Default Parameter values vs. Optmized Parameter values'

param_arrays = top_opts_across_costfuns.iloc[:,0].apply(parse_param_string).to_list()
cost_values = top_opts_across_costfuns.iloc[:,1].to_numpy().reshape(-1, 1) 
# Stack into single array [params | cost]
initial_opt_params = np.hstack([np.vstack(param_arrays), cost_values]) 
main_params = initial_opt_params[:, :16]  # first 16 numbers
cost = initial_opt_params[:, 16:]         # last value (cost column)
cost = np.flipud(cost)   
n_param_sets = initial_opt_params.shape[0] 
fig, ax = plt.subplots(figsize=(11, 0.1 * n_param_sets + 1))
    
# Plot main parameters
im1 = ax.imshow(np.clip(main_params, 0, 1),
                aspect='auto', cmap='coolwarm', vmin=0, vmax=1)   
# Plot last parameter with a different colormap
im2 = ax.imshow(cost, aspect='auto', cmap='Greens',
                vmin=np.min(cost), vmax=np.max(cost),
                extent=[16-0.5, 16+0.5, -0.5, n_param_sets-0.5])
    
# Ticks and labels — remove parameterization names
ax.set_yticks(np.arange(n_param_sets))
ax.set_yticklabels(['default', 'std_cost', 'precip2', 'precip4', 'no_region', 'tropics2'])  # no names
    
ax.set_xticks(list(range(16)) + [16])
ax.set_xticklabels(parameters_names_list + ['Cost'], rotation=90)
ax.set_xlabel('Parameters')
ax.set_ylabel('Cost Function')
ax.set_title(title)
    
# Grid lines
ax.set_xticks(np.arange(-.5, 17, 1), minor=True)
ax.set_yticks(np.arange(-.5, n_param_sets, 1), minor=True)
ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax.tick_params(which='minor', bottom=False, left=False)
    
ax.set_ylim(n_param_sets - 0.5, -0.5)
    
#plt.axhline(y=n_param_sets-30.5, color='black', linestyle='-', linewidth=2)
    
# Colorbars
cbar1 = fig.colorbar(im1, ax=ax, fraction=0.025, pad=0.065, label='Normalized Parameter Value')
cbar2 = fig.colorbar(im2, ax=ax, fraction=0.025, pad=0.01, label=' Cost')
    
plt.tight_layout()
plt.show()


# #### Predicting Optimizing Results

# In[421]:


def predict_n_plot_opts(opts_df, cost_name):
    opt_params_norm = parse_param_string(opts_df.iloc[0,0])
    m_opt, v_opt = model_gp.predict(opt_params_norm.reshape(1,-1))
    
    PCP_proj_norm_opt = pd.DataFrame(m_opt[:, :, 0], index = ['opt_'+cost_name])
    TLWP_proj_norm_opt = pd.DataFrame(m_opt[:, :, 1], index = ['opt_'+cost_name])
    OSR_proj_norm_opt = pd.DataFrame(m_opt[:, :, 2], index = ['opt_'+cost_name])
    OLR_proj_norm_opt = pd.DataFrame(m_opt[:, :, 3], index = ['opt_'+cost_name])
    
    PCP_proj_opt = pd.DataFrame(Y_pipe_sk_ss_PCP.inverse_transform(PCP_proj_norm_opt))
    TLWP_proj_opt = pd.DataFrame(Y_pipe_sk_ss_TLWP.inverse_transform(TLWP_proj_norm_opt))
    OSR_proj_opt = pd.DataFrame(Y_pipe_sk_ss_OSR.inverse_transform(OSR_proj_norm_opt))
    OLR_proj_opt = pd.DataFrame(Y_pipe_sk_ss_OLR.inverse_transform(OLR_proj_norm_opt))

    plot_opts(PCP_proj_opt*1e3*24*3600, Y_train_ZRG[0,:,0]*1e3*24*3600, PCP_zrg_obs*1e3*24*3600, cost_name, 'PCP', 'mm day^-1')
    plot_opts(TLWP_proj_opt, Y_train_ZRG[0,:,1], TLWP_zrg_obs, cost_name, 'TLWP', 'kg m^-2')
    plot_opts(OSR_proj_opt, Y_train_ZRG[0,:,2], OSR_zrg_obs, cost_name, 'OSR', 'W m^-2')
    plot_opts(OLR_proj_opt, Y_train_ZRG[0,:,3], OLR_zrg_obs, cost_name, 'OLR', 'W m^-2')

def plot_opts(proj_def, def_ZRG, zrg_obs, title, var, units):
    plt.figure(figsize=(10, 4))
    point_size = 30
    x_range = range(50)
    
    plt.scatter(x_range, proj_def, label="Projected opt", marker = 's', edgecolor='green', facecolors="none", s=point_size)
    plt.scatter(x_range, def_ZRG, label="Default (m0000)", marker = 'x', color='blue', s=point_size)
    plt.scatter(x_range, zrg_obs, label="Obs", marker = '^', edgecolor='red', facecolors="none", s=point_size)
    
    plt.axvline(x=24.5, color='black')
    
    obs_untransform
    plt.xticks(range(50), zrg_labels+zrg_labels, rotation=45, ha='right', fontsize=8)
    plt.xlabel("DY1 Center of latitude band       DY1 Region               DY1 Global" + 
               "       DY2 Center of latitude band       DY2 Region               DY2 Global")
    plt.ylabel(var+" error ("+units+")")
    plt.title(title+" cost function projection comparison - "+var)
    plt.legend(fontsize=7)
    plt.tight_layout()

    save_path = os.path.join('/global/cfs/cdirs/e3sm/jpaige3/Validation_plots', title+var+'_cost_fun_compare.png')
    plt.savefig(save_path)

    plt.show()
    


# In[423]:


predict_n_plot_opts(default_opts, 'Default')


# In[424]:


predict_n_plot_opts(precip_opts, 'Precip upweighted')


# In[425]:


predict_n_plot_opts(no_region_opts, 'No regions')


# In[426]:


predict_n_plot_opts(tropics_weighted_opts, 'Tropics upweighted')


# ##### Example

# ### Analzing Optimal runs
Runs that have been completed so far:
DY2 default Frontier
DY2 default PM
DY2 Initial Results Opt run Frontier
DY1 default PM
#DY1 Initial Results Ot
# In[432]:


parameters_names_list = ['thl2tune', 'qw2tune', 'length_fac', 'c_diag_3rd_mom', 'Ckh', 'Ckm',
       'lambda_low', 'lambda_high', 'p3_spa_to_nc', 'p3_eci', 'p3_eri',
       'p3_k_accretion', 'p3_dep_nucleation_exponent', 'max_total_ni',
       'p3_ice_sed_knob', 'p3_d_breakup_cutoff']


# In[439]:


#filter variables in output - keep only the 4 of interest
to_keep = ['precip_total_surf_mass_flux','LiqWaterPath','RainWaterPath','SW_flux_up_at_model_top','LW_flux_up_at_model_top']
           
to_leave = ['SW_flux_dn','SW_flux_dn_at_model_bot','SW_flux_up','SW_flux_up_at_model_bot','SW_flux_dn_at_model_top', 'T_2m',
'T_mid', 'precip_ice_surf_mass_flux','precip_liq_surf_mass_flux','ps','qc','qi','qm','qr','qv','qv_2m','LW_flux_up','LW_flux_up_at_model_bot',
'IceWaterPath', 'LW_flux_dn','LW_flux_dn_at_model_bot','time_bnds', 'LongwaveCloudForcing', 'MeridionalVapFlux','ShortwaveCloudForcing','U','V', 
'VapWaterPath', 'ZonalVapFlux','bm','eddy_diff_mom', 'eff_radius_qc_at_cldtop','eff_radius_qi_at_cldtop', 'homme_T_mid_tend', 'homme_qv_tend',
'horiz_winds_at_model_bot', 'nc', 'ni', 'nr', 'omega', 'p3_T_mid_tend', 'p3_qv_tend', 'rrtmgp_T_mid_tend', 'sgs_buoy_flux', 'shoc_T_mid_tend',
'shoc_qv_tend', 'surf_evap', 'surf_mom_flux', 'surf_radiative_T','surf_sens_flux', 'surface_upward_latent_heat_flux', 'avg_count_ncol', 
'avg_count_ncol_lev', 'avg_count_ncol_dim', 'area', 'lat', 'lon']


# In[447]:


def load_in_new_run(file_path):
    data = xr.open_dataset(file_path)
    #data_small = data.drop_vars(to_leave)
    data_small = data[to_keep]
    data_small['TotalLiqWaterPath'] = (data_small.LiqWaterPath + data_small.RainWaterPath)
    data_small.squeeze('time')
    return data_small


# In[451]:


new_default_data = load_in_new_run('/global/cfs/cdirs/e3smdata/simulations/ecp-autotune/sims-dec3-2024/hh1024/m0000/SCREAM.2024-autocal-00.ne1024pg2/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc')
#/global/cfs/cdirs/e3smdata/simulations/ecp-autotune/sims-dec3-2024/hh1024/m0000/SCREAM.2024-autocal-00.ne1024pg2/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc')


# In[456]:


#New New default analysis, run on Frontier (10/9/2025) (I believe I ran this or a similar one)
newnew_default_data = load_in_new_run('/global/cfs/cdirs/e3smdata/simulations/ecp-autotune/f00-dec2-2024-2444ff44ec/hh1024/m0000defle/SCREAM.2024-autocal-00.ne1024pg2/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc')
#xr.open_dataset('/global/cfs/cdirs/e3smdata/simulations/ecp-autotune/sims-f00-dec2-2024-2444ff44ec/def-aug2019.ne1024pg2_ne1024pg2.F2010-SCREAMv1.f00-dec2-2024-2444ff44ec.2d.n2048.default/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2019-08-02-00000.nc')
#/global/cfs/cdirs/e3smdata/simulations/ecp-autotune/sims-dec3-2024/hh1024/m0000/SCREAM.2024-autocal-00.ne1024pg2/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc')


# In[455]:


#New New New default analysis, on PM now, run by Noel (2/11/2026)
newnewnew_default_data = load_in_new_run('/pscratch/sd/n/ndk/e3sm_scratch/pm-gpu/p01-dec2-2024-2444ff44ec/ii1024/m0000/SCREAM.2024-autocal-00.ne1024pg2/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc')


# In[452]:


int_opt_data_Oct25 = load_in_new_run('/global/cfs/cdirs/e3sm/jpaige3/optimizing_SCREAM_runs/optbasin1rank1_DY2/SCREAM.2024-autocal-00.ne1024pg2/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc')


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
    
DY2_final_default_error = DY2_PCP_default_zrg_error+DY2_TLWP_default_zrg_error+DY2_OSR_default_zrg_error+DY2_OLR_default_zrg_error

print('Argmin PCP:', sim_names[np.argmin(DY2_PCP_default_zrg_error)])
print('Argmin TLWP:', sim_names[np.argmin(DY2_TLWP_default_zrg_error)])
print('Argmin OSR:', sim_names[np.argmin(DY2_OSR_default_zrg_error)])
print('Argmin OLR:', sim_names[np.argmin(DY2_OLR_default_zrg_error)])

print('Final Argmin:', sim_names[np.argmin(DY2_final_default_zrg_error)])


# #### Plotting

# In[ ]:





# In[44]:


placeholder_zrgdataset_train = PCP_train
title = 'PCP'
#precip_total_surf_mass_flux
#TotalLiqWaterPath
#SW_flux_up_at_model_top
#LW_flux_up_at_model_top


# Configuration
datasets = [
    {
        'data': (control.precip_total_surf_mass_flux),
        #'data': (control.LiqWaterPath + control.RainWaterPath)
        'label': 'PPE Default simulation',
        'color': 'blue',
        'marker': 'o'
    },
    {
        'data': (newnew_default_data_small.precip_total_surf_mass_flux),
        'label': 'New Frontier Default simulation',
        'color': 'purple',
        'marker': 'o'
    },
        {
        'data': (newnewnew_default_data_small.precip_total_surf_mass_flux),
        'label': 'New PM Opt Default simulation',
        'color': 'green',
        'marker': 'o'
    }
]

# Calculate means for all datasets
results = []
for dataset in datasets:
    result = {
        'zonal': zonal_means_native(dataset['data'], area, lat, lon),
        'regional': regional_means_native(dataset['data'], area),
        'global': np.array([global_means_native(dataset['data'], area)]),
        'label': dataset['label'],
        'color': dataset['color'],
        'marker': dataset['marker']
    }
    results.append(result)

# Extract keys and create x positions (using first dataset as reference)
zonal_keys = list(results[0]['zonal'].keys())
regional_keys = list(results[0]['regional'].keys())
array_labels = ['global']

z_num = len(zonal_keys)
r_num = len(regional_keys)
all_num = z_num + r_num + 1

x_zonal = range(z_num)
x_regional = range(z_num, z_num + r_num)
x_global = range(z_num + r_num, z_num + r_num + 1)

# Create plot
plt.figure(figsize=(8, 4))
plt.axhline(y=0, color="black", linewidth=1, zorder=1, alpha=0.7)

# Plot each dataset
for result in results:
    zonal_values = list(result['zonal'].values())
    regional_values = list(result['regional'].values())
    global_value = result['global'][0]
    
    # Plot zonal means
    plt.plot(x_zonal, 
             np.array(zonal_values) - placeholder_zrgdataset_train[0, all_num:z_num+all_num],
             #np.array(zonal_values) - DY2_OLR_zrg_ppedataset.iloc[0, 0:z_num],
             f'-{result["marker"]}', 
             label=result['label'], 
             color=result['color'])
    
    # Plot regional means
    plt.plot(x_regional, 
             np.array(regional_values) - placeholder_zrgdataset_train[0, all_num+z_num:(z_num+r_num)+all_num],
             #np.array(regional_values) - DY2_OLR_zrg_ppedataset.iloc[0, z_num:(z_num+r_num)],
             f'-{result["marker"]}', 
             color=result['color'])
    
    # Plot global mean
    plt.plot(x_global, 
             global_value - placeholder_zrgdataset_train[0, -1], 
             #global_value - DY2_OLR_zrg_ppedataset.iloc[0, -1],
             f'-{result["marker"]}', 
             color=result['color'])

# Customize axes
all_keys = zonal_keys + regional_keys + array_labels
all_positions = list(x_zonal) + list(x_regional) + list(x_global)

plt.xticks(all_positions, all_keys, rotation=45, ha='right')
plt.xlabel("           Center of latitude band                             Region               Global")
plt.ylabel("Target variable error")
plt.title(title+" comparison plot (to m0000)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()


# ##### Comparison plots

# In[434]:


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

