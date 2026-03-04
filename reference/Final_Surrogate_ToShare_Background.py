#!/usr/bin/env python
# coding: utf-8

# # Background on surrogate training workflow
# 

# Load in packages - Tensorflow warnings may affect ESEM GP performance

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
from sklearn.metrics import make_scorer, r2_score, root_mean_squared_error
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline

import tensorflow as tf
import gpflow
from datetime import date
from datetime import datetime

from scipy.optimize import minimize


# ## Running the models

# ### Load the files/parameters/variables

# Load parameter sampling

# In[2]:


path_to_json = '/global/cfs/cdirs/e3sm/jpaige3/ESEm/SCREAM.2024-autocal-00.ne1024pg2-params.json' #this file contains the specific parameters for each run
ppe_params_all = pd.read_json(path_to_json)


# In[3]:


(ppe_params_all['p3_ice_sed_knob']>=1).sum() #number of ppe runs with required restricted parameter ranges


# In[4]:


##collects the labels for all runs --- DY2
DY2_path = "/global/cfs/cdirs/e3smdata/simulations/ecp-autotune/SCREAM.2024-autocal-00.ne1024pg2/" #path to the runs
DY2_missing_folders =[] #will collect all removed runs
DY2_folders = []

#Reading in all file folders
#m**** files
    #When trying increasing amounts of data we will go with up to 50, 100, 150, 200, 287
for m in range(0, 301): #DY1 goes to 301
    folder = 'm{:04}'.format(m)
    if os.path.exists(DY2_path+folder):
        DY2_folders.append(m)
    else:
        DY2_missing_folders.append(m) #just in integers, not file name (if you want this append folder)  
DY2_folders = ['m{:04}'.format(m) for m in DY2_folders] #writes the runs in the m**** format
#opt**** files
for file in os.listdir(DY2_path):
    #if file.startswith("m0") or file.startswith("opt"):
    if file.startswith("opt"):
        DY2_folders.append(file)
#does not include t0000 as this is pointed to by m0000

print(len(DY2_folders)) 

#removing files that don't have all the data--could be doing this better
DY2_folders.remove('m0230') #this file is missing 3hr averages
DY2_folders.remove('optmar20day5') #this file is missing 2nd day daily averages

#not in DY1
DY2_folders.remove('m0024')
DY2_folders.remove('m0025')
DY2_folders.remove('m0061')
DY2_folders.remove('optmar22hd')

##adds the path name for all files--only 2nd day, daily averages
DY2_filename_list = []
DY2_data_dir = DY2_path #'/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/SCREAM.2024-autocal-00.ne1024pg2/'
#data_dir = '/global/cfs/projectdirs/e3smdata/simulations/SCREAM.2024-autocal-00.ne1024pg2/' #pointed to by above
for f in DY2_folders:
    file_path = DY2_data_dir+f+'/SCREAM.2024-autocal-00.ne1024pg2/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc'
    #file_path1 = data_dir+f+'/SCREAM.2024-autocal-00.ne1024pg2/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-20-00000.nc'
    #file_path2 = data_dir+f+'/SCREAM.2024-autocal-00.ne1024pg2/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc'
    DY2_filename_list.append(file_path)

#count_fail = 0
##filter ice_sed < 1 to exclude runs outside of typical range
DY2_file_check = np.zeros(len(DY2_filename_list),dtype=bool) #initialize file check
for i in range(len(DY2_filename_list)):
    ppe_member = DY2_folders[i]
    if (float(ppe_params_all['p3_ice_sed_knob'][ppe_member]) >= 1.0):
        DY2_file_check[i] = True
#    else:
#        count_fail += 1
#print(count_fail)

DY2_filename_list_filtered = np.array(DY2_filename_list)[DY2_file_check==True] #turns the file name list to a np array
DY2_sim_names = np.array(DY2_folders)[DY2_file_check==True]
DY2_ppe_params = ppe_params_all[ppe_params_all.index.isin(DY2_sim_names)]
len(DY2_ppe_params)


# In[5]:


##collects the labels for all runs --- DY1
DY1_path = "/pscratch/sd/j/jpaige3/dy1ne1024/" #new scratch location of DY1
#"/global/cfs/cdirs/e3smdata/simulations/ecp-autotune/sims-s15-mar7/setupA1/" #previous path to the runs
DY1_missing_folders =[] #will collect all removed runs
DY1_folders = []

#When trying increasing amounts of data we will go with up to 50, 100, 150, 200, 287 (this is where DY2 ends)
for m in range(0, 301): 
    folder = 'm{:04}'.format(m)
    if os.path.exists(DY1_path+folder):
        DY1_folders.append(m)
    else:
        DY1_missing_folders.append(m) #just in integers, not file name (if you want this append folder)  
DY1_folders = ['m{:04}'.format(m) for m in DY1_folders] #writes the runs in the m**** format

#Could also pull all files in this way, but will be out of order and include m0000-5day
for file in os.listdir(DY1_path):
    #if file.startswith("m0") or file.startswith("opt"):
    if file.startswith("opt"):
        DY1_folders.append(file)
#does not include t0000 as this is pointed to by m0000

#removing files that don't have all the data
#folders.remove('m0230') #this file is missing 3hr averages
#folders.remove('optmar20day5') #this file is missing 2nd day daily averages

#DY2 -- invalid runs -- also not in the json
DY1_folders.remove('m0024')
DY1_folders.remove('m0025')
DY1_folders.remove('m0061')
DY1_folders.remove('optmar22hd')

DY1_folders.remove('m0262')
DY1_folders.remove('m0263')
DY1_folders.remove('m0264')
DY1_folders.remove('m0266')
DY1_folders.remove('m0267')
DY1_folders.remove('m0270')
DY1_folders.remove('m0272')
DY1_folders.remove('m0274')
DY1_folders.remove('m0275')
DY1_folders.remove('m0279')
DY1_folders.remove('m0289')
DY1_folders.remove('m0290')
DY1_folders.remove('m0292')
DY1_folders.remove('m0293')
DY1_folders.remove('m0294')
DY1_folders.remove('m0295')
DY1_folders.remove('m0296')
DY1_folders.remove('m0299')
DY1_folders.remove('m0300')

DY1_folders.remove('optmar15seed0')
DY1_folders.remove('optmar27a')
DY1_folders.remove('optmar20dayAll')
DY1_folders.remove('optmar20day2-fail')
DY1_folders.remove('optmar15b')
DY1_folders.remove('optmar20day2-ltend')

#Not in DY2
DY1_folders.remove('m0230')
DY1_folders.remove('optmar20day5')
print(len(DY1_folders))

##adds the path name for all files--only 2nd day, daily averages
DY1_filename_list = []
DY1_data_dir = DY1_path #'/global/cfs/cdirs/e3smdata/simulations/ecp-autotune/sims-s15-mar7/setupA1/'
#data_dir = '/global/cfs/projectdirs/e3smdata/simulations/SCREAM.2024-autocal-00.ne1024pg2/' #pointed to by above
for f in DY1_folders:
    file_path = DY1_data_dir+f+'/SCREAM.2024-autocal-00.ne1024pg2/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2016-08-07-00000.nc'
    DY1_filename_list.append(file_path)
    
##filter ice_sed < 1 to exclude runs outside of typical range
DY1_file_check = np.zeros(len(DY1_filename_list),dtype=bool) #initialize file check
for i in range(len(DY1_filename_list)):
    ppe_member = DY1_folders[i]
    #if (int(sim_names[i][1:]) < 265): #try it with all the runs
    if (float(ppe_params_all['p3_ice_sed_knob'][ppe_member]) >= 1.0):
        DY1_file_check[i] = True

DY1_filename_list_filtered = np.array(DY1_filename_list)[DY1_file_check==True] #turns the file name list to a np array
DY1_sim_names = np.array(DY1_folders)[DY1_file_check==True]
DY1_ppe_params = ppe_params_all[ppe_params_all.index.isin(DY1_sim_names)]
len(DY1_ppe_params)


# In[6]:


#Check for intersection between DY1 and DY2, take only runs in both
print(list(set(DY1_sim_names) - set(DY2_sim_names))) # should be empty
print(list(set(DY2_sim_names) - set(DY1_sim_names)))
sim_names = [sim for sim in DY1_sim_names if sim in DY2_sim_names] 

ppe_params = ppe_params_all[ppe_params_all.index.isin(sim_names)]
ppe_params


# In[7]:


# Open and concatenate the datasets along the new 'run_number' dimension
DY1_concat_dataset = xr.open_mfdataset(DY1_filename_list_filtered, concat_dim='run_label', combine='nested')
DY1_ppe_dataset_time = DY1_concat_dataset.assign_coords(run_label=('run_label', DY1_sim_names)) # Assign the 'run_number' coordinate
DY1_ppe_dataset = DY1_ppe_dataset_time.squeeze('time')


# In[8]:


# Open and concatenate the datasets along the new 'run_number' dimension
DY2_concat_dataset = xr.open_mfdataset(DY2_filename_list_filtered, concat_dim='run_label', combine='nested')
DY2_ppe_dataset_time = DY2_concat_dataset.assign_coords(run_label=('run_label', DY2_sim_names)) # Assign the 'run_number' coordinate
DY2_ppe_dataset = DY2_ppe_dataset_time.squeeze('time')


# ##### Observations

# In[9]:


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


# In[10]:


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


# ##### Filter data

# In[11]:


#filter variables in output - keep only the 4 of interest
to_keep = ['precip_total_surf_mass_flux','LiqWaterPath','RainWaterPath','SW_flux_up_at_model_top','LW_flux_up_at_model_top']
           
to_leave = ['SW_flux_dn','SW_flux_dn_at_model_bot','SW_flux_up','SW_flux_up_at_model_bot','SW_flux_dn_at_model_top', 'T_2m',
'T_mid', 'precip_ice_surf_mass_flux','precip_liq_surf_mass_flux','ps','qc','qi','qm','qr','qv','qv_2m','LW_flux_up','LW_flux_up_at_model_bot',
'IceWaterPath', 'LW_flux_dn','LW_flux_dn_at_model_bot','time_bnds', 'LongwaveCloudForcing', 'MeridionalVapFlux','ShortwaveCloudForcing','U','V', 
'VapWaterPath', 'ZonalVapFlux','bm','eddy_diff_mom', 'eff_radius_qc_at_cldtop','eff_radius_qi_at_cldtop', 'homme_T_mid_tend', 'homme_qv_tend',
'horiz_winds_at_model_bot', 'nc', 'ni', 'nr', 'omega', 'p3_T_mid_tend', 'p3_qv_tend', 'rrtmgp_T_mid_tend', 'sgs_buoy_flux', 'shoc_T_mid_tend',
'shoc_qv_tend', 'surf_evap', 'surf_mom_flux', 'surf_radiative_T','surf_sens_flux', 'surface_upward_latent_heat_flux', 'avg_count_ncol', 
'avg_count_ncol_lev', 'avg_count_ncol_dim', 'area', 'lat', 'lon']


# In[12]:


DY1_ppe_dataset_small = DY1_ppe_dataset.drop_vars(to_leave)
DY1_ppe_dataset_small['TotalLiqWaterPath'] = (DY1_ppe_dataset_small.LiqWaterPath + DY1_ppe_dataset_small.RainWaterPath)
#ppe_dataset_small['precip_total_surf_mass_flux'] = ppe_dataset_small['precip_total_surf_mass_flux']*1e-3*24*3600
DY1_ppe_dataset_small = DY1_ppe_dataset_small.drop_vars('p_levs')
DY1_ppe_dataset_small = DY1_ppe_dataset_small.rename({var: f"DY1_{var}" for var in DY1_ppe_dataset_small.data_vars})


# In[13]:


DY2_ppe_dataset_small = DY2_ppe_dataset.drop_vars(to_leave)
DY2_ppe_dataset_small['TotalLiqWaterPath'] = (DY2_ppe_dataset_small.LiqWaterPath + DY2_ppe_dataset_small.RainWaterPath)
#ppe_dataset_small['precip_total_surf_mass_flux'] = ppe_dataset_small['precip_total_surf_mass_flux']*1e-3*24*3600]
#DY2_ppe_dataset_small = DY1_ppe_dataset_small.drop_vars('p_levs')
DY2_ppe_dataset_small = DY2_ppe_dataset_small.rename({var: f"DY2_{var}" for var in DY2_ppe_dataset_small.data_vars})


# In[14]:


#ppe_dataset_small = xr.concat([DY1_ppe_dataset_small, DY2_ppe_dataset_small], dim='run_label')
ppe_dataset_small = DY1_ppe_dataset_small.sel(run_label=sim_names).combine_first(DY2_ppe_dataset_small.sel(run_label=sim_names))
ppe_dataset_small

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
# #### Taking spatial averages - zonal, regional, global

# ##### Mask to observations
# Observations are available on a subset of grid cells; this will affect averages, so first mask to the subset of grid cells

# In[74]:


##two ways to get these masks
mask_path ='/global/cfs/cdirs/e3sm/jpaige3/mahf708_newvs_surrogate/Autotuning-NGD/src/scream-jp/surrogate_and_optimization/'

DY1_PCP_mask = xr.open_dataset(mask_path+'/masks/precip_1_output.nc').squeeze('time', drop=True)
DY2_PCP_mask = xr.open_dataset(mask_path+'/masks/precip_2_output.nc').squeeze('time', drop=True)
DY1_TLWP_mask = xr.open_dataset(mask_path+'/masks/tlwp_1_output.nc').squeeze('time', drop=True)
DY2_TLWP_mask = xr.open_dataset(mask_path+'/masks/tlwp_2_output.nc').squeeze('time', drop=True)

#len(np.where(((DY1_TLWP_mask.to_dataarray()[0]) == True).values)[0]) - ncol drops from 21600 to 15313


# In[16]:


DY1_ppe_dataset_mask = DY1_ppe_dataset_small.copy(deep=True)
DY2_ppe_dataset_mask = DY1_ppe_dataset_small.copy(deep=True)

#DY1
DY1_ppe_dataset_mask['DY1_precip_total_surf_mass_flux'] = DY1_ppe_dataset_small.DY1_precip_total_surf_mass_flux.where(np.isnan(DY1_PCP_obs) == False)
DY1_ppe_dataset_mask['DY1_TotalLiqWaterPath'] = DY1_ppe_dataset_small.DY1_TotalLiqWaterPath.where(np.isnan(DY1_TLWP_obs) == False)
DY1_ppe_dataset_mask['DY1_SW_flux_up_at_model_top'] = DY1_ppe_dataset_small.DY1_SW_flux_up_at_model_top.where(np.isnan(DY1_OSR_obs) == False)
DY1_ppe_dataset_mask['DY1_LW_flux_up_at_model_top'] = DY1_ppe_dataset_small.DY1_LW_flux_up_at_model_top.where(np.isnan(DY1_OLR_obs) == False)

#DY2
DY2_ppe_dataset_mask['DY2_precip_total_surf_mass_flux'] = DY2_ppe_dataset_small.DY2_precip_total_surf_mass_flux.where(np.isnan(DY2_PCP_obs) == False)
DY2_ppe_dataset_mask['DY2_TotalLiqWaterPath'] = DY2_ppe_dataset_small.DY2_TotalLiqWaterPath.where(np.isnan(DY2_TLWP_obs) == False)
DY2_ppe_dataset_mask['DY2_SW_flux_up_at_model_top'] = DY2_ppe_dataset_small.DY2_SW_flux_up_at_model_top.where(np.isnan(DY2_OSR_obs) == False)
DY2_ppe_dataset_mask['DY2_LW_flux_up_at_model_top'] = DY2_ppe_dataset_small.DY2_LW_flux_up_at_model_top.where(np.isnan(DY2_OLR_obs) == False)

DY1_ppe_dataset_small = DY1_ppe_dataset_mask
DY2_ppe_dataset_small = DY2_ppe_dataset_mask


# In[17]:


regions_file = xr.open_dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/regions.nc')
regions_list = ['poles','extratropical_land','extratropical_ocean','tropical_land','ascending_tropical_ocean','descending_tropical_ocean']
#area = ppe_dataset.area[1,:] #only taking the first row, because all rows should have the same values
control = xr.open_dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/SCREAM.2024-autocal-00.ne1024pg2/m0000/SCREAM.2024-autocal-00.ne1024pg2/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc')
area = control.variables['area'][:]
lat = control.variables['lat'][:]
lon = control.variables['lon'][:]


# In[18]:


def zonal_means_native(data, area, lat, lon):
    lat_bands = np.linspace(-90,90,19) #currently dividing globe in 18 zones - 10 degree bands
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


# Below we are just taking the averages, in a likely not very efficient way... :(

# In[20]:


DY1_PCP_zonal_data = dict()
DY1_TLWP_zonal_data = dict()
DY1_OSR_zonal_data = dict()
DY1_OLR_zonal_data = dict()

DY1_PCP_regional_data = dict()
DY1_TLWP_regional_data = dict()
DY1_OSR_regional_data = dict()
DY1_OLR_regional_data = dict()

DY1_PCP_global_data = []
DY1_TLWP_global_data = []
DY1_OSR_global_data = []
DY1_OLR_global_data = []

for run in sim_names:
    precip = ppe_dataset_small.sel(run_label = run)['DY1_precip_total_surf_mass_flux']
    tlwp = ppe_dataset_small.sel(run_label = run)['DY1_TotalLiqWaterPath']
    swflux = ppe_dataset_small.sel(run_label = run)['DY1_SW_flux_up_at_model_top']
    lwflux = ppe_dataset_small.sel(run_label = run)['DY1_LW_flux_up_at_model_top']

    DY1_PCP_zonal_data[run] = zonal_means_native(precip, area, lat, lon)
    DY1_TLWP_zonal_data[run] = zonal_means_native(tlwp, area, lat, lon)
    DY1_OSR_zonal_data[run] = zonal_means_native(swflux, area, lat, lon)
    DY1_OLR_zonal_data[run] = zonal_means_native(lwflux, area, lat, lon)
    
    DY1_PCP_regional_data[run] = regional_means_native(precip, area)
    DY1_TLWP_regional_data[run] = regional_means_native(tlwp, area)
    DY1_OSR_regional_data[run] = regional_means_native(swflux, area)
    DY1_OLR_regional_data[run] = regional_means_native(lwflux, area)    
    
    DY1_PCP_global_data.append(global_means_native(precip, area))
    DY1_TLWP_global_data.append(global_means_native(tlwp, area))
    DY1_OSR_global_data.append(global_means_native(swflux, area))
    DY1_OLR_global_data.append(global_means_native(lwflux, area))


# In[21]:


DY2_PCP_zonal_data = dict()
DY2_TLWP_zonal_data = dict()
DY2_OSR_zonal_data = dict()
DY2_OLR_zonal_data = dict()

DY2_PCP_regional_data = dict()
DY2_TLWP_regional_data = dict()
DY2_OSR_regional_data = dict()
DY2_OLR_regional_data = dict()

DY2_PCP_global_data = []
DY2_TLWP_global_data = []
DY2_OSR_global_data = []
DY2_OLR_global_data = []

for run in sim_names:
    precip = ppe_dataset_small.sel(run_label = run)['DY2_precip_total_surf_mass_flux']
    tlwp = ppe_dataset_small.sel(run_label = run)['DY2_TotalLiqWaterPath']
    swflux = ppe_dataset_small.sel(run_label = run)['DY2_SW_flux_up_at_model_top']
    lwflux = ppe_dataset_small.sel(run_label = run)['DY2_LW_flux_up_at_model_top']

    DY2_PCP_zonal_data[run] = zonal_means_native(precip, area, lat, lon)
    DY2_TLWP_zonal_data[run] = zonal_means_native(tlwp, area, lat, lon)
    DY2_OSR_zonal_data[run] = zonal_means_native(swflux, area, lat, lon)
    DY2_OLR_zonal_data[run] = zonal_means_native(lwflux, area, lat, lon)
    
    DY2_PCP_regional_data[run] = regional_means_native(precip, area)
    DY2_TLWP_regional_data[run] = regional_means_native(tlwp, area)
    DY2_OSR_regional_data[run] = regional_means_native(swflux, area)
    DY2_OLR_regional_data[run] = regional_means_native(lwflux, area)    
    
    DY2_PCP_global_data.append(global_means_native(precip, area))
    DY2_TLWP_global_data.append(global_means_native(tlwp, area))
    DY2_OSR_global_data.append(global_means_native(swflux, area))
    DY2_OLR_global_data.append(global_means_native(lwflux, area))


# In[22]:


DY1_PCP_z_df = pd.DataFrame.from_dict(DY1_PCP_zonal_data, orient="index")
DY1_PCP_r_df = pd.DataFrame.from_dict(DY1_PCP_regional_data, orient="index")
DY1_PCP_zrg_ppedataset = pd.concat([DY1_PCP_z_df, DY1_PCP_r_df], axis=1)
DY1_PCP_zrg_ppedataset["DY1_global"] = DY1_PCP_global_data

DY2_PCP_z_df = pd.DataFrame.from_dict(DY2_PCP_zonal_data, orient="index")
DY2_PCP_r_df = pd.DataFrame.from_dict(DY2_PCP_regional_data, orient="index")
DY2_PCP_zrg_ppedataset = pd.concat([DY2_PCP_z_df, DY2_PCP_r_df], axis=1)
DY2_PCP_zrg_ppedataset["DY2_global"] = DY2_PCP_global_data

PCP_zrg_ppedataset = pd.concat([DY1_PCP_zrg_ppedataset, DY2_PCP_zrg_ppedataset], axis=1)


# In[23]:


DY1_TLWP_z_df = pd.DataFrame.from_dict(DY1_TLWP_zonal_data, orient="index")
DY1_TLWP_r_df = pd.DataFrame.from_dict(DY1_TLWP_regional_data, orient="index")
DY1_TLWP_zrg_ppedataset = pd.concat([DY1_TLWP_z_df, DY1_TLWP_r_df], axis=1)
DY1_TLWP_zrg_ppedataset["DY1_global"] = DY1_TLWP_global_data

DY2_TLWP_z_df = pd.DataFrame.from_dict(DY2_TLWP_zonal_data, orient="index")
DY2_TLWP_r_df = pd.DataFrame.from_dict(DY2_TLWP_regional_data, orient="index")
DY2_TLWP_zrg_ppedataset = pd.concat([DY2_TLWP_z_df, DY2_TLWP_r_df], axis=1)
DY2_TLWP_zrg_ppedataset["DY2_global"] = DY2_TLWP_global_data

TLWP_zrg_ppedataset = pd.concat([DY1_TLWP_zrg_ppedataset, DY2_TLWP_zrg_ppedataset], axis=1)


# In[24]:


DY1_OSR_z_df = pd.DataFrame.from_dict(DY1_OSR_zonal_data, orient="index")
DY1_OSR_r_df = pd.DataFrame.from_dict(DY1_OSR_regional_data, orient="index")
DY1_OSR_zrg_ppedataset = pd.concat([DY1_OSR_z_df, DY1_OSR_r_df], axis=1)
DY1_OSR_zrg_ppedataset["DY1_global"] = DY1_OSR_global_data

DY2_OSR_z_df = pd.DataFrame.from_dict(DY2_OSR_zonal_data, orient="index")
DY2_OSR_r_df = pd.DataFrame.from_dict(DY2_OSR_regional_data, orient="index")
DY2_OSR_zrg_ppedataset = pd.concat([DY2_OSR_z_df, DY2_OSR_r_df], axis=1)
DY2_OSR_zrg_ppedataset["DY2_global"] = DY2_OSR_global_data

OSR_zrg_ppedataset = pd.concat([DY1_OSR_zrg_ppedataset, DY2_OSR_zrg_ppedataset], axis=1)


# In[25]:


DY1_OLR_z_df = pd.DataFrame.from_dict(DY1_OLR_zonal_data, orient="index")
DY1_OLR_r_df = pd.DataFrame.from_dict(DY1_OLR_regional_data, orient="index")
DY1_OLR_zrg_ppedataset = pd.concat([DY1_OLR_z_df, DY1_OLR_r_df], axis=1)
DY1_OLR_zrg_ppedataset["DY1_global"] = DY1_OLR_global_data

DY2_OLR_z_df = pd.DataFrame.from_dict(DY2_OLR_zonal_data, orient="index")
DY2_OLR_r_df = pd.DataFrame.from_dict(DY2_OLR_regional_data, orient="index")
DY2_OLR_zrg_ppedataset = pd.concat([DY2_OLR_z_df, DY2_OLR_r_df], axis=1)
DY2_OLR_zrg_ppedataset["DY2_global"] = DY2_OLR_global_data

OLR_zrg_ppedataset = pd.concat([DY1_OLR_zrg_ppedataset, DY2_OLR_zrg_ppedataset], axis=1)


# In[26]:


#### Ordering matters here
zrg_ppedataset = pd.concat([PCP_zrg_ppedataset, TLWP_zrg_ppedataset, OSR_zrg_ppedataset, OLR_zrg_ppedataset], axis=1) 


# In[28]:


zrg_ppedataset #for both DY1 and DY2, should be 200 wide (2 seasons of 4 variables of geographic averages) and number of runs tall


# In[29]:


#calculating the geographic averages for observations as well
DY1_PCP_zonal_obs = dict()
DY1_TLWP_zonal_obs = dict()
DY1_OSR_zonal_obs = dict()
DY1_OLR_zonal_obs = dict()

DY1_PCP_regional_obs = dict()
DY1_TLWP_regional_obs = dict()
DY1_OSR_regional_obs = dict()
DY1_OLR_regional_obs = dict()

DY1_PCP_global_obs = []
DY1_TLWP_global_obs = []
DY1_OSR_global_obs = []
DY1_OLR_global_obs = []
   
DY1_PCP_zonal_obs['obs'] = zonal_means_native(DY1_PCP_obs, area, lat, lon)
DY1_TLWP_zonal_obs['obs'] = zonal_means_native(DY1_TLWP_obs, area, lat, lon)
DY1_OSR_zonal_obs['obs'] = zonal_means_native(DY1_OSR_obs, area, lat, lon)
DY1_OLR_zonal_obs['obs'] = zonal_means_native(DY1_OLR_obs, area, lat, lon)
    
DY1_PCP_regional_obs['obs'] = regional_means_native(DY1_PCP_obs, area)
DY1_TLWP_regional_obs['obs'] = regional_means_native(DY1_TLWP_obs, area)
DY1_OSR_regional_obs['obs'] = regional_means_native(DY1_OSR_obs, area)
DY1_OLR_regional_obs['obs'] = regional_means_native(DY1_OLR_obs, area)    
    
DY1_PCP_global_obs.append(global_means_native(DY1_PCP_obs, area))
DY1_TLWP_global_obs.append(global_means_native(DY1_TLWP_obs, area))
DY1_OSR_global_obs.append(global_means_native(DY1_OSR_obs, area))
DY1_OLR_global_obs.append(global_means_native(DY1_OLR_obs, area))


# In[30]:


DY2_PCP_zonal_obs = dict()
DY2_TLWP_zonal_obs = dict()
DY2_OSR_zonal_obs = dict()
DY2_OLR_zonal_obs = dict()

DY2_PCP_regional_obs = dict()
DY2_TLWP_regional_obs = dict()
DY2_OSR_regional_obs = dict()
DY2_OLR_regional_obs = dict()

DY2_PCP_global_obs = []
DY2_TLWP_global_obs = []
DY2_OSR_global_obs = []
DY2_OLR_global_obs = []
   
DY2_PCP_zonal_obs['obs'] = zonal_means_native(DY2_PCP_obs, area, lat, lon)
DY2_TLWP_zonal_obs['obs'] = zonal_means_native(DY2_TLWP_obs, area, lat, lon)
DY2_OSR_zonal_obs['obs'] = zonal_means_native(DY2_OSR_obs, area, lat, lon)
DY2_OLR_zonal_obs['obs'] = zonal_means_native(DY2_OLR_obs, area, lat, lon)
    
DY2_PCP_regional_obs['obs'] = regional_means_native(DY2_PCP_obs, area)
DY2_TLWP_regional_obs['obs'] = regional_means_native(DY2_TLWP_obs, area)
DY2_OSR_regional_obs['obs'] = regional_means_native(DY2_OSR_obs, area)
DY2_OLR_regional_obs['obs'] = regional_means_native(DY2_OLR_obs, area)    
    
DY2_PCP_global_obs.append(global_means_native(DY2_PCP_obs, area))
DY2_TLWP_global_obs.append(global_means_native(DY2_TLWP_obs, area))
DY2_OSR_global_obs.append(global_means_native(DY2_OSR_obs, area))
DY2_OLR_global_obs.append(global_means_native(DY2_OLR_obs, area))


# In[31]:


DY1_PCP_obs_z_df = pd.DataFrame.from_dict(DY1_PCP_zonal_obs, orient="index")
DY1_PCP_obs_r_df = pd.DataFrame.from_dict(DY1_PCP_regional_obs, orient="index")
DY1_PCP_zrg_obs = pd.concat([DY1_PCP_obs_z_df, DY1_PCP_obs_r_df], axis=1)
DY1_PCP_zrg_obs["DY1_global"] = DY1_PCP_global_obs
DY2_PCP_obs_z_df = pd.DataFrame.from_dict(DY2_PCP_zonal_obs, orient="index")
DY2_PCP_obs_r_df = pd.DataFrame.from_dict(DY2_PCP_regional_obs, orient="index")
DY2_PCP_zrg_obs = pd.concat([DY2_PCP_obs_z_df, DY2_PCP_obs_r_df], axis=1)
DY2_PCP_zrg_obs["DY2_global"] = DY2_PCP_global_obs
PCP_zrg_obs = pd.concat([DY1_PCP_zrg_obs, DY2_PCP_zrg_obs], axis=1)

DY1_TLWP_obs_z_df = pd.DataFrame.from_dict(DY1_TLWP_zonal_obs, orient="index")
DY1_TLWP_obs_r_df = pd.DataFrame.from_dict(DY1_TLWP_regional_obs, orient="index")
DY1_TLWP_zrg_obs = pd.concat([DY1_TLWP_obs_z_df, DY1_TLWP_obs_r_df], axis=1)
DY1_TLWP_zrg_obs["DY1_global"] = DY1_TLWP_global_obs
DY2_TLWP_obs_z_df = pd.DataFrame.from_dict(DY2_TLWP_zonal_obs, orient="index")
DY2_TLWP_obs_r_df = pd.DataFrame.from_dict(DY2_TLWP_regional_obs, orient="index")
DY2_TLWP_zrg_obs = pd.concat([DY2_TLWP_obs_z_df, DY2_TLWP_obs_r_df], axis=1)
DY2_TLWP_zrg_obs["DY2_global"] = DY2_TLWP_global_obs
TLWP_zrg_obs = pd.concat([DY1_TLWP_zrg_obs, DY2_TLWP_zrg_obs], axis=1)

DY1_OSR_obs_z_df = pd.DataFrame.from_dict(DY1_OSR_zonal_obs, orient="index")
DY1_OSR_obs_r_df = pd.DataFrame.from_dict(DY1_OSR_regional_obs, orient="index")
DY1_OSR_zrg_obs = pd.concat([DY1_OSR_obs_z_df, DY1_OSR_obs_r_df], axis=1)
DY1_OSR_zrg_obs["DY1_global"] = DY1_OSR_global_obs
DY2_OSR_obs_z_df = pd.DataFrame.from_dict(DY2_OSR_zonal_obs, orient="index")
DY2_OSR_obs_r_df = pd.DataFrame.from_dict(DY2_OSR_regional_obs, orient="index")
DY2_OSR_zrg_obs = pd.concat([DY2_OSR_obs_z_df, DY2_OSR_obs_r_df], axis=1)
DY2_OSR_zrg_obs["DY2_global"] = DY2_OSR_global_obs
OSR_zrg_obs = pd.concat([DY1_OSR_zrg_obs, DY2_OSR_zrg_obs], axis=1)

DY1_OLR_obs_z_df = pd.DataFrame.from_dict(DY1_OLR_zonal_obs, orient="index")
DY1_OLR_obs_r_df = pd.DataFrame.from_dict(DY1_OLR_regional_obs, orient="index")
DY1_OLR_zrg_obs = pd.concat([DY1_OLR_obs_z_df, DY1_OLR_obs_r_df], axis=1)
DY1_OLR_zrg_obs["DY1_global"] = DY1_OLR_global_obs
DY2_OLR_obs_z_df = pd.DataFrame.from_dict(DY2_OLR_zonal_obs, orient="index")
DY2_OLR_obs_r_df = pd.DataFrame.from_dict(DY2_OLR_regional_obs, orient="index")
DY2_OLR_zrg_obs = pd.concat([DY2_OLR_obs_z_df, DY2_OLR_obs_r_df], axis=1)
DY2_OLR_zrg_obs["DY2_global"] = DY2_OLR_global_obs
OLR_zrg_obs = pd.concat([DY1_OLR_zrg_obs, DY2_OLR_zrg_obs], axis=1)

#### Ordering matters here
zrg_obs = pd.concat([PCP_zrg_obs, TLWP_zrg_obs, OSR_zrg_obs, OLR_zrg_obs], axis=1) 


# In[32]:


zrg_obs #should just be one row of the observations by 200


# ### K-fold cross validation
# These are mostly functions from sklearn because we wanted to explore a range of processing steps, but ESEm has built in functions that could make this simpler
#from ESEm
normaliser = Normalise()
X_train_norm = normaliser.process(X_train)
# In[37]:


folds = 5 # number of folds
#kf = KFold(n_splits=folds, shuffle=True, random_state=42)
kf = KFold(n_splits=folds, shuffle=True, random_state=2)
kf.get_n_splits(ppe_params)

k = 1 #just for example purposes, showing fold by fold, starting with the first one

for i, (train_index, test_index) in enumerate(kf.split(ppe_params)):
    if i == k:
        X_train = ppe_params.iloc[train_index]
        X_test = ppe_params.iloc[test_index]
        train_run_labels = X_train.index.to_list()
        test_run_labels = X_test.index.to_list()
        print('k =', i, test_run_labels)
        
        Y_train = ppe_dataset_small.sel(run_label=train_run_labels) #.to_array()
        Y_test = ppe_dataset_small.sel(run_label=test_run_labels) #.to_array()
        Y_train_array = ppe_dataset_small.sel(run_label=train_run_labels).to_array()
        Y_test_array = ppe_dataset_small.sel(run_label=test_run_labels).to_array()
        
        print("X_test shape:", X_test.shape, "type:", type(X_test))
        print("X_train shape:", X_train.shape, "type:", type(X_train))
        print("Y_test shape:", Y_test_array.shape, "type:", type(Y_test_array))
        print("Y_train shape:", Y_train_array.shape, "type:", type(Y_train_array))


# In[38]:


PCP_train = PCP_zrg_ppedataset.loc[train_run_labels] #**(1/8)
TLWP_train = TLWP_zrg_ppedataset.loc[train_run_labels] #**(1/4)
OSR_train = OSR_zrg_ppedataset.loc[train_run_labels] #**(1/4)
OLR_train = OLR_zrg_ppedataset.loc[train_run_labels] #**(1/8)

PCP_train.columns = PCP_zrg_ppedataset.columns.astype(str)
TLWP_train.columns = TLWP_zrg_ppedataset.columns.astype(str)
OSR_train.columns = OSR_zrg_ppedataset.columns.astype(str)
OLR_train.columns = OLR_zrg_ppedataset.columns.astype(str)

vars_train_list = [PCP_train, TLWP_train, OSR_train, OLR_train]


# In[39]:


PCP_test = PCP_zrg_ppedataset.loc[test_run_labels] #**(1/8)
TLWP_test = TLWP_zrg_ppedataset.loc[test_run_labels] #**(1/4)
OSR_test = OSR_zrg_ppedataset.loc[test_run_labels] #**(1/4)
OLR_test = OLR_zrg_ppedataset.loc[test_run_labels] #**(1/8)

PCP_test.columns = PCP_zrg_ppedataset.columns.astype(str)
TLWP_test.columns = TLWP_zrg_ppedataset.columns.astype(str)
OSR_test.columns = OSR_zrg_ppedataset.columns.astype(str)
OLR_test.columns = OLR_zrg_ppedataset.columns.astype(str)

vars_test_list = [PCP_test, TLWP_test, OSR_test, OLR_test]


# In[41]:


#for random forest, no preprocessing will be used
Y_train_ZRG = np.stack((PCP_train, TLWP_train, OSR_train, OLR_train), axis = 0)
Y_train_ZRG = np.transpose(Y_train_ZRG, (1, 2, 0))
Y_train_ZRG.shape

Y_test_ZRG = np.stack((PCP_test, TLWP_test, OSR_test, OLR_test), axis = 0)
Y_test_ZRG = np.transpose(Y_test_ZRG, (1, 2, 0))
Y_test_ZRG.shape


# In[44]:


#transform data
X_pipe_sk_minmax = preprocessing.MinMaxScaler()
X_pipe_sk_minmax.fit(X_train)
X_train_norm = X_pipe_sk_minmax.transform(X_train)

#from scikitlearn
Y_pipe_sk_ss_PCP = preprocessing.StandardScaler()
Y_pipe_sk_ss_PCP.fit(PCP_train)
PCP_train_norm = Y_pipe_sk_ss_PCP.transform(PCP_train)
PCP_test_norm = Y_pipe_sk_ss_PCP.transform(PCP_test)

Y_pipe_sk_ss_TLWP = preprocessing.StandardScaler() #lots of other options for this: RobustScaler(), etc.
Y_pipe_sk_ss_TLWP.fit(TLWP_train)
TLWP_train_norm = Y_pipe_sk_ss_TLWP.transform(TLWP_train)
TLWP_test_norm = Y_pipe_sk_ss_TLWP.transform(TLWP_test)

Y_pipe_sk_ss_OSR = preprocessing.StandardScaler()
Y_pipe_sk_ss_OSR.fit(OSR_train)
OSR_train_norm = Y_pipe_sk_ss_OSR.transform(OSR_train)
OSR_test_norm = Y_pipe_sk_ss_OSR.transform(OSR_test)

Y_pipe_sk_ss_OLR = preprocessing.StandardScaler()
Y_pipe_sk_ss_OLR.fit(OLR_train)
OLR_train_norm = Y_pipe_sk_ss_OLR.transform(OLR_train)
OLR_test_norm = Y_pipe_sk_ss_OLR.transform(OLR_test)

Y_train_norm = np.stack((PCP_train_norm, TLWP_train_norm, OSR_train_norm, OLR_train_norm), axis = 0)
Y_train_norm = np.transpose(Y_train_norm, (1, 2, 0))
print(X_train_norm.shape, Y_train_norm.shape)


# In[45]:


X_test_norm = X_pipe_sk_minmax.transform(X_test)
PCP_test_norm = Y_pipe_sk_ss_PCP.transform(PCP_test)
TLWP_test_norm = Y_pipe_sk_ss_TLWP.transform(TLWP_test)
OSR_test_norm = Y_pipe_sk_ss_OSR.transform(OSR_test)
OLR_test_norm = Y_pipe_sk_ss_OLR.transform(OLR_test)

Y_test_norm = np.stack((PCP_test_norm, TLWP_test_norm, OSR_test_norm, OLR_test_norm), axis = 0)
Y_test_norm = np.transpose(Y_test_norm, (1, 2, 0))
print(X_test_norm.shape, Y_test_norm.shape)


# #### Employing models

# In[46]:


print(X_train_norm.shape, Y_train_norm.shape)


# ##### Model

# In[59]:


model_gp = gp_model(X_train_norm, Y_train_norm)
#model_cnn = cnn_model(X_train_norm, Y_train_norm)
#model_rf = rf_model(X_train.to_numpy(), Y_train_ZRG) #not preprocessed


# In[50]:


model_gp.train()


# ##### $R^2$ Score

# In[51]:


m_gp, v_gp = model_gp.predict(X_test_norm)


# In[53]:


#vars_list = ['PCP','TLWP', 'OSR', 'OLR']
PCP_proj_norm_gp = pd.DataFrame(m_gp[:, :, 0], index = X_test.index)
TLWP_proj_norm_gp = pd.DataFrame(m_gp[:, :, 1], index = X_test.index)
OSR_proj_norm_gp = pd.DataFrame(m_gp[:, :, 2], index = X_test.index)
OLR_proj_norm_gp = pd.DataFrame(m_gp[:, :, 3], index = X_test.index)

PCP_v_proj_norm_gp = pd.DataFrame(v_gp[:, :, 0], index = X_test.index)
TLWP_v_proj_norm_gp = pd.DataFrame(v_gp[:, :, 1], index = X_test.index)
OSR_v_proj_norm_gp = pd.DataFrame(v_gp[:, :, 2], index = X_test.index)
OLR_v_proj_norm_gp = pd.DataFrame(v_gp[:, :, 3], index = X_test.index)


# In[54]:


#Normalied space
#Variance weighted -- lots of flexibility in the definition of this
PCP_r_squared_unflat = r2_score(PCP_test_norm, PCP_proj_norm_gp, multioutput='variance_weighted')
PCP_rmse_squared_unflat = root_mean_squared_error(PCP_test_norm, PCP_proj_norm_gp)

TLWP_r_squared_unflat = r2_score(TLWP_test_norm, TLWP_proj_norm_gp, multioutput='variance_weighted')
TLWP_rmse_squared_unflat = root_mean_squared_error(TLWP_test_norm, TLWP_proj_norm_gp)

OSR_r_squared_unflat = r2_score(OSR_test_norm, OSR_proj_norm_gp, multioutput='variance_weighted')
OSR_rmse_squared_unflat = root_mean_squared_error(OSR_test_norm, OSR_proj_norm_gp)

OLR_r_squared_unflat = r2_score(OLR_test_norm, OLR_proj_norm_gp, multioutput='variance_weighted')
OLR_rmse_squared_unflat = root_mean_squared_error(OLR_test_norm, OLR_proj_norm_gp)

print('PCP:', PCP_r_squared_unflat, 'TLWP:', TLWP_r_squared_unflat, 'OSR:', OSR_r_squared_unflat, 'OLR:', OLR_r_squared_unflat)
print('PCP:', PCP_rmse_squared_unflat, 'TLWP:', TLWP_rmse_squared_unflat, 'OSR:', OSR_rmse_squared_unflat, 'OLR:', OLR_rmse_squared_unflat)


# In[56]:


#Non-normalized space
PCP_proj_gp = pd.DataFrame(Y_pipe_sk_ss_PCP.inverse_transform(PCP_proj_norm_gp))
PCP_v_proj_gp = pd.DataFrame(Y_pipe_sk_ss_PCP.inverse_transform(PCP_proj_norm_gp))

TLWP_proj_gp = pd.DataFrame(Y_pipe_sk_ss_TLWP.inverse_transform(TLWP_proj_norm_gp))
TLWP_v_proj_gp = pd.DataFrame(Y_pipe_sk_ss_TLWP.inverse_transform(TLWP_proj_norm_gp))

OSR_proj_gp = pd.DataFrame(Y_pipe_sk_ss_OSR.inverse_transform(OSR_proj_norm_gp))
OSR_v_proj_gp = pd.DataFrame(Y_pipe_sk_ss_OSR.inverse_transform(OSR_proj_norm_gp))

OLR_proj_gp = pd.DataFrame(Y_pipe_sk_ss_OLR.inverse_transform(OLR_proj_norm_gp))
OLR_v_proj_gp = pd.DataFrame(Y_pipe_sk_ss_OLR.inverse_transform(OLR_proj_norm_gp))


# In[57]:


#Variance weighted -- lots of flexibility in the definition of this

#PCP_preds_flat = PCP_proj_gp.values.flatten()
#PCP_scream_flat = PCP_test.values.flatten()
#PCP_r_squared = r2_score(PCP_scream_flat, PCP_preds_flat)
#PCP_rmse_squared = root_mean_squared_error(PCP_scream_flat, PCP_preds_flat)
PCP_r_squared_unflat = r2_score(PCP_test, PCP_proj_gp, multioutput='variance_weighted') # = 'raw_values')
PCP_rmse_squared_unflat = root_mean_squared_error(PCP_test, PCP_proj_gp)

#TLWP_preds_flat = TLWP_proj_gp.values.flatten()
#TLWP_scream_flat = TLWP_test.values.flatten()
#TLWP_r_squared = r2_score(TLWP_scream_flat, TLWP_preds_flat)
#TLWP_rmse_squared = root_mean_squared_error(TLWP_scream_flat, TLWP_preds_flat)
TLWP_r_squared_unflat = r2_score(TLWP_test, TLWP_proj_gp, multioutput='variance_weighted')
TLWP_rmse_squared_unflat = root_mean_squared_error(TLWP_test, TLWP_proj_gp)

#OSR_preds_flat = OSR_proj_gp.values.flatten()
#OSR_scream_flat = OSR_test.values.flatten()
#OSR_r_squared = r2_score(OSR_scream_flat, OSR_preds_flat)
#OSR_rmse_squared = root_mean_squared_error(OSR_scream_flat, OSR_preds_flat)
OSR_r_squared_unflat = r2_score(OSR_test, OSR_proj_gp, multioutput='variance_weighted')
OSR_rmse_squared_unflat = root_mean_squared_error(OSR_test, OSR_proj_gp)

#OLR_preds_flat = OLR_proj_gp.values.flatten()
#OLR_scream_flat = OLR_test.values.flatten()
#OLR_r_squared = r2_score(OLR_scream_flat, OLR_preds_flat)
#OLR_rmse_squared = root_mean_squared_error(OLR_scream_flat, OLR_preds_flat)
#OLR_r_squared_unflat = r2_score(OLR_test, OLR_proj_gp)
OLR_r_squared_unflat = r2_score(OLR_test, OLR_proj_gp, multioutput='variance_weighted')
OLR_rmse_squared_unflat = root_mean_squared_error(OLR_test, OLR_proj_gp)

print('PCP:', PCP_r_squared_unflat, 'TLWP:', TLWP_r_squared_unflat, 'OSR:', OSR_r_squared_unflat, 'OLR:', OLR_r_squared_unflat)
print('PCP:', PCP_rmse_squared_unflat, 'TLWP:', TLWP_rmse_squared_unflat, 'OSR:', OSR_rmse_squared_unflat, 'OLR:', OLR_rmse_squared_unflat)


# In[24]:


save_dir = '/global/cfs/cdirs/e3sm/jpaige3/ESEm/TF_saving/'
save_name = save_dir + 'GPFlow_save' + str(date.today())
tf.saved_model.save(model_gp.model.model, save_name)
loaded_model = tf.saved_model.load(save_dir)


# In[33]:


save_name = '/global/cfs/cdirs/e3sm/jpaige3/ESEm/TF_saving/GPFlow_save2025-06-12'


# In[79]:


#recreating
loaded_model = tf.saved_model.load(save_name)
model_gp = loaded_model


# In[60]:


#creates a checkpoint at the final trained stage
checkpoint = tf.train.Checkpoint(model=model_gp.model.model)

#saves this checkpoint to GPFlow_save_ckpt with today's date
checkpoint_dir = '/global/cfs/cdirs/e3sm/jpaige3/ESEm/TF_checkpoint_saving/'
save_date = str(date.today())
checkpoint_savename = checkpoint_dir + 'GPFlow_save_ckpt' + save_date + 'Y_ss_wallrts'
checkpoint.save(file_prefix=checkpoint_savename)


# In[2]:


#recreating
model_gp_recreate = gp_model(X_train_norm, Y_train_norm) #create an instance of the model
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)) #recreate


# In[53]:


m_gp, v_gp = model_gp.predict(X_train_norm)


# In[55]:


#vars_list = ['PCP','TLWP', 'OSR', 'OLR']
PCP_proj_norm_gp = pd.DataFrame(m_gp[:, :, 0], index = X_train.index)
TLWP_proj_norm_gp = pd.DataFrame(m_gp[:, :, 1], index = X_train.index)
OSR_proj_norm_gp = pd.DataFrame(m_gp[:, :, 2], index = X_train.index)
OLR_proj_norm_gp = pd.DataFrame(m_gp[:, :, 3], index = X_train.index)

PCP_v_proj_norm_gp = pd.DataFrame(v_gp[:, :, 0], index = X_train.index)
TLWP_v_proj_norm_gp = pd.DataFrame(v_gp[:, :, 1], index = X_train.index)
OSR_v_proj_norm_gp = pd.DataFrame(v_gp[:, :, 2], index = X_train.index)
OLR_v_proj_norm_gp = pd.DataFrame(v_gp[:, :, 3], index = X_train.index)


# In[57]:


#NO TRANSPOSE!!
PCP_r_squared_unflat = r2_score(PCP_train_norm, PCP_proj_norm_gp, multioutput='raw_values') #'variance_weighted')
PCP_rmse_squared_unflat = root_mean_squared_error(PCP_train_norm, PCP_proj_norm_gp)

TLWP_r_squared_unflat = r2_score(TLWP_train_norm, TLWP_proj_norm_gp, multioutput='raw_values')
TLWP_rmse_squared_unflat = root_mean_squared_error(TLWP_train_norm, TLWP_proj_norm_gp)

OSR_r_squared_unflat = r2_score(OSR_train_norm, OSR_proj_norm_gp, multioutput='raw_values')
OSR_rmse_squared_unflat = root_mean_squared_error(OSR_train_norm, OSR_proj_norm_gp)

OLR_r_squared_unflat = r2_score(OLR_train_norm, OLR_proj_norm_gp, multioutput='raw_values')
OLR_rmse_squared_unflat = root_mean_squared_error(OLR_train_norm, OLR_proj_norm_gp)

print('PCP:', PCP_r_squared_unflat, 'TLWP:', TLWP_r_squared_unflat, 'OSR:', OSR_r_squared_unflat, 'OLR:', OLR_r_squared_unflat)
print('PCP:', PCP_rmse_squared_unflat, 'TLWP:', TLWP_rmse_squared_unflat, 'OSR:', OSR_rmse_squared_unflat, 'OLR:', OLR_rmse_squared_unflat)


# In[63]:


PCP_proj_gp = pd.DataFrame(Y_pipe_sk_ss_PCP.inverse_transform(PCP_proj_norm_gp))
PCP_v_proj_gp = pd.DataFrame(Y_pipe_sk_ss_PCP.inverse_transform(PCP_proj_norm_gp))

TLWP_proj_gp = pd.DataFrame(Y_pipe_sk_ss_TLWP.inverse_transform(TLWP_proj_norm_gp))
TLWP_v_proj_gp = pd.DataFrame(Y_pipe_sk_ss_TLWP.inverse_transform(TLWP_proj_norm_gp))

OSR_proj_gp = pd.DataFrame(Y_pipe_sk_ss_OSR.inverse_transform(OSR_proj_norm_gp))
OSR_v_proj_gp = pd.DataFrame(Y_pipe_sk_ss_OSR.inverse_transform(OSR_proj_norm_gp))

OLR_proj_gp = pd.DataFrame(Y_pipe_sk_ss_OLR.inverse_transform(OLR_proj_norm_gp))
OLR_v_proj_gp = pd.DataFrame(Y_pipe_sk_ss_OLR.inverse_transform(OLR_proj_norm_gp))


# In[60]:


#NO TRANSPOSE!!

#PCP_preds_flat = PCP_proj_gp.values.flatten()
#PCP_scream_flat = PCP_test.values.flatten()
#PCP_r_squared = r2_score(PCP_scream_flat, PCP_preds_flat)
#PCP_rmse_squared = root_mean_squared_error(PCP_scream_flat, PCP_preds_flat)
PCP_r_squared_unflat = r2_score(PCP_train, PCP_proj_gp, multioutput='variance_weighted') # = 'raw_values')
PCP_rmse_squared_unflat = root_mean_squared_error(PCP_train, PCP_proj_gp)

#TLWP_preds_flat = TLWP_proj_gp.values.flatten()
#TLWP_scream_flat = TLWP_test.values.flatten()
#TLWP_r_squared = r2_score(TLWP_scream_flat, TLWP_preds_flat)
#TLWP_rmse_squared = root_mean_squared_error(TLWP_scream_flat, TLWP_preds_flat)
TLWP_r_squared_unflat = r2_score(TLWP_train, TLWP_proj_gp, multioutput='variance_weighted')
TLWP_rmse_squared_unflat = root_mean_squared_error(TLWP_train, TLWP_proj_gp)

#OSR_preds_flat = OSR_proj_gp.values.flatten()
#OSR_scream_flat = OSR_test.values.flatten()
#OSR_r_squared = r2_score(OSR_scream_flat, OSR_preds_flat)
#OSR_rmse_squared = root_mean_squared_error(OSR_scream_flat, OSR_preds_flat)
OSR_r_squared_unflat = r2_score(OSR_train, OSR_proj_gp, multioutput='variance_weighted')
OSR_rmse_squared_unflat = root_mean_squared_error(OSR_train, OSR_proj_gp)

#OLR_preds_flat = OLR_proj_gp.values.flatten()
#OLR_scream_flat = OLR_test.values.flatten()
#OLR_r_squared = r2_score(OLR_scream_flat, OLR_preds_flat)
#OLR_rmse_squared = root_mean_squared_error(OLR_scream_flat, OLR_preds_flat)
#OLR_r_squared_unflat = r2_score(OLR_test, OLR_proj_gp)
OLR_r_squared_unflat = r2_score(OLR_train, OLR_proj_gp, multioutput='variance_weighted')
OLR_rmse_squared_unflat = root_mean_squared_error(OLR_train, OLR_proj_gp)

print('PCP:', PCP_r_squared_unflat, 'TLWP:', TLWP_r_squared_unflat, 'OSR:', OSR_r_squared_unflat, 'OLR:', OLR_r_squared_unflat)
print('PCP:', PCP_rmse_squared_unflat, 'TLWP:', TLWP_rmse_squared_unflat, 'OSR:', OSR_rmse_squared_unflat, 'OLR:', OLR_rmse_squared_unflat)


# In[65]:


proj_gp = np.stack((PCP_proj_gp.to_numpy(), TLWP_proj_gp.to_numpy(), OSR_proj_gp.to_numpy(), OLR_proj_gp.to_numpy()), axis = 2)
proj_v_gp = np.stack((PCP_v_proj_gp.to_numpy(), TLWP_v_proj_gp.to_numpy(), OSR_v_proj_gp.to_numpy(), OLR_v_proj_gp.to_numpy()), axis = 2)

#Checks
#type(X_test_norm), X_test_norm.shape, type(X_test.to_numpy()), X_test.to_numpy().shape
#type(Y_test_norm), Y_test_norm.shape, type(Y_test_ZRG), Y_test_ZRG.shape
#type(m_gp), m_gp.shape, type(proj_gp), proj_gp.shape
#type(v_gp), v_gp.shape, type(proj_v_gp), proj_v_gp.shape


# In[163]:


### Saving projections both normed and not and R2s
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#path - /global/cfs/cdirs/e3sm/jpaige3/ESEm/Final_projections_GP_CNN_RF
GP_proj_filename = f"Final_projections_GP_CNN_RF/GP_ZRG_proj_k={k}_{timestamp}.pkl"

#make dictionary to save
GP_data_to_save = {
    'X_pipeline': X_pipe_sk_minmax, 
    'Y_pipeline_PCP': Y_pipe_sk_ss_PCP, 
    'Y_pipeline_TLWP': Y_pipe_sk_ss_TLWP, 
    'Y_pipeline_OSR': Y_pipe_sk_ss_OSR, 
    'Y_pipeline_OLR': Y_pipe_sk_ss_OLR,
    ####
    'X_test_index': test_run_labels,
    ### normalized/transformed
    'X_test_norm': X_test_norm, 
    'Y_test_norm': Y_test_norm,
    'proj_norm_gp': m_gp,
    'proj_v_norm_gp': v_gp,
    ### unnormalized/untransformed
    'X_test': X_test.to_numpy(), 
    'Y_test': Y_test_ZRG,
    'proj_gp': proj_gp,
    'proj_v_gp': proj_v_gp,
}

#save using pickle
with open(GP_proj_filename, 'wb') as f:
    pickle.dump(GP_data_to_save, f)

'''
# Load back
with open(GP_proj_filename, 'rb') as f:
    loaded = pickle.load(f)

# Access data
restored_pipeline = loaded['X_pipeline']
restored_df1 = loaded['proj_gp']
'''


# In[164]:


# Saving R2 variance weighted
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
r2output_filename = f"R2_output_varweight/GP_ZRG_r2output_k={k}_{timestamp}.txt"

with open(r2output_filename, "w") as file:
    print('GP regionally trained, summer and winter, in variable units (untransformed), variance weighted', file=file)
    #print('r squared flat:', 'PCP:', PCP_r_squared, 'TLWP:', TLWP_r_squared, 'OSR:', OSR_r_squared, 'OLR:', OLR_r_squared, file=file)
    #print('     rmse flat:', 'PCP:', PCP_rmse_squared, 'TLWP:', TLWP_rmse_squared, 'OSR:', OSR_rmse_squared, 'OLR:', OLR_rmse_squared, file=file)
    print('r squared unflat:','PCP:', PCP_r_squared_unflat, 'TLWP:', TLWP_r_squared_unflat, 'OSR:', OSR_r_squared_unflat, 'OLR:', OLR_r_squared_unflat, file=file)
    print('     rmse unflat:','PCP:', PCP_rmse_squared_unflat, 'TLWP:', TLWP_rmse_squared_unflat, 'OSR:', OSR_rmse_squared_unflat, 'OLR:', OLR_rmse_squared_unflat, file=file)
    


# #### Trying to save the transforms

# In[31]:


model_gp.model.model


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


# In[5]:


# Load back
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


# ### Regression
# An aside about simpler regression techniques--linear and polynomial

# ##### MLR

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[40]:


PCP_MLR_model = LinearRegression()
PCP_MLR_model.fit(X_train_norm, Y_train_norm[:, :, 0])

TLWP_MLR_model = LinearRegression()
TLWP_MLR_model.fit(X_train_norm, Y_train_norm[:, :, 1])

OSR_MLR_model = LinearRegression()
OSR_MLR_model.fit(X_train_norm, Y_train_norm[:, :, 2])

OLR_MLR_model = LinearRegression()
OLR_MLR_model.fit(X_train_norm, Y_train_norm[:, :, 3])


# In[41]:


PCP_proj_norm_mlr = PCP_MLR_model.predict(X_test_norm)
TLWP_proj_norm_mlr = TLWP_MLR_model.predict(X_test_norm)
OSR_proj_norm_mlr = OSR_MLR_model.predict(X_test_norm)
OLR_proj_norm_mlr = OLR_MLR_model.predict(X_test_norm)


# In[42]:


proj_norm_mlr = np.stack((PCP_proj_norm_mlr, TLWP_proj_norm_mlr, OSR_proj_norm_mlr, OLR_proj_norm_mlr), axis = 2)


# In[43]:


PCP_r_squared_unflat = r2_score(PCP_test_norm, PCP_proj_norm_mlr, multioutput='variance_weighted')
PCP_rmse_squared_unflat = root_mean_squared_error(PCP_test_norm, PCP_proj_norm_mlr)

TLWP_r_squared_unflat = r2_score(TLWP_test_norm, TLWP_proj_norm_mlr, multioutput='variance_weighted')
TLWP_rmse_squared_unflat = root_mean_squared_error(TLWP_test_norm, TLWP_proj_norm_mlr)

OSR_r_squared_unflat = r2_score(OSR_test_norm, OSR_proj_norm_mlr, multioutput='variance_weighted')
OSR_rmse_squared_unflat = root_mean_squared_error(OSR_test_norm, OSR_proj_norm_mlr)

OLR_r_squared_unflat = r2_score(OLR_test_norm, OLR_proj_norm_mlr, multioutput='variance_weighted')
OLR_rmse_squared_unflat = root_mean_squared_error(OLR_test_norm, OLR_proj_norm_mlr)

print('PCP:', PCP_r_squared_unflat, 'TLWP:', TLWP_r_squared_unflat, 'OSR:', OSR_r_squared_unflat, 'OLR:', OLR_r_squared_unflat)
print('PCP:', PCP_rmse_squared_unflat, 'TLWP:', TLWP_rmse_squared_unflat, 'OSR:', OSR_rmse_squared_unflat, 'OLR:', OLR_rmse_squared_unflat)


# In[44]:


PCP_proj_mlr = pd.DataFrame(Y_pipe_sk_ss_PCP.inverse_transform(PCP_proj_norm_mlr))
TLWP_proj_mlr = pd.DataFrame(Y_pipe_sk_ss_TLWP.inverse_transform(TLWP_proj_norm_mlr))
OSR_proj_mlr = pd.DataFrame(Y_pipe_sk_ss_OSR.inverse_transform(OSR_proj_norm_mlr))
OLR_proj_mlr = pd.DataFrame(Y_pipe_sk_ss_OLR.inverse_transform(OLR_proj_norm_mlr))


# In[45]:


PCP_r_squared_unflat = r2_score(PCP_test, PCP_proj_mlr, multioutput='variance_weighted')
PCP_rmse_squared_unflat = root_mean_squared_error(PCP_test, PCP_proj_mlr)

TLWP_r_squared_unflat = r2_score(TLWP_test, TLWP_proj_mlr, multioutput='variance_weighted')
TLWP_rmse_squared_unflat = root_mean_squared_error(TLWP_test, TLWP_proj_mlr)

OSR_r_squared_unflat = r2_score(OSR_test, OSR_proj_mlr, multioutput='variance_weighted')
OSR_rmse_squared_unflat = root_mean_squared_error(OSR_test, OSR_proj_mlr)

OLR_r_squared_unflat = r2_score(OLR_test, OLR_proj_mlr, multioutput='variance_weighted')
OLR_rmse_squared_unflat = root_mean_squared_error(OLR_test, OLR_proj_mlr)

print('PCP:', PCP_r_squared_unflat, 'TLWP:', TLWP_r_squared_unflat, 'OSR:', OSR_r_squared_unflat, 'OLR:', OLR_r_squared_unflat)
print('PCP:', PCP_rmse_squared_unflat, 'TLWP:', TLWP_rmse_squared_unflat, 'OSR:', OSR_rmse_squared_unflat, 'OLR:', OLR_rmse_squared_unflat)


# ##### Polynomial regression

# In[77]:


from sklearn.preprocessing import PolynomialFeatures
degree = 1


# In[78]:


PCP_MPR_model = make_pipeline(PolynomialFeatures(degree = degree, interaction_only=False, include_bias=True), LinearRegression())
PCP_MPR_model.fit(X_train_norm, Y_train_norm[:, :, 0])

TLWP_MPR_model = make_pipeline(PolynomialFeatures(degree = degree, interaction_only=False, include_bias=True), LinearRegression())
TLWP_MPR_model.fit(X_train_norm, Y_train_norm[:, :, 1])

OSR_MPR_model = make_pipeline(PolynomialFeatures(degree = degree, interaction_only=False, include_bias=True), LinearRegression())
OSR_MPR_model.fit(X_train_norm, Y_train_norm[:, :, 2])

OLR_MPR_model = make_pipeline(PolynomialFeatures(degree = degree, interaction_only=False, include_bias=True), LinearRegression())
OLR_MPR_model.fit(X_train_norm, Y_train_norm[:, :, 3])


# In[79]:


PCP_proj_norm_mpr = PCP_MPR_model.predict(X_test_norm)
TLWP_proj_norm_mpr = TLWP_MPR_model.predict(X_test_norm)
OSR_proj_norm_mpr = OSR_MPR_model.predict(X_test_norm)
OLR_proj_norm_mpr = OLR_MPR_model.predict(X_test_norm)


# In[80]:


proj_norm_mpr = np.stack((PCP_proj_norm_mpr, TLWP_proj_norm_mpr, OSR_proj_norm_mpr, OLR_proj_norm_mpr), axis = 2)


# In[81]:


PCP_r_squared_unflat = r2_score(PCP_test_norm, PCP_proj_norm_mpr, multioutput='variance_weighted')
PCP_rmse_squared_unflat = root_mean_squared_error(PCP_test_norm, PCP_proj_norm_mpr)

TLWP_r_squared_unflat = r2_score(TLWP_test_norm, TLWP_proj_norm_mpr, multioutput='variance_weighted')
TLWP_rmse_squared_unflat = root_mean_squared_error(TLWP_test_norm, TLWP_proj_norm_mpr)

OSR_r_squared_unflat = r2_score(OSR_test_norm, OSR_proj_norm_mpr, multioutput='variance_weighted')
OSR_rmse_squared_unflat = root_mean_squared_error(OSR_test_norm, OSR_proj_norm_mpr)

OLR_r_squared_unflat = r2_score(OLR_test_norm, OLR_proj_norm_mpr, multioutput='variance_weighted')
OLR_rmse_squared_unflat = root_mean_squared_error(OLR_test_norm, OLR_proj_norm_mpr)

print('PCP:', PCP_r_squared_unflat, 'TLWP:', TLWP_r_squared_unflat, 'OSR:', OSR_r_squared_unflat, 'OLR:', OLR_r_squared_unflat)
print('PCP:', PCP_rmse_squared_unflat, 'TLWP:', TLWP_rmse_squared_unflat, 'OSR:', OSR_rmse_squared_unflat, 'OLR:', OLR_rmse_squared_unflat)


# In[82]:


PCP_proj_mpr = pd.DataFrame(Y_pipe_sk_ss_PCP.inverse_transform(PCP_proj_norm_mpr))
TLWP_proj_mpr = pd.DataFrame(Y_pipe_sk_ss_TLWP.inverse_transform(TLWP_proj_norm_mpr))
OSR_proj_mpr = pd.DataFrame(Y_pipe_sk_ss_OSR.inverse_transform(OSR_proj_norm_mpr))
OLR_proj_mpr = pd.DataFrame(Y_pipe_sk_ss_OLR.inverse_transform(OLR_proj_norm_mpr))


# In[83]:


PCP_r_squared_unflat = r2_score(PCP_test, PCP_proj_mpr, multioutput='variance_weighted')
PCP_rmse_squared_unflat = root_mean_squared_error(PCP_test, PCP_proj_mpr)

TLWP_r_squared_unflat = r2_score(TLWP_test, TLWP_proj_mpr, multioutput='variance_weighted')
TLWP_rmse_squared_unflat = root_mean_squared_error(TLWP_test, TLWP_proj_mpr)

OSR_r_squared_unflat = r2_score(OSR_test, OSR_proj_mpr, multioutput='variance_weighted')
OSR_rmse_squared_unflat = root_mean_squared_error(OSR_test, OSR_proj_mpr)

OLR_r_squared_unflat = r2_score(OLR_test, OLR_proj_mpr, multioutput='variance_weighted')
OLR_rmse_squared_unflat = root_mean_squared_error(OLR_test, OLR_proj_mpr)

print('PCP:', PCP_r_squared_unflat, 'TLWP:', TLWP_r_squared_unflat, 'OSR:', OSR_r_squared_unflat, 'OLR:', OLR_r_squared_unflat)
print('PCP:', PCP_rmse_squared_unflat, 'TLWP:', TLWP_rmse_squared_unflat, 'OSR:', OSR_rmse_squared_unflat, 'OLR:', OLR_rmse_squared_unflat)


# ##### Spline regression

# In[61]:


from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression


# In[62]:


#knots = 6
#degree = 1

PCP_knots = 3
PCP_degree = 2
TLWP_knots = 2
TLWP_degree = 3
OSR_knots = 2
OSR_degree = 3
OLR_knots = 2
OLR_degree = 2

PCP_spline_model = make_pipeline(SplineTransformer(n_knots=PCP_knots, degree=PCP_degree,include_bias=True), LinearRegression())
PCP_spline_model.fit(X_train_norm, Y_train_norm[:, :, 0])

TLWP_spline_model = make_pipeline(SplineTransformer(n_knots=TLWP_knots, degree=TLWP_degree,include_bias=True), LinearRegression())
TLWP_spline_model.fit(X_train_norm, Y_train_norm[:, :, 1])

OSR_spline_model = make_pipeline(SplineTransformer(n_knots=OSR_knots, degree=OSR_degree,include_bias=True), LinearRegression())
OSR_spline_model.fit(X_train_norm, Y_train_norm[:, :, 2])

OLR_spline_model = make_pipeline(SplineTransformer(n_knots=OLR_knots, degree=OLR_degree,include_bias=True), LinearRegression())
OLR_spline_model.fit(X_train_norm, Y_train_norm[:, :, 3])


# In[63]:


PCP_proj_norm_spline = PCP_spline_model.predict(X_test_norm)
TLWP_proj_norm_spline = TLWP_spline_model.predict(X_test_norm)
OSR_proj_norm_spline = OSR_spline_model.predict(X_test_norm)
OLR_proj_norm_spline = OLR_spline_model.predict(X_test_norm)


# ##### All mostly on the same range -- should help with optimizing

# In[47]:


np.min(PCP_train_norm), np.max(PCP_train_norm)


# In[48]:


np.min(TLWP_train_norm), np.max(TLWP_train_norm)


# In[49]:


np.min(OSR_train_norm), np.max(OSR_train_norm)


# In[50]:


np.min(OLR_train_norm), np.max(OLR_train_norm)


# ##### Model structure

# In[82]:


actual_model = model_gp.model.model


# In[72]:


actual_model

# Idk about this bro - why are there 7, but only 3 with dim 16
for var in actual_model.trainable_variables:
    #var.trainable = False
    print(var)
# ##### Weights and Cost function

# In[41]:


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


# In[42]:


lat_bands = np.linspace(-90,90,18) 

PCP_guess = PCP_spline_model.predict(optimal_params_final[0:-1].reshape(1, -1))
TLWP_guess = TLWP_spline_model.predict(optimal_params_final[0:-1].reshape(1, -1))
OSR_guess = OSR_spline_model.predict(optimal_params_final[0:-1].reshape(1, -1))
OLR_guess = OLR_spline_model.predict(optimal_params_final[0:-1].reshape(1, -1))

np.expand_dims(np.vstack((PCP_guess, TLWP_guess)), axis = 0).transpose(0,2,1).shape
# In[53]:


def params_to_cost_spline(params_guess): #, obs, area_weights, var_weights_dict, zrg_weights_dict):
    PCP_guess = PCP_spline_model.predict(params_guess.reshape(1, -1))
    TLWP_guess = TLWP_spline_model.predict(params_guess.reshape(1, -1))
    OSR_guess = OSR_spline_model.predict(params_guess.reshape(1, -1))
    OLR_guess = OLR_spline_model.predict(params_guess.reshape(1, -1))
    m_spline_guess = np.expand_dims(np.vstack((PCP_guess, TLWP_guess, OSR_guess, OLR_guess)), axis = 0).transpose(0,2,1)
    cost = ZRG_cost_function_rmse(m_spline_guess, obs_norm, var_weights_dict, zrg_weights_dict) #area_weights,
    return cost


# In[43]:


def params_to_cost_actually_good(params_guess): #, obs, area_weights, var_weights_dict, zrg_weights_dict):
    # Enforce parameter constraints (hard-coded for now)
    #if the lambda high and lambda low are not feasible
    
    #CHECK ON IF UNTRANSFORM IS NEEDED OR IF TRANSFORM IS THE SAME FOR BOTH! - i think it is not
    untransformed_params_guess = X_pipe_sk_minmax.inverse_transform(params_guess.reshape(1, -1))[0]
    ### NEED these indices to match to lambda low and lambda high
    if untransformed_params_guess[6] > untransformed_params_guess[7]:
    #if params_guess['lambda_low'] > params_guess['lambda_high']
        cost = 1e100
    else:
        m_gp_guess, v_gp_guess = model_gp.predict(params_guess.reshape(1, -1))
        cost = ZRG_cost_function(m_gp_guess, obs_norm, var_weights_dict, zrg_weights_dict) #area_weights,
    return cost

def params_to_cost(params_guess): #, obs, area_weights, var_weights_dict, zrg_weights_dict):
    m_gp_guess, v_gp_guess = model_gp.predict(params_guess.reshape(1, -1))
    cost = ZRG_cost_function_rmse(m_gp_guess, obs_norm, var_weights_dict, zrg_weights_dict) #area_weights,
    return cost

def params_to_cost_print(params_guess): #, obs, area_weights, var_weights_dict, zrg_weights_dict):
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

PCP_train_norm[:,0:17]

PCP_zrg_obs.iloc[:,0:z_num] #dy1 zones
PCP_zrg_obs.iloc[:,(z_num):(z_num+r_num)] #dy1 regions
PCP_zrg_obs.iloc[:,(z_num):(all_num-1)] #dy1 regions
PCP_zrg_obs.iloc[:,all_num-1] #dy1 global

PCP_zrg_obs.iloc[:,(all_num):(all_num+z_num)] #dy2 zones
PCP_zrg_obs.iloc[:,(all_num+z_num):(all_num+z_num+r_num)] #dy2 regions
PCP_zrg_obs.iloc[:,-1] #dy2 global
# In[44]:


`def ZRG_cost_function(preds, obs, var_weights_dict, zrg_weights_dict): #area_weights,
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
    
    total_PCP_cost = abs(PCP_obs_c - PCP_proj_c)
    total_TLWP_cost = abs(TLWP_obs_c - TLWP_proj_c)
    total_OSR_cost = abs(OSR_obs_c - OSR_proj_c)
    total_OLR_cost = abs(OLR_obs_c - OLR_proj_c)

    DY1_zonal_cost = zrg_weights_dict['zonal']*(zrg_weights_dict['zonal']*np.mean(var_weights_dict['PCP']*np.nanmean(total_PCP_cost[:, 0:z_num])
                                                   +var_weights_dict['TLWP']*np.nanmean(total_TLWP_cost[:, 0:z_num])
                                                   +var_weights_dict['OSR']*np.nanmean(total_OSR_cost[:, 0:z_num])
                                                   +var_weights_dict['OLR']*np.nanmean(total_OLR_cost[:, 0:z_num])))
    DY2_zonal_cost = zrg_weights_dict['zonal']*(zrg_weights_dict['zonal']*np.mean(var_weights_dict['PCP']*np.nanmean(total_PCP_cost[:,(all_num):(all_num+z_num)])
                                                   +var_weights_dict['TLWP']*np.nanmean(total_TLWP_cost[:,(all_num):(all_num+z_num)])
                                                   +var_weights_dict['OSR']*np.nanmean(total_OSR_cost[:,(all_num):(all_num+z_num)])
                                                   +var_weights_dict['OLR']*np.nanmean(total_OLR_cost[:,(all_num):(all_num+z_num)])))
    
    DY1_regional_cost = zrg_weights_dict['regional']*np.mean(var_weights_dict['PCP']*np.nanmean(total_PCP_cost[:,(z_num):(z_num+r_num)])
                                                         +var_weights_dict['TLWP']*np.nanmean(total_TLWP_cost[:,(z_num):(z_num+r_num)])
                                                         +var_weights_dict['OSR']*np.nanmean(total_OSR_cost[:,(z_num):(z_num+r_num)])
                                                         +var_weights_dict['OLR']*np.nanmean(total_OLR_cost[:,(z_num):(z_num+r_num)]))
    DY2_regional_cost = zrg_weights_dict['regional']*np.mean(var_weights_dict['PCP']*np.nanmean(total_PCP_cost[:,(all_num+z_num):(all_num+z_num+r_num)])
                                                         +var_weights_dict['TLWP']*np.nanmean(total_TLWP_cost[:,(all_num+z_num):(all_num+z_num+r_num)])
                                                         +var_weights_dict['OSR']*np.nanmean(total_OSR_cost[:,(all_num+z_num):(all_num+z_num+r_num)])
                                                         +var_weights_dict['OLR']*np.nanmean(total_OLR_cost[:,(all_num+z_num):(all_num+z_num+r_num)]))
    
    DY1_global_cost = zrg_weights_dict['global']*np.mean(var_weights_dict['PCP']*total_PCP_cost[:,all_num-1]
                                                     +var_weights_dict['TLWP']*total_TLWP_cost[:,all_num-1]
                                                     +var_weights_dict['OSR']*total_OSR_cost[:,all_num-1]
                                                     +var_weights_dict['OLR']*total_OLR_cost[:,all_num-1])
    DY2_global_cost = zrg_weights_dict['global']*np.mean(var_weights_dict['PCP']*total_PCP_cost[:,-1]
                                                     +var_weights_dict['TLWP']*total_TLWP_cost[:,-1]
                                                     +var_weights_dict['OSR']*total_OSR_cost[:,-1]
                                                     +var_weights_dict['OLR']*total_OLR_cost[:,-1])

    cost = DY_weights_dict['DY1']*(DY1_zonal_cost + DY1_regional_cost + DY1_global_cost) + DY_weights_dict['DY2']*(DY2_zonal_cost + DY2_regional_cost + DY2_global_cost)
    return cost


# In[45]:


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

    DY1_zonal_cost = zrg_weights_dict['zonal']*np.mean(var_weights_dict['PCP']*root_mean_squared_error(PCP_obs_c[0, 0:z_num], PCP_proj_c[0, 0:z_num])
                                                   +var_weights_dict['TLWP']*root_mean_squared_error(TLWP_obs_c[0, 0:z_num], TLWP_proj_c[0, 0:z_num])
                                                   +var_weights_dict['OSR']*root_mean_squared_error(OSR_obs_c[0, 0:z_num], OSR_proj_c[0, 0:z_num])
                                                   +var_weights_dict['OLR']*root_mean_squared_error(OLR_obs_c[0, 0:z_num], OLR_proj_c[0, 0:z_num]))
    DY2_zonal_cost = zrg_weights_dict['zonal']*np.mean(var_weights_dict['PCP']*root_mean_squared_error(PCP_obs_c[0,(all_num):(all_num+z_num)], PCP_proj_c[0,(all_num):(all_num+z_num)])
                                                   +var_weights_dict['TLWP']*root_mean_squared_error(TLWP_obs_c[0,(all_num):(all_num+z_num)], TLWP_proj_c[0,(all_num):(all_num+z_num)])
                                                   +var_weights_dict['OSR']*root_mean_squared_error(OSR_obs_c[0,(all_num):(all_num+z_num)], OSR_proj_c[0,(all_num):(all_num+z_num)])
                                                   +var_weights_dict['OLR']*root_mean_squared_error(OLR_obs_c[0,(all_num):(all_num+z_num)], OLR_proj_c[0,(all_num):(all_num+z_num)]))

    DY1_regional_cost = zrg_weights_dict['regional']*np.mean(var_weights_dict['PCP']*root_mean_squared_error(PCP_obs_c[0,(z_num):(z_num+r_num)], PCP_proj_c[0,(z_num):(z_num+r_num)])
                                                         +var_weights_dict['TLWP']*root_mean_squared_error(TLWP_obs_c[0,(z_num):(z_num+r_num)], TLWP_proj_c[0,(z_num):(z_num+r_num)])
                                                         +var_weights_dict['OSR']*root_mean_squared_error(OSR_obs_c[0,(z_num):(z_num+r_num)], OSR_proj_c[0,(z_num):(z_num+r_num)])
                                                         +var_weights_dict['OLR']*root_mean_squared_error(OLR_obs_c[0,(z_num):(z_num+r_num)], OLR_proj_c[0,(z_num):(z_num+r_num)]))
    DY2_regional_cost = zrg_weights_dict['regional']*np.mean(var_weights_dict['PCP']*root_mean_squared_error(PCP_obs_c[0,(all_num+z_num):(all_num+z_num+r_num)], PCP_proj_c[0,(all_num+z_num):(all_num+z_num+r_num)])
                                                         +var_weights_dict['TLWP']*root_mean_squared_error(TLWP_obs_c[0,(all_num+z_num):(all_num+z_num+r_num)], TLWP_proj_c[0,(all_num+z_num):(all_num+z_num+r_num)])
                                                         +var_weights_dict['OSR']*root_mean_squared_error(OSR_obs_c[0,(all_num+z_num):(all_num+z_num+r_num)], OSR_proj_c[0,(all_num+z_num):(all_num+z_num+r_num)])
                                                         +var_weights_dict['OLR']*root_mean_squared_error(OLR_obs_c[0,(all_num+z_num):(all_num+z_num+r_num)], OLR_proj_c[0,(all_num+z_num):(all_num+z_num+r_num)]))
    
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


# In[57]:


def ZRG_cost_function_untransform(preds_untransform, obs_untransform, var_weights_dict, zrg_weights_dict): #area_weights,
    PCP_proj_c = preds_untransform[:, :, 0] #this is a numpy array
    TLWP_proj_c = preds_untransform[:, :, 1]
    OSR_proj_c = preds_untransform[:, :, 2]
    OLR_proj_c = preds_untransform[:, :, 3]
    
    PCP_obs_c = obs_untransform[:, :, 0] #this is a numpy array
    TLWP_obs_c = obs_untransform[:, :, 1]
    OSR_obs_c = obs_untransform[:, :, 2]
    OLR_obs_c = obs_untransform[:, :, 3]
    
    z_num = len(lat_bands)
    r_num = len(regions_list)
    all_num = len(lat_bands) + len(regions_list) + 1
    
    total_PCP_cost = abs(PCP_proj_c - PCP_obs_c)
    total_TLWP_cost = abs(TLWP_proj_c - TLWP_obs_c)
    total_OSR_cost = abs(OSR_proj_c - OSR_obs_c)
    total_OLR_cost = abs(OLR_proj_c - OLR_obs_c)
    
    DY1_zonal_cost = zrg_weights_dict['zonal']*(zrg_weights_dict['zonal']*np.mean(var_weights_dict['PCP']*np.nanmean(total_PCP_cost[:, 0:z_num])
                                                   +var_weights_dict['TLWP']*np.nanmean(total_TLWP_cost[:, 0:z_num])
                                                   +var_weights_dict['OSR']*np.nanmean(total_OSR_cost[:, 0:z_num])
                                                   +var_weights_dict['OLR']*np.nanmean(total_OLR_cost[:, 0:z_num])))
    DY2_zonal_cost = zrg_weights_dict['zonal']*(zrg_weights_dict['zonal']*np.mean(var_weights_dict['PCP']*np.nanmean(total_PCP_cost[:,(all_num):(all_num+z_num)])
                                                   +var_weights_dict['TLWP']*np.nanmean(total_TLWP_cost[:,(all_num):(all_num+z_num)])
                                                   +var_weights_dict['OSR']*np.nanmean(total_OSR_cost[:,(all_num):(all_num+z_num)])
                                                   +var_weights_dict['OLR']*np.nanmean(total_OLR_cost[:,(all_num):(all_num+z_num)])))
    
    DY1_regional_cost = zrg_weights_dict['regional']*np.mean(var_weights_dict['PCP']*np.nanmean(total_PCP_cost[:,(z_num):(z_num+r_num)])
                                                         +var_weights_dict['TLWP']*np.nanmean(total_TLWP_cost[:,(z_num):(z_num+r_num)])
                                                         +var_weights_dict['OSR']*np.nanmean(total_OSR_cost[:,(z_num):(z_num+r_num)])
                                                         +var_weights_dict['OLR']*np.nanmean(total_OLR_cost[:,(z_num):(z_num+r_num)]))
    DY2_regional_cost = zrg_weights_dict['regional']*np.mean(var_weights_dict['PCP']*np.nanmean(total_PCP_cost[:,(all_num+z_num):(all_num+z_num+r_num)])
                                                         +var_weights_dict['TLWP']*np.nanmean(total_TLWP_cost[:,(all_num+z_num):(all_num+z_num+r_num)])
                                                         +var_weights_dict['OSR']*np.nanmean(total_OSR_cost[:,(all_num+z_num):(all_num+z_num+r_num)])
                                                         +var_weights_dict['OLR']*np.nanmean(total_OLR_cost[:,(all_num+z_num):(all_num+z_num+r_num)]))
    
    DY1_global_cost = zrg_weights_dict['global']*np.mean(var_weights_dict['PCP']*(total_PCP_cost[:,all_num-1])
                                                     +var_weights_dict['TLWP']*total_TLWP_cost[:,all_num-1]
                                                     +var_weights_dict['OSR']*total_OSR_cost[:,all_num-1]
                                                     +var_weights_dict['OLR']*total_OLR_cost[:,all_num-1])
    DY2_global_cost = zrg_weights_dict['global']*np.mean(var_weights_dict['PCP']*(total_PCP_cost[:,-1])
                                                     +var_weights_dict['TLWP']*total_TLWP_cost[:,-1]
                                                     +var_weights_dict['OSR']*total_OSR_cost[:,-1]
                                                     +var_weights_dict['OLR']*total_OLR_cost[:,-1])

    cost = DY_weights_dict['DY1']*(DY1_zonal_cost + DY1_regional_cost + DY1_global_cost) + DY_weights_dict['DY2']*(DY2_zonal_cost + DY2_regional_cost + DY2_global_cost)
    return cost


# In[46]:


def ZRG_cost_function_print(preds, obs, var_weights_dict, zrg_weights_dict): #area_weights,
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
    
    total_PCP_cost = abs(PCP_proj_c - PCP_obs_c)
    total_TLWP_cost = abs(TLWP_proj_c - TLWP_obs_c)
    total_OSR_cost = abs(OSR_proj_c - OSR_obs_c)
    total_OLR_cost = abs(OLR_proj_c - OLR_obs_c)
    
    DY1_zonal_cost = zrg_weights_dict['zonal']*(zrg_weights_dict['zonal']*np.mean(var_weights_dict['PCP']*np.nanmean(total_PCP_cost[:, 0:z_num])
                                                   +var_weights_dict['TLWP']*np.nanmean(total_TLWP_cost[:, 0:z_num])
                                                   +var_weights_dict['OSR']*np.nanmean(total_OSR_cost[:, 0:z_num])
                                                   +var_weights_dict['OLR']*np.nanmean(total_OLR_cost[:, 0:z_num])))
    DY2_zonal_cost = zrg_weights_dict['zonal']*(zrg_weights_dict['zonal']*np.mean(var_weights_dict['PCP']*np.nanmean(total_PCP_cost[:,(all_num):(all_num+z_num)])
                                                   +var_weights_dict['TLWP']*np.nanmean(total_TLWP_cost[:,(all_num):(all_num+z_num)])
                                                   +var_weights_dict['OSR']*np.nanmean(total_OSR_cost[:,(all_num):(all_num+z_num)])
                                                   +var_weights_dict['OLR']*np.nanmean(total_OLR_cost[:,(all_num):(all_num+z_num)])))
    
    DY1_regional_cost = zrg_weights_dict['regional']*np.mean(var_weights_dict['PCP']*np.nanmean(total_PCP_cost[:,(z_num):(z_num+r_num)])
                                                         +var_weights_dict['TLWP']*np.nanmean(total_TLWP_cost[:,(z_num):(z_num+r_num)])
                                                         +var_weights_dict['OSR']*np.nanmean(total_OSR_cost[:,(z_num):(z_num+r_num)])
                                                         +var_weights_dict['OLR']*np.nanmean(total_OLR_cost[:,(z_num):(z_num+r_num)]))
    DY2_regional_cost = zrg_weights_dict['regional']*np.mean(var_weights_dict['PCP']*np.nanmean(total_PCP_cost[:,(all_num+z_num):(all_num+z_num+r_num)])
                                                         +var_weights_dict['TLWP']*np.nanmean(total_TLWP_cost[:,(all_num+z_num):(all_num+z_num+r_num)])
                                                         +var_weights_dict['OSR']*np.nanmean(total_OSR_cost[:,(all_num+z_num):(all_num+z_num+r_num)])
                                                         +var_weights_dict['OLR']*np.nanmean(total_OLR_cost[:,(all_num+z_num):(all_num+z_num+r_num)]))
    
    DY1_global_cost = zrg_weights_dict['global']*np.mean(var_weights_dict['PCP']*(total_PCP_cost[:,all_num-1])
                                                     +var_weights_dict['TLWP']*total_TLWP_cost[:,all_num-1]
                                                     +var_weights_dict['OSR']*total_OSR_cost[:,all_num-1]
                                                     +var_weights_dict['OLR']*total_OLR_cost[:,all_num-1])
    DY2_global_cost = zrg_weights_dict['global']*np.mean(var_weights_dict['PCP']*(total_PCP_cost[:,-1])
                                                     +var_weights_dict['TLWP']*total_TLWP_cost[:,-1]
                                                     +var_weights_dict['OSR']*total_OSR_cost[:,-1]
                                                     +var_weights_dict['OLR']*total_OLR_cost[:,-1])

    cost = DY_weights_dict['DY1']*(DY1_zonal_cost + DY1_regional_cost + DY1_global_cost) + DY_weights_dict['DY2']*(DY2_zonal_cost + DY2_regional_cost + DY2_global_cost)
    
    print('PCP cost', np.mean(total_PCP_cost), 'TLWP cost', np.mean(total_TLWP_cost), 'OSR cost', np.mean(total_OSR_cost), 'OLR cost', np.mean(total_OLR_cost))                       
    print('zonal_cost', (DY1_zonal_cost+DY2_zonal_cost), 'regional_cost', (DY1_regional_cost+DY2_regional_cost), 'global_cost', (DY1_global_cost+DY2_global_cost))
    print('DY1 cost', (DY1_zonal_cost + DY1_regional_cost + DY1_global_cost), 'DY2 cost', (DY2_zonal_cost + DY2_regional_cost + DY2_global_cost))
    
    return cost

def cost_function(preds, obs, var_weights_dict, zrg_weights_dict): #area_weights,
    PCP_proj_c = preds[:, :, 0] #this is a numpy array
    TLWP_proj_c = preds[:, :, 1]
    OSR_proj_c = preds[:, :, 2]
    OLR_proj_c = preds[:, :, 3]
    
    PCP_obs_c = obs[:, :, 0] #this is a numpy array
    TLWP_obs_c = obs[:, :, 1]
    OSR_obs_c = obs[:, :, 2]
    OLR_obs_c = obs[:, :, 3]
    
    z_num = len(lat_bands)-1
    r_num = len(regions_list)
    
    total_PCP_cost = abs(PCP_proj_c - PCP_obs_c)
    total_TLWP_cost = abs(TLWP_proj_c - TLWP_obs_c)
    total_OSR_cost = abs(OSR_proj_c - OSR_obs_c)
    total_OLR_cost = abs(OLR_proj_c - OLR_obs_c)
    
    ### This needs to get updated
    zonal_cost = zrg_weights_dict['zonal']*np.mean(var_weights_dict['PCP']*np.nanmean(total_PCP_cost[:, 0:z_num])
                                                   +var_weights_dict['TLWP']*np.nanmean(total_TLWP_cost[:, 0:z_num])
                                                   +var_weights_dict['OSR']*np.nanmean(total_OSR_cost[:, 0:z_num])
                                                   +var_weights_dict['OLR']*np.nanmean(total_OLR_cost[:, 0:z_num]))
    regional_cost = zrg_weights_dict['regional']*np.mean(var_weights_dict['PCP']*np.nanmean(total_PCP_cost[:, (z_num+1):(z_num+1+r_num)])
                                                         +var_weights_dict['TLWP']*np.nanmean(total_TLWP_cost[:, (z_num+1):(z_num+1+r_num)])
                                                         +var_weights_dict['OSR']*np.nanmean(total_OSR_cost[:, (z_num+1):(z_num+1+r_num)])
                                                         +var_weights_dict['OLR']*np.nanmean(total_OLR_cost[:, (z_num+1):(z_num+1+r_num)]))
    global_cost = zrg_weights_dict['global']*np.mean(var_weights_dict['PCP']*total_PCP_cost[:,-1]
                                                     +var_weights_dict['TLWP']*total_TLWP_cost[:,-1]
                                                     +var_weights_dict['OSR']*total_OSR_cost[:,-1]
                                                     +var_weights_dict['OLR']*total_OLR_cost[:,-1])

    cost = zonal_cost + regional_cost + global_cost
    return costdef cost_function(preds, obs, var_weights_dict, zrg_weights_dict): #area_weights,
    PCP_proj_c = preds[:, :, 0] #this is a numpy array
    TLWP_proj_c = preds[:, :, 1]
    OSR_proj_c = preds[:, :, 2]
    OLR_proj_c = preds[:, :, 3]
    
    PCP_obs_c = obs[:, :, 0] #this is a numpy array
    TLWP_obs_c = obs[:, :, 1]
    OSR_obs_c = obs[:, :, 2]
    OLR_obs_c = obs[:, :, 3]
    
    z_num = len(lat_bands)-1
    r_num = len(regions_list)
    all_num = len(lat_bands) + len(regions_list)
    
    total_PCP_cost = abs(PCP_proj_c - PCP_obs_c)
    total_TLWP_cost = abs(TLWP_proj_c - TLWP_obs_c)
    total_OSR_cost = abs(OSR_proj_c - OSR_obs_c)
    total_OLR_cost = abs(OLR_proj_c - OLR_obs_c)
    
    DY1_zonal_cost = zrg_weights_dict['zonal']*(zrg_weights_dict['zonal']*np.mean(var_weights_dict['PCP']*np.nanmean(total_PCP_cost[:, 0:z_num])
                                                   +var_weights_dict['TLWP']*np.nanmean(total_TLWP_cost[:, 0:z_num])
                                                   +var_weights_dict['OSR']*np.nanmean(total_OSR_cost[:, 0:z_num])
                                                   +var_weights_dict['OLR']*np.nanmean(total_OLR_cost[:, 0:z_num])))
    DY2_zonal_cost = zrg_weights_dict['zonal']*(zrg_weights_dict['zonal']*np.mean(var_weights_dict['PCP']*np.nanmean(total_PCP_cost[:, all_num:(all_num+z_num)])
                                                   +var_weights_dict['TLWP']*np.nanmean(total_TLWP_cost[:, all_num:(all_num+z_num)])
                                                   +var_weights_dict['OSR']*np.nanmean(total_OSR_cost[:, all_num:(all_num+z_num)])
                                                   +var_weights_dict['OLR']*np.nanmean(total_OLR_cost[:, all_num:(all_num+z_num)])))
    
    DY1_regional_cost = zrg_weights_dict['regional']*np.mean(var_weights_dict['PCP']*np.nanmean(total_PCP_cost[:, (z_num+1):(z_num+1+r_num)])
                                                         +var_weights_dict['TLWP']*np.nanmean(total_TLWP_cost[:, (z_num+1):(z_num+1+r_num)])
                                                         +var_weights_dict['OSR']*np.nanmean(total_OSR_cost[:, (z_num+1):(z_num+1+r_num)])
                                                         +var_weights_dict['OLR']*np.nanmean(total_OLR_cost[:, (z_num+1):(z_num+1+r_num)]))
    DY2_regional_cost = zrg_weights_dict['regional']*np.mean(var_weights_dict['PCP']*np.nanmean(total_PCP_cost[:, (all_num+z_num+1):(all_num+z_num+1+r_num)])
                                                         +var_weights_dict['TLWP']*np.nanmean(total_TLWP_cost[:, (all_num+z_num+1):(all_num+z_num+1+r_num)])
                                                         +var_weights_dict['OSR']*np.nanmean(total_OSR_cost[:, (all_num+z_num+1):(all_num+z_num+1+r_num)])
                                                         +var_weights_dict['OLR']*np.nanmean(total_OLR_cost[:, (all_num+z_num+1):(all_num+z_num+1+r_num)]))
    
    DY1_global_cost = zrg_weights_dict['global']*np.mean(var_weights_dict['PCP']*(total_PCP_cost[:,all_num-1])
                                                     +var_weights_dict['TLWP']*total_TLWP_cost[:,all_num-1]
                                                     +var_weights_dict['OSR']*total_OSR_cost[:,all_num-1]
                                                     +var_weights_dict['OLR']*total_OLR_cost[:,all_num-1])
    DY2_global_cost = zrg_weights_dict['global']*np.mean(var_weights_dict['PCP']*(total_PCP_cost[:,-1])
                                                     +var_weights_dict['TLWP']*total_TLWP_cost[:,-1]
                                                     +var_weights_dict['OSR']*total_OSR_cost[:,-1]
                                                     +var_weights_dict['OLR']*total_OLR_cost[:,-1])

    cost = DY_weights_dict['DY1']*(DY1_zonal_cost + DY1_regional_cost + DY1_global_cost) + DY_weights_dict['DY2']*(DY2_zonal_cost + DY2_regional_cost + DY2_global_cost)
    return costdef cost_function_print(preds, obs, var_weights_dict, zrg_weights_dict): #area_weights,
    PCP_proj_c = preds[:, :, 0] #this is a numpy array
    TLWP_proj_c = preds[:, :, 1]
    OSR_proj_c = preds[:, :, 2]
    OLR_proj_c = preds[:, :, 3]
    
    PCP_obs_c = obs[:, :, 0] #this is a numpy array
    TLWP_obs_c = obs[:, :, 1]
    OSR_obs_c = obs[:, :, 2]
    OLR_obs_c = obs[:, :, 3]
    
    z_num = len(lat_bands)-1
    r_num = len(regions_list)
    all_num = len(lat_bands) + len(regions_list)
    
    total_PCP_cost = abs(PCP_proj_c - PCP_obs_c)
    total_TLWP_cost = abs(TLWP_proj_c - TLWP_obs_c)
    total_OSR_cost = abs(OSR_proj_c - OSR_obs_c)
    total_OLR_cost = abs(OLR_proj_c - OLR_obs_c)
    
    DY1_zonal_cost = zrg_weights_dict['zonal']*(zrg_weights_dict['zonal']*np.mean(var_weights_dict['PCP']*np.nanmean(total_PCP_cost[:, 0:z_num])
                                                   +var_weights_dict['TLWP']*np.nanmean(total_TLWP_cost[:, 0:z_num])
                                                   +var_weights_dict['OSR']*np.nanmean(total_OSR_cost[:, 0:z_num])
                                                   +var_weights_dict['OLR']*np.nanmean(total_OLR_cost[:, 0:z_num])))
    DY2_zonal_cost = zrg_weights_dict['zonal']*(zrg_weights_dict['zonal']*np.mean(var_weights_dict['PCP']*np.nanmean(total_PCP_cost[:, all_num:(all_num+z_num)])
                                                   +var_weights_dict['TLWP']*np.nanmean(total_TLWP_cost[:, all_num:(all_num+z_num)])
                                                   +var_weights_dict['OSR']*np.nanmean(total_OSR_cost[:, all_num:(all_num+z_num)])
                                                   +var_weights_dict['OLR']*np.nanmean(total_OLR_cost[:, all_num:(all_num+z_num)])))
    
    DY1_regional_cost = zrg_weights_dict['regional']*np.mean(var_weights_dict['PCP']*np.nanmean(total_PCP_cost[:, (z_num+1):(z_num+1+r_num)])
                                                         +var_weights_dict['TLWP']*np.nanmean(total_TLWP_cost[:, (z_num+1):(z_num+1+r_num)])
                                                         +var_weights_dict['OSR']*np.nanmean(total_OSR_cost[:, (z_num+1):(z_num+1+r_num)])
                                                         +var_weights_dict['OLR']*np.nanmean(total_OLR_cost[:, (z_num+1):(z_num+1+r_num)]))
    DY2_regional_cost = zrg_weights_dict['regional']*np.mean(var_weights_dict['PCP']*np.nanmean(total_PCP_cost[:, (all_num+z_num+1):(all_num+z_num+1+r_num)])
                                                         +var_weights_dict['TLWP']*np.nanmean(total_TLWP_cost[:, (all_num+z_num+1):(all_num+z_num+1+r_num)])
                                                         +var_weights_dict['OSR']*np.nanmean(total_OSR_cost[:, (all_num+z_num+1):(all_num+z_num+1+r_num)])
                                                         +var_weights_dict['OLR']*np.nanmean(total_OLR_cost[:, (all_num+z_num+1):(all_num+z_num+1+r_num)]))
    
    DY1_global_cost = zrg_weights_dict['global']*np.mean(var_weights_dict['PCP']*(total_PCP_cost[:,all_num-1])
                                                     +var_weights_dict['TLWP']*total_TLWP_cost[:,all_num-1]
                                                     +var_weights_dict['OSR']*total_OSR_cost[:,all_num-1]
                                                     +var_weights_dict['OLR']*total_OLR_cost[:,all_num-1])
    DY2_global_cost = zrg_weights_dict['global']*np.mean(var_weights_dict['PCP']*(total_PCP_cost[:,-1])
                                                     +var_weights_dict['TLWP']*total_TLWP_cost[:,-1]
                                                     +var_weights_dict['OSR']*total_OSR_cost[:,-1]
                                                     +var_weights_dict['OLR']*total_OLR_cost[:,-1])

    cost = DY_weights_dict['DY1']*(DY1_zonal_cost + DY1_regional_cost + DY1_global_cost) + DY_weights_dict['DY2']*(DY2_zonal_cost + DY2_regional_cost + DY2_global_cost)
                                     
    print('PCP cost', np.mean(total_PCP_cost), 'TLWP cost', np.mean(total_TLWP_cost), 'OSR cost', np.mean(total_OSR_cost), 'OLR cost', np.mean(total_OLR_cost))                       
    print('zonal_cost', (DY1_zonal_cost+DY2_zonal_cost), 'regional_cost', (DY1_regional_cost+DY2_regional_cost), 'global_cost', (DY1_global_cost+DY2_global_cost))
    print('DY1 cost', (DY1_zonal_cost + DY1_regional_cost + DY1_global_cost), 'DY2 cost', (DY2_zonal_cost + DY2_regional_cost + DY2_global_cost))
    return cost
# In[47]:


PCP_zrg_obs_named = PCP_zrg_obs #.columns.astype(str)
PCP_zrg_obs_named.columns = zrg_obs.iloc[:, 0:50].columns.astype(str) #PCP_train.columns
PCP_obs_norm = Y_pipe_sk_ss_PCP.transform(PCP_zrg_obs_named)

TLWP_zrg_obs_named = TLWP_zrg_obs #.columns.astype(str)
TLWP_zrg_obs_named.columns = zrg_obs.iloc[:, 0:50].columns.astype(str) #TLWP_train.columns
TLWP_obs_norm = Y_pipe_sk_ss_TLWP.transform(TLWP_zrg_obs_named)

OSR_zrg_obs_named = OSR_zrg_obs #.columns.astype(str)
OSR_zrg_obs_named.columns = zrg_obs.iloc[:, 0:50].columns.astype(str) #OSR_train.columns
OSR_obs_norm = Y_pipe_sk_ss_OSR.transform(OSR_zrg_obs_named)

OLR_zrg_obs_named = OLR_zrg_obs #.columns.astype(str)
OLR_zrg_obs_named.columns = zrg_obs.iloc[:, 0:50].columns.astype(str) #OLR_train.columns
OLR_obs_norm = Y_pipe_sk_ss_OLR.transform(OLR_zrg_obs_named)

obs_norm = np.stack([PCP_obs_norm, TLWP_obs_norm, OSR_obs_norm, OLR_obs_norm])
obs_norm = obs_norm.transpose(1, 2, 0)
obs_norm.shape

obs_untransform = np.stack([PCP_zrg_obs, TLWP_zrg_obs, OSR_zrg_obs, OLR_zrg_obs])
obs_untransform = obs_untransform.transpose(1, 2, 0)
obs_untransform.shape


# In[48]:


PCP_default_cost = PCP_train_norm[0] - PCP_obs_norm
TLWP_default_cost = TLWP_train_norm[0] - TLWP_obs_norm
OSR_default_cost = OSR_train_norm[0] - OSR_obs_norm
OLR_default_cost = OLR_train_norm[0] - OLR_obs_norm

default_cost = ZRG_cost_function_rmse(Y_train_norm[0][np.newaxis, :], obs_norm, var_weights_dict, zrg_weights_dict)
default_cost


# ##### Longer functions

# In[181]:


regions_file = xr.open_dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/regions.nc')
regions_list = ['poles','extratropical_land','extratropical_ocean','tropical_land','ascending_tropical_ocean','descending_tropical_ocean']
#area = ppe_dataset.area[1,:] #only taking the first row, because all rows should have the same values
control = xr.open_dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/SCREAM.2024-autocal-00.ne1024pg2/m0000/SCREAM.2024-autocal-00.ne1024pg2/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc')
area = control.variables['area'][:]
lat = control.variables['lat'][:]
lon = control.variables['lon'][:]
lat_bands = np.linspace(-90,90,19) #currently dividing globe in 18 zones - 10 degree bands


# In[182]:


def zonal_means_native(data, area, lat, lon):
    lat_bands = np.linspace(-90,90,19) #currently dividing globe in 18 zones - 10 degree bands
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


# In[183]:


def abs_error_dict(pred_dict, obs_dict, weights=None):
    # mean absolute error
    differences = [abs(pred_dict[key] - obs_dict[key]) for key in pred_dict]
    if weights is None:
        weights = 1/len(differences) 
    avg_diff = np.sum(np.dot(differences, weights))
    #avg_diff = sum(differences) / weights
    return avg_diff


# In[184]:


def abs_error(pred, obs, weights=None):
    # mean absolute error
    # y : numpy array of shape (ncol,) containing observations of a vectorized output variable
    # yhat : numpy array of shape (ncol,) containing predictions of a vectorized output variable
    if weights is None:
        weights = 1/len(pred)
    abs_out = np.sum(np.dot((pred - obs), weights))
    return abs_out

def rmse_error(y, yhat, weights=None):
    #rmse_out : a single-number calculation of RMSE, weighted by weights
    # y : numpy array of shape (ncol,) containing observations of a vectorized output variable
    # yhat : numpy array of shape (ncol,) containing predictions of a vectorized output variable

    if weights is None:
        weights = 1/len(y)
    rmse_out = np.sqrt(np.sum(area_weights * ((y - yhat) / s)**2))
    return rmse_out

def zonal_means_native(output_var_, lat, area_weights, lat_south, lat_north, dlat):
    # Use squeeze to deal with case of input data having an extra dimension of length 1
    output_var  = output_var_.squeeze()
    lat1        = lat_south
    lat2        = lat_north
    nbin        = np.round( ( lat2 - lat1 + dlat )/dlat ).astype(int)
    bin_coord   = np.linspace(lat1,lat2,nbin)
    bin_cnt     = np.zeros((nbin,))
    bin_val     = np.zeros((nbin,))
    bin_cnt[:]  = np.nan
    bin_val[:]  = np.nan
    condition   = np.full(np.shape(lat), False, dtype=bool)
    # Loop through latitude bins to calculate area-weighted zonal mean
    for b in range(nbin):
        bin_bot = lat1 - dlat/2. + dlat*(b  )
        bin_top = lat1 - dlat/2. + dlat*(b+1)
        condition = ( lat >=bin_bot )  &  ( lat < bin_top )
        bin_cnt[b] = np.sum(condition)
        if bin_cnt[b]>0:
            bin_val[b] = np.sum(output_var[condition] * area_weights[condition]) / np.sum(area_weights[condition])
    return bin_valdef get_zonal_avg(preds):
    lat_bands = N.linspace(-90,90,18) #currently dividing globe in 18 zones - 10 degree bands
    return zonal_avg

def get_regional_avg3(PCP_proj, TLWP_proj, OSR_proj, OLR_proj):
    SNF_control = S.Dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/SCREAM.2024-autocal-00.ne1024pg2/m0000/SCREAM.2024-autocal-00.ne1024pg2/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc')
    ar = SNF_control.variables['area'][:]
    lat = SNF_control.variables['lat'][:]
    lon = SNF_control.variables['lon'][:]
    
    regions_file = S.Dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/regions.nc')
    regions_list = ['poles','extratropical_land','extratropical_ocean','tropical_land','ascending_tropical_ocean','descending_tropical_ocean']
    regional_area_weighting = np.zeros(len(regions_list))
    regional_area_weighting_ocean = np.zeros(len(regions_list))
    
    lat_bands = np.linspace(-90,90,18) #this is where to change the refinement of the zones; start, end, # of partitions #Peter suggested 10 degree bands
    
    
    landfrac = regions_file.variables['landfrac'][0,:]
    dict_vars_regions_test = dict()
    #shape = (PCP_test.shape[0]+PCP_train.shape[0], PCP_test.shape[1]+PCP_train.shape[1])
    Y_test_shape = PCP_proj.shape

    for i in range(len(regions_list)): #for each region
        reg = regions_file.variables[regions_list[i]][0,:]
        reg = np.array(reg,dtype=float) #extract region mask
        region = np.zeros(Y_test_shape) #this is number of files by ncol
        ocean = np.zeros(Y_test_shape)
        oc = 1 - landfrac
        oc[oc < 0.5] = 0
        oc[oc >= 0.5] = 1
        oc[oc==0] = np.nan #oc is every cell that is less than half land (from landfrac)
        for j in range(Y_test_shape[0]):
            region[j,:] = np.array(reg,dtype=float)
            ocean[j,:] = 1 - np.array(landfrac,dtype=float)
        ocean[ocean > 0.5] = 1
        ocean[ocean < 0.5] = 0
        ocean[ocean==0] = np.nan
        region[region==0] = np.nan
        reg[reg==0] = np.nan
        regional_area_weighting[i] = np.nansum(ar*reg)
        regional_area_weighting_ocean[i] = np.nansum(ar*reg*oc)

        dict_vars_regions_test[regions_list[i]+'PCP'] = np.nanmean(PCP_proj*[ar]*region,(1))/np.nanmean(ar*region,(1))
        #dict_vars_regions_test[regions_list[i]+'TLWP'] = np.nanmean((CWP+LWP)*area*region,(1))/np.nanmean(area*region,(1))
        dict_vars_regions_test[regions_list[i]+'TLWP'] = np.nanmean(TLWP_proj*ar*region,(1))/np.nanmean(ar*region,(1))
        dict_vars_regions_test[regions_list[i]+'OSR'] =  np.nanmean(OSR_proj*ar*region,(1))/np.nanmean(ar*region,(1))
        dict_vars_regions_test[regions_list[i]+'OLR'] = np.nanmean(OLR_proj*ar*region,(1))/np.nanmean(ar*region,(1))
        
                dict_vars_regions[regions_list[i]+'P_obs'] = N.nanmean(P_obs*ar*reg)/N.nanmean(ar*reg)
        dict_vars_regions[regions_list[i]+'LWP_obs'] = N.nanmean((LWP_obs)*ar*reg*oc)/N.nanmean(ar*reg*oc)
        dict_vars_regions[regions_list[i]+'SW_obs'] = N.nanmean((SW_obs[0,:])*ar*reg)/N.nanmean(ar*reg)
        dict_vars_regions[regions_list[i]+'OLR_obs'] = N.nanmean((OLR_obs[0,:])*ar*reg)/N.nanmean(ar*reg)
        dict_vars_regions[regions_list[i]+'SW_obs_month'] = N.nanmean((SW_obs_month[0,:])*ar*reg)/N.nanmean(ar*reg)
        dict_vars_regions[regions_list[i]+'OLR_obs_month'] = N.nanmean((OLR_obs_month[0,:])*ar*reg)/N.nanmean(ar*reg)
        regional_bias_P[i,:] = dict_vars_regions[regions_list[i]+'P'] - dict_vars_regions[regions_list[i]+'P_obs']
        regional_bias_LWP[i,:] = dict_vars_regions[regions_list[i]+'LWP'] - dict_vars_regions[regions_list[i]+'LWP_obs']
        regional_bias_LWP[i,:] = N.nanmean((CWP+RWP)*area*region*ocean,(1))/N.nanmean(area*region*ocean,(1)) - dict_vars_regions[regions_list[i]+'LWP_obs']
        regional_bias_SW[i,:] = dict_vars_regions[regions_list[i]+'SW'] - dict_vars_regions[regions_list[i]+'SW_obs']
        regional_bias_OLR[i,:] = dict_vars_regions[regions_list[i]+'OLR'] - dict_vars_regions[regions_list[i]+'OLR_obs']
        regional_P[i,:] = dict_vars_regions[regions_list[i]+'P'] 
        regional_LWP[i,:] = dict_vars_regions[regions_list[i]+'LWP']
        regional_SW[i,:] = dict_vars_regions[regions_list[i]+'SW'] 
        regional_OLR[i,:] = dict_vars_regions[regions_list[i]+'OLR'] 

    return dict_vars_regions_test

def get_regional_avg2(PCP_proj, TLWP_proj, OSR_proj, OLR_proj):
    SNF_control = S.Dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/SCREAM.2024-autocal-00.ne1024pg2/m0000/SCREAM.2024-autocal-00.ne1024pg2/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc')
    ar = SNF_control.variables['area'][:]
    lat = SNF_control.variables['lat'][:]
    lon = SNF_control.variables['lon'][:]
    
    regions_file = S.Dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/regions.nc')
    regions_list = ['poles','extratropical_land','extratropical_ocean','tropical_land','ascending_tropical_ocean','descending_tropical_ocean']
    regional_area_weighting = np.zeros(len(regions_list))
    regional_area_weighting_ocean = np.zeros(len(regions_list))
    
    lat_bands = np.linspace(-90,90,18) #this is where to change the refinement of the zones; start, end, # of partitions #Peter suggested 10 degree bands
    
    landfrac = regions_file.variables['landfrac'][0,:]
    dict_vars_regions_test = dict()
    #shape = (PCP_test.shape[0]+PCP_train.shape[0], PCP_test.shape[1]+PCP_train.shape[1])
    Y_test_shape = PCP_proj.shape

    for i in range(len(regions_list)): #for each region
        reg = regions_file.variables[regions_list[i]][0,:]
        reg = np.array(reg,dtype=float) #extract region mask
        region = np.zeros(Y_test_shape) #this is number of files by ncol
        ocean = np.zeros(Y_test_shape)
        oc = 1 - landfrac
        oc[oc < 0.5] = 0
        oc[oc >= 0.5] = 1
        oc[oc==0] = np.nan #oc is every cell that is less than half land (from landfrac)
        for j in range(Y_test_shape[0]):
            region[j,:] = np.array(reg,dtype=float)
            ocean[j,:] = 1 - np.array(landfrac,dtype=float)
        ocean[ocean > 0.5] = 1
        ocean[ocean < 0.5] = 0
        ocean[ocean==0] = np.nan
        region[region==0] = np.nan
        reg[reg==0] = np.nan
        regional_area_weighting[i] = np.nansum(ar*reg)
        regional_area_weighting_ocean[i] = np.nansum(ar*reg*oc)

        dict_vars_regions_test[regions_list[i]+'PCP'] = np.nanmean(PCP_proj*ar*region,(1))/np.nanmean(ar*region,(1))
        #dict_vars_regions_test[regions_list[i]+'TLWP'] = np.nanmean((CWP+LWP)*area*region,(1))/np.nanmean(area*region,(1))
        dict_vars_regions_test[regions_list[i]+'TLWP'] = np.nanmean(TLWP_proj*ar*region,(1))/np.nanmean(ar*region,(1))
        dict_vars_regions_test[regions_list[i]+'OSR'] =  np.nanmean(OSR_proj*ar*region,(1))/np.nanmean(ar*region,(1))
        dict_vars_regions_test[regions_list[i]+'OLR'] = np.nanmean(OLR_proj*ar*region,(1))/np.nanmean(ar*region,(1))
        
                dict_vars_regions[regions_list[i]+'P_obs'] = N.nanmean(P_obs*ar*reg)/N.nanmean(ar*reg)
        dict_vars_regions[regions_list[i]+'LWP_obs'] = N.nanmean((LWP_obs)*ar*reg*oc)/N.nanmean(ar*reg*oc)
        dict_vars_regions[regions_list[i]+'SW_obs'] = N.nanmean((SW_obs[0,:])*ar*reg)/N.nanmean(ar*reg)
        dict_vars_regions[regions_list[i]+'OLR_obs'] = N.nanmean((OLR_obs[0,:])*ar*reg)/N.nanmean(ar*reg)
        dict_vars_regions[regions_list[i]+'SW_obs_month'] = N.nanmean((SW_obs_month[0,:])*ar*reg)/N.nanmean(ar*reg)
        dict_vars_regions[regions_list[i]+'OLR_obs_month'] = N.nanmean((OLR_obs_month[0,:])*ar*reg)/N.nanmean(ar*reg)
        regional_bias_P[i,:] = dict_vars_regions[regions_list[i]+'P'] - dict_vars_regions[regions_list[i]+'P_obs']
        regional_bias_LWP[i,:] = dict_vars_regions[regions_list[i]+'LWP'] - dict_vars_regions[regions_list[i]+'LWP_obs']
        regional_bias_LWP[i,:] = N.nanmean((CWP+RWP)*area*region*ocean,(1))/N.nanmean(area*region*ocean,(1)) - dict_vars_regions[regions_list[i]+'LWP_obs']
        regional_bias_SW[i,:] = dict_vars_regions[regions_list[i]+'SW'] - dict_vars_regions[regions_list[i]+'SW_obs']
        regional_bias_OLR[i,:] = dict_vars_regions[regions_list[i]+'OLR'] - dict_vars_regions[regions_list[i]+'OLR_obs']
        regional_P[i,:] = dict_vars_regions[regions_list[i]+'P'] 
        regional_LWP[i,:] = dict_vars_regions[regions_list[i]+'LWP']
        regional_SW[i,:] = dict_vars_regions[regions_list[i]+'SW'] 
        regional_OLR[i,:] = dict_vars_regions[regions_list[i]+'OLR'] 

    
    return dict_vars_regions_test

        
def get_global_avg(preds):
    return zonal_avg
# In[362]:


get_regional_avg2(PCP_proj_norm_gp, TLWP_proj_norm_gp, OSR_proj_norm_gp, OLR_proj_norm_gp)

#from old surrogate
def parallel_optimization(xstart):
    res = minimize(
        params_to_cost,
        xstart,
        method="L-BFGS-B",
        jac=None,
        bounds=X_s_bounds,
        options={"ftol": 1e-10, "maxiter": 70000, "disp": False}
    )
    return res.fun, res.x, res.nfev, res.nit

def optimize_params():
    rn = np.random.RandomState(optimization["seed"])
    R = int(mp.cpu_count())
    print(R, 'number of cpus')
    R = 10000 #1000
    print(R, 'R we are using')
    xstarts = 2 * rn.rand(R, n_inputs) - 1
    
    # Make sure we aren't starting out-of-bounds
    if optimization['param_start'] is None:
        idx_oob = np.where(xstarts[:,idx_lambda_low] > xstarts[:,idx_lambda_high])[0]
        while(len(idx_oob) > 0):
            xstarts[idx_oob,:] = 2 * rn.rand(len(idx_oob), n_inputs) - 1
            idx_oob = np.where(xstarts[:,idx_lambda_low] > xstarts[:,idx_lambda_high])[0]
    else:
        xstarts[:] = optimization['param_start']
   
    # optimize
    with mp.get_context("fork").Pool() as pool:
        results = pool.map(parallel_optimization, xstarts)
    
    
    # extract results
    fevals = [soln[0] for soln in results]
    xopts = [soln[1] for soln in results]
    optarg = np.argmin(fevals)
    xopt_s = xopts[optarg]
    xopt_ = feature_transform.inverse_transform(xopt_s)[0]
    
    '''
    optarg_sort = np.argsort(fevals)
    optarg3_indices = sorted_indices[:3]
    #xopt_s3 = xopts[optarg3]
    fevals3 = [fevals[i] for i in optarg3_indices]
    xopts3 = [xopts[i] for i in optarg3_indices]
    top_three_transformed = [feature_transform.inverse_transform(param)[0] for param in xopts3]
    '''
    
    sorted_indices = np.argsort(fevals)
    top_three_indices = sorted_indices[:10]
    top_three_values = [fevals[i] for i in top_three_indices]
    top_three_params = [xopts[i] for i in top_three_indices]
    top_three_transformed = [feature_transform.inverse_transform(param)[0] for param in top_three_params]
    
    return xopt_, xopt_s, top_three_transformed
    
def parallel_optimization(xstart):
    result = minimize(
        params_to_cost,
        xstart,
        method="L-BFGS-B",
        bounds=[(0.0, 1.0)],
        options={"ftol": 1e-10, "maxiter": 70000, "disp": True}
    )
    return res.fun, res.x, res.nfev, res.nit
# ##### some random checks

# In[496]:


np.mean(PCP_train[0,:].values), np.mean(TLWP_train[0,:].values), np.mean(OSR_train[0,:].values), np.mean(OLR_train[0,:].values), 


# In[497]:


np.mean(PCP_obs), np.mean(TLWP_obs), np.mean(OSR_obs), np.mean(OLR_obs)


# In[475]:


regional_means_native(PCP_proj_norm_gp, area)


# In[482]:


regional_means_native(PCP_obs_norm, area)


# In[484]:


zonal_means_native(PCP_proj_norm_gp, area, lat, lon)


# In[485]:


zonal_means_native(PCP_obs_norm, area, lat, lon)


# In[49]:


from esem.utils import validation_plot, plot_parameter_space, get_random_params, ensemble_collocate
from esem import gp_model
from esem.abc_sampler import ABCSampler, constrain


# In[ ]:


sampler = ABCSampler(model_gp, obs_norm)
samples = sampler.sample(n_samples=1, threshold=0.5)


# In[ ]:


valid_points = pd.DataFrame(data=samples, columns=X_train_norm.columns)


# In[ ]:


m, _ = model.predict(valid_points.values)
Zs = m.data


# In[ ]:


plot_parameter_space(valid_points, target_df=X_test.iloc[0])


# In[ ]:


print('hello


# In[ ]:





# In[ ]:


#collocate the model to observations 
col_ppe = ensemble_collocate(ppe_aaod, aaod)


# In[78]:


# Sample and constrain the models
#Emulating 1e6 sample points directly would require 673 Gb of memory so we can either run 1e6 samples for each point, or run the constraint everywhere, but in batches. 
#Here we do the latter, optioanlly on the GPU, using the ‘naive’ algorithm for calculating the running mean and variance of the various properties.
#The rejection sampling happens in a similar manner so that only as much memory as is used for one batch is ever used.

sample_points = pd.DataFrame(data=get_random_params(X_train.shape[1], int(1e6)), columns=X_train.columns)


# In[79]:


sample_points.shape


# In[80]:


# Note that smoothing the parameter distribution can be slow for large numbers of points
plot_parameter_space(sample_points, fig_size=(3,6), smooth=False)


# In[81]:


# Setup the sampler to compare against our AeroNet data
sampler = ABCSampler(model_gp, obs_norm, obs_uncertainty=0.5, repres_uncertainty=0.5)


# In[82]:


# Calculate the implausibilty for each sample against each observation - note this can be very large so we only sample a fraction!
implaus = sampler.get_implausibility(sample_points[::100], batch_size=1) #batch_size = 1000


# In[88]:


# The implausibility distributions for different observations can be very different.
#_ = plt.hist(implaus.data[1400, :, :])
#_ = plt.hist(implaus.data[14])
#plt.gca().set(xlabel='Implausibility')

implaus_data = np.array(implaus.data)
_ = plt.hist(implaus_data[140, 0, :, 0])
_ = plt.hist(implaus_data[14, 0, :, 0])
plt.gca().set(xlabel='Implausibility')


# In[ ]:


# Find the valid samples in our full 1million samples by comparing against a given tolerance and threshold
valid_samples = sampler.batch_constrain(sample_points, tolerance=.1) #batch_size=10000,
print("Remaining points: {}".format(valid_samples.sum()))


# In[214]:


constrained_sample.shape


# In[211]:


# Plot the reduced parameter distribution
constrained_sample = sample_points[valid_samples]
plot_parameter_space(constrained_sample, nbins=99)


# In[218]:


Zs


# In[220]:


# Mimic Seaborn scaling without requiring the whole package
scale = 1.5
plt.rcParams['font.size'] = 12 * scale
plt.rcParams['axes.labelsize'] = 12 * scale
plt.rcParams['axes.titlesize'] = 12 * scale
plt.rcParams['xtick.labelsize'] = 11 * scale
plt.rcParams['ytick.labelsize'] = 11 * scale
plt.rcParams['lines.linewidth'] = 1.5 * scale
plt.rcParams['lines.markersize'] = 6 * scale
#

m, _ = model_gp.predict(constrained_sample[::100].values)
Zs = m.data
# Plot the emulated AAOD value (averaged over observation locations) for each point

scatter = pd.plotting.scatter_matrix(
    constrained_sample[::100],
    figsize=(12, 10),
    marker='o',
    hist_kwds={'bins': 20},
    alpha=0.8,
    range_padding=0.,
)

# Color points by Zs.mean(axis=1)
color = np.array(Zs).mean(axis=1)[::100]
for i, j in zip(*np.triu_indices_from(scatter, k=1)):
    scatter[i, j].collections[0].set_array(color)
    scatter[i, j].collections[0].set_cmap('viridis')
    # Optionally, set vmin/vmax here if needed

'''    
grr = pd.plotting.scatter_matrix(constrained_sample[::100], c=np.array(Zs).mean(axis=1), figsize=(12, 10), marker='o',
                                 hist_kwds={'bins': 20,}, s=20, alpha=.8, vmin=1e-3, vmax=1e-2, range_padding=0.,
                                 density_kwds={'range': [[0., 1.], [0., 1.]], 'colormap':'viridis'},
                                 )

# Matplotlib dragons...
grr[0][0].set_yticklabels([0.2, 0.4, 0.6, 0.8], fontsize=12 * scale)
for i in range(2):
    grr[i+1][0].set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8], fontsize=12 * scale)
for i in range(3):
    grr[2][i].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    grr[2][i].set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8], fontsize=12 * scale)
'''

plt.colorbar(grr[0][1].collections[0], ax=grr, use_gridspec=True, label='AAOD (1)')
#plt.savefig('BCPPE_constrained_params_paper.png', transparent=True)


# In[222]:


from esem.sampler import MCMCSampler


# In[223]:


sampler_mcmc = MCMCSampler(model_gp, obs_norm)


# In[225]:


samples_mcmc = sampler_mcmc.sample() #n_samples=100, mcmc_kwargs=dict(num_burnin_steps=100)


# In[72]:


import timeit
from timeit import default_timer as timer


# In[171]:


start = timer()
for i in range(100):
    params_to_cost_spline(optimal_params_final[0:-1])
end = timer()
print(end - start)

start = timer()
for i in range(100):
    params_to_cost(optimal_params_final[0:-1])
end = timer()
print(end - start)

start = timer()
for i in range(100):
    params_to_cost_actually_good(optimal_params_final[0:-1])
end = timer()
print(end - start)

start = timer()
for i in range(100):
    params_to_cost_untransform(optimal_params_final[0:-1])
end = timer()
print(end - start)


# In[ ]:


#try max workers /3 allow 3 cpus per worker


# In[115]:


#export OPENBLAS_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"

#so might need more than BLAS threads to 1 


# In[116]:


def parallel_optimization(xstart):
    result = minimize(
            params_to_cost,
            xstart,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)],
            options={"maxiter": 1000, "disp": True} # "ftol": 1e-10, 70000
        )
    return result


# 

# In[225]:


from scipy.optimize import basinhopping
seed = 23
N_xstarts = 1
rn = np.random.RandomState(seed)
xstarts = rn.rand(N_xstarts, 16)
#xstarts = guess_norm
#xstart = xstart.reshape(1, -1)#
print(xstarts)


results = np.full((N_xstarts, 17), np.nan)
for x0 in range(xstarts.shape[0]):
    xstart = xstarts[x0]

    start = timer()
    local_minimizer = {
        "method": "L-BFGS-B",
        "bounds": [(0.0, 1.0)],
        }
    ret = basinhopping(params_to_cost, xstart, minimizer_kwargs=local_minimizer, niter=100)
    end = timer()
    print('time:', end - start)

    #optimized_input = result.x
    #function_value = result.fun
    results[x0] = np.hstack((result.x, result.fun))
    print(result)

optimal_row = np.argmin(abs(results[:, -1]))
optimal_params = results[optimal_row][0:-1]
optimal_cost = results[optimal_row, -1]
print(optimal_params, optimal_cost)


# In[136]:


#not parallelized
seed = 23
N_xstarts = 1
rn = np.random.RandomState(seed)
xstarts = rn.rand(N_xstarts, 16)
#xstarts = guess_norm
#xstart = xstart.reshape(1, -1)#
print(xstarts)


results = np.full((N_xstarts, 17), np.nan)
for x0 in range(xstarts.shape[0]):
    xstart = xstarts[x0]

    start = timer()
    result = minimize(
        params_to_cost,
        xstart,
        method= "Powell",
        bounds=[(0.0, 1.0)],
        options={"maxiter": 1000, "disp": True} #70000
        #options={"ftol": 1e-10, "maxiter": 1000, "disp": False} #70000
    )
    end = timer()
    print('time:', end - start)

    #optimized_input = result.x
    #function_value = result.fun
    results[x0] = np.hstack((result.x, result.fun))
    print(result)

optimal_row = np.argmin(abs(results[:, -1]))
optimal_params = results[optimal_row][0:-1]
optimal_cost = results[optimal_row, -1]
print(optimal_params, optimal_cost)


# In[222]:


#not parallelized
seed = 23
N_xstarts = 1
rn = np.random.RandomState(seed)
xstarts = rn.rand(N_xstarts, 16)
#xstarts = guess_norm
#xstart = xstart.reshape(1, -1)#
print(xstarts)


results = np.full((N_xstarts, 17), np.nan)
for x0 in range(xstarts.shape[0]):
    xstart = xstarts[x0]

    start = timer()
    result = minimize(
        params_to_cost,
        xstart,
        method= "SLSQP", #Powell",
        bounds=[(0.0, 1.0)],
        options={"maxiter": 1000, "disp": True} #70000
        #options={"ftol": 1e-10, "maxiter": 1000, "disp": False} #70000
    )
    end = timer()
    print('time': end - start)

    #optimized_input = result.x
    #function_value = result.fun
    results[x0] = np.hstack((result.x, result.fun))
    print(result)

optimal_row = np.argmin(abs(results[:, -1]))
optimal_params = results[optimal_row][0:-1]
optimal_cost = results[optimal_row, -1]
print(optimal_params, optimal_cost)


# In[172]:


#not parallelized
seed = 23
N_xstarts = 10
rn = np.random.RandomState(seed)
xstarts = rn.rand(N_xstarts, 16)
#xstarts = guess_norm
#xstart = xstart.reshape(1, -1)#
print(xstarts)


results = np.full((N_xstarts, 17), np.nan)
for x0 in range(xstarts.shape[0]):
    xstart = xstarts[x0]

    start = timer()
    result = minimize(
        params_to_cost_spline, ### trying spline here!
        xstart,
        method="L-BFGS-B",
        bounds=[(0.0, 1.0)],
        options={"maxiter": 1000, "disp": True} #70000
        #options={"ftol": 1e-10, "maxiter": 1000, "disp": False} #70000
    )
    end = timer()
    print(end - start)

    #optimized_input = result.x
    #function_value = result.fun
    results[x0] = np.hstack((result.x, result.fun))
    print(result)

optimal_row = np.argmin(abs(results[:, -1]))
optimal_params = results[optimal_row][0:-1]
optimal_cost = results[optimal_row, -1]
print(optimal_params, optimal_cost)


# In[216]:


import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
#probably want to use processpool 


seed = 34
N_xstarts = 10
rn = np.random.RandomState(seed)
xstarts = rn.rand(N_xstarts, 16)

with ThreadPoolExecutor(max_workers=200) as executor: #set the max_workers to the cpus of the node requested - os.environ[cpu count], mp.cpus count
    results = list(executor.map(parallel_optimization, xstarts))
    print(results)
    
#print(results)
# Print the results
for i, res in enumerate(results):
    print(f"Optimization {i+1}:")
    print(f"  x: {res.x}")
    print(f"  fun: {res.fun}")


# In[ ]:


import multiprocessing as mp
seed = 12
N_xstarts = 2
rn = np.random.RandomState(seed)
xstarts = rn.rand(N_xstarts, 16)
print(xstarts)
# optimize
with mp.get_context("fork").Pool() as pool:
    results_par = pool.map(parallel_optimization, xstarts)
print(results_par)

#This sets the seed and number of initial guess for optimizer
seed = 42
N_xstarts = 3
rn = np.random.RandomState(seed)
initial_xstarts = rn.rand(N_xstarts, X.shape[1])
#initial_xstarts = [
#    np.array([0.1, 0.2, 0.3, 0.4]),
#    np.array([0.1, 0.2, 0.3, 0.4]),
#    np.array([0.1, 0.2, 0.3, 0.4])
#]

results_vect = []
for i, xstart in enumerate(initial_xstarts):
    result = minimize(
        params_to_cost,
        xstart,
        method='L-BFGS-B',
        bounds=[(0.0, 1.0)],
        options={"ftol": 1e-10, "maxiter": 100, "disp": True} #70000
    )
    results.append(results_vect)
    
    print(f'Starting position {i + 1}: Optimized Input: {result.x}, Cost: {result.fun}')

final_result = min(results, key=lambda res: res.fun)
print(f'Optimal Result: Input = {final_result.x}, Cost = {final_result.fun}')

# ##### Transform back

# In[69]:


### These are the ones the first optimal params were run on
optimal_params_final = np.array([4.23801336e-04, 4.72556745e-01, 4.61459405e-02, 5.55611233e-02,
                        7.22765089e-01, 7.89540697e-01, 2.06803650e-01, 8.54695402e-01,
                        0.00000000e+00, 1.00000000e+00, 4.17848979e-03, 0.00000000e+00,
                        6.00730977e-01, 5.79519003e-02, 1.82339368e-01, 0.00000000e+00, 1.3608689501034148])


# In[77]:


print('optimal:')
print(params_to_cost_print(optimal_params_final [0:-1])) #
print('default:')
print(params_to_cost_print(X_train_norm[0]))

print('doing default better:')
standard_default = Y_train_norm[0]
for_function_default = standard_default[None, :]
print(cost_function_print(for_function_default, obs_norm, var_weights_dict, zrg_weights_dict))


# In[209]:


final_preds_norm = model_gp.predict(optimal_params_final[0:-1].reshape(1,-1))[0]
PCP_final_preds_norm = final_preds_norm[:, :, 0] #this is a numpy array
TLWP_final_preds_norm = final_preds_norm[:, :, 1]
OSR_final_preds_norm = final_preds_norm[:, :, 2]
OLR_final_preds_norm = final_preds_norm[:, :, 3]

PCP_final_preds = Y_pipe_sk_ss_PCP.inverse_transform(PCP_final_preds_norm)[0]
TLWP_final_preds = Y_pipe_sk_ss_TLWP.inverse_transform(TLWP_final_preds_norm)[0]
OSR_final_preds = Y_pipe_sk_ss_OSR.inverse_transform(OSR_final_preds_norm)[0]
OLR_final_preds = Y_pipe_sk_ss_OLR.inverse_transform(OLR_final_preds_norm)[0]

#PCP_final_preds = (PCP_final_preds) #**(8)
#TLWP_final_preds = (TLWP_final_preds) #**(4)
#OSR_final_preds = (OSR_final_preds) #**(4)
#OLR_final_preds = (OLR_final_preds) #**(8)

#PCP_obs = (PCP_obs) #**(8)
#TLWP_obs = (TLWP_obs) #**(4)
#OSR_obs = (OSR_obs)**(4)
#OLR_obs = (OLR_obs)**(8)

final_params = X_pipe_sk_minmax.inverse_transform(optimal_params_final[0:-1].reshape(1,-1))


# In[210]:


final_params

This is what was used in the first validation run:
thl2tune: 0.104196   
qw2tune:  4.778312  
length_fac: 0.619575  
c_diag_3rd_mom: 0.650055       
Ckh: 0.750489       
Ckm: 0.810587 
lambda_low: 0.020144
lambda_high: 0.086117  
p3_spa_to_nc: 0.1  
p3_eci: 1.0    
p3_eri: 0.103761 
p3_k_accretion:  0.01                                      
p3_dep_nucleation_exponent: 0.262657  
max_total_ni: 1.050543e+06 
p3_ice_sed_knob: 1.182339 
p3_d_breakup_cutoff: 0.0 
# ##### Some plotting checks

# In[125]:


plt.hist(PCP_final_preds[0:17] - PCP_obs.iloc[0,0:17])
plt.title('PCP error')
plt.show()
plt.hist((TLWP_final_preds[0:17]-TLWP_obs.iloc[0,0:17]))
plt.title('TLWP error')
plt.show()
plt.hist((OSR_final_preds[0:17]-OSR_obs.iloc[0,0:17]))
plt.title('OSR error')
plt.show()
plt.hist((OLR_final_preds[0:17]-OLR_obs.iloc[0,0:17]))
plt.title('OLR error')
plt.show()


# ##### Some numberical checks

# In[43]:


np.min(PCP_train_norm), np.max(PCP_train_norm), np.min(PCP_obs_norm), np.max(PCP_obs_norm)


# In[44]:


np.min(TLWP_train_norm),np.max(TLWP_train_norm),np.min(TLWP_obs_norm), np.max(TLWP_obs_norm)


# In[45]:


np.min(OSR_train_norm),np.max(OSR_train_norm),np.min(OSR_obs_norm), np.max(OSR_obs_norm)


# In[46]:


np.min(OLR_train_norm),np.max(OLR_train_norm),np.min(OLR_obs_norm), np.max(OLR_obs_norm)


# In[47]:


PCP_zrg_obs, PCP_train.iloc[0], PCP_final_preds
PCP_train_norm[0]


# In[67]:


trainvar1 = PCP_train.iloc[0]
obsvar1 = PCP_zrg_obs
predsvar1 = PCP_final_preds
#trainvar1 = TLWP_train.iloc[0]
#obsvar1 = TLWP_zrg_obs
#predsvar1 = TLWP_final_preds
#trainvar1 = OSR_train.iloc[0]
#obsvar1 = OSR_zrg_obs
#predsvar1 = OSR_final_preds
#trainvar1 = OLR_train.iloc[0]
#obsvar1 = OLR_zrg_obs
#predsvar1 = OLR_final_preds

print(np.max(abs(predsvar1 - obsvar1)), np.min(abs(predsvar1 - obsvar1)), np.mean(abs(predsvar1 - obsvar1)))
abs(predsvar1 - obsvar1)


# In[68]:


print(np.max(abs(trainvar1 - obsvar1)), np.min(abs(trainvar1 - obsvar1)), np.mean(abs(trainvar1 - obsvar1)))
abs(trainvar1 - obsvar1)


# In[81]:


row1 = abs(PCP_final_preds - PCP_zrg_obs)
row2 = abs(PCP_train.iloc[0] - PCP_zrg_obs)

df = pd.DataFrame([row1.iloc[0], row2.iloc[0]], index=['Opt', 'Default'])
higher_values = df.idxmin()
print("Lower error value per zrg column:")
print(higher_values)


# In[82]:


row1 = abs(TLWP_final_preds - TLWP_zrg_obs)
row2 = abs(TLWP_train.iloc[0] - TLWP_zrg_obs)

df = pd.DataFrame([row1.iloc[0], row2.iloc[0]], index=['Opt', 'Default'])
higher_values = df.idxmin()
print("Lower error value per zrg column:")
print(higher_values)


# In[83]:


row1 = abs(OSR_final_preds - OSR_zrg_obs)
row2 = abs(OSR_train.iloc[0] - OSR_zrg_obs)

df = pd.DataFrame([row1.iloc[0], row2.iloc[0]], index=['Opt', 'Default'])
higher_values = df.idxmin()
print("Lower error value per zrg column:")
print(higher_values)


# In[84]:


row1 = abs(OLR_final_preds - OLR_zrg_obs)
row2 = abs(OLR_train.iloc[0] - OLR_zrg_obs)

df = pd.DataFrame([row1.iloc[0], row2.iloc[0]], index=['Opt', 'Default'])
higher_values = df.idxmin()
print("Lower error value per zrg column:")
print(higher_values)

## Are there any data transforms to undo??
'''
#ppe_dataset_small = ppe_dataset.drop_vars(to_leave)
#ppe_dataset_small['TotalLiqWaterPath'] = (ppe_dataset_small.LiqWaterPath + ppe_dataset_small.RainWaterPath)
#ppe_dataset_small['precip_total_surf_mass_flux'] = ppe_dataset_small['precip_total_surf_mass_flux']*1e-3*24*3600
#ppe_dataset_small = ppe_dataset_small.drop_dims('lev')
#ppe_dataset_small.attrs['name'] = "SmallScreamPPEData"

ppe_dataset_small['precip_total_surf_mass_flux'] = ppe_dataset_small['precip_total_surf_mass_flux']**(1/4)
#ppe_dataset_small['LiqWaterPath'] = ppe_dataset_small['LiqWaterPath']**(1/4)
#ppe_dataset_small['RainWaterPath'] = ppe_dataset_small['RainWaterPath']**(1/4)
ppe_dataset_small['TotalLiqWaterPath'] = ppe_dataset_small['TotalLiqWaterPath']**(1/4)
#ppe_dataset_small['LW_flux_up_at_model_top'] = ppe_dataset_small['LW_flux_up_at_model_top']**(1/4)
#ppe_dataset_small['SW_flux_up_at_model_top'] = ppe_dataset_small['SW_flux_up_at_model_top']**(1/4)
'''

PCP_proj_norm_gp_untrans = PCP_proj_norm_gp**(4)
TLWP_proj_norm_gp_untrans = TLWP_proj_norm_gp**(4)PCP_proj_gp = pd.DataFrame(Y_pipe_sk_ss_PCP.inverse_transform(PCP_proj_norm_gp_untrans))
TLWP_proj_gp = pd.DataFrame(Y_pipe_sk_ss_TLWP.inverse_transform(TLWP_proj_norm_gp_untrans))
OSR_proj_gp = pd.DataFrame(Y_pipe_sk_ss_OSR.inverse_transform(OSR_proj_norm_gp))
OLR_proj_gp = pd.DataFrame(Y_pipe_sk_ss_OLR.inverse_transform(OLR_proj_norm_gp))thl2tune: 0.104196   
qw2tune:  4.778312  
length_fac: 0.619575  
c_diag_3rd_mom: 0.650055       
Ckh: 0.750489       
Ckm: 0.810587 
lambda_low: 0.020144
lambda_high: 0.086117  
p3_spa_to_nc: 0.1  
p3_eci: 1.0    
p3_eri: 0.103761 
p3_k_accretion:  0.01                                      
p3_dep_nucleation_exponent: 0.262657  
max_total_ni: 1.050543e+06 
p3_ice_sed_knob: 1.182339 
p3_d_breakup_cutoff: 0.0 
# In[212]:


z_num = len(lat_bands)-1
r_num = len(regions_list)
all_num = len(lat_bands) + len(regions_list)


# In[216]:


## Old optimal runs -- wrong date and hash
#optimal_run_data = xr.open_dataset('/global/cfs/cdirs/e3smdata/simulations/ecp-autotune/sims-dec3-2024/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2019-08-02-00000.nc')
#optimal_run_data = xr.open_dataset('/global/cfs/cdirs/e3smdata/simulations/ecp-autotune/sims-dec3-2024/opt-aug2019.ne1024pg2_ne1024pg2.F2010-SCREAMv1.c00-dec2.n2048.sk.optdec3-2024/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2019-08-02-00000.nc')

optimal_run_data_2019 = xr.open_dataset('/global/cfs/cdirs/e3smdata/simulations/ecp-autotune/sims-dec3-2024/opt-aug2019.ne1024pg2_ne1024pg2.F2010-SCREAMv1.c00-dec2.n2048.sk.optdec3-2024/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2019-08-06-00000.nc')
optimal_run_data_2019


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
    DY2_PCP_default_error[runn] = root_mean_squared_error(new_default_data_small.precip_total_surf_mass_flux[0], DY2_ppe_dataset_small.sel(run_label=run_lab).DY2_precip_total_surf_mass_flux)
    DY2_TLWP_default_error[runn] = root_mean_squared_error(new_default_data_small.TotalLiqWaterPath[0], DY2_ppe_dataset_small.sel(run_label=run_lab).DY2_TotalLiqWaterPath)
    DY2_OSR_default_error[runn] = root_mean_squared_error(new_default_data_small.SW_flux_up_at_model_top[0], DY2_ppe_dataset_small.sel(run_label=run_lab).DY2_SW_flux_up_at_model_top)
    DY2_OLR_default_error[runn] = root_mean_squared_error(new_default_data_small.LW_flux_up_at_model_top[0], DY2_ppe_dataset_small.sel(run_label=run_lab).DY2_LW_flux_up_at_model_top)
    
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
    DY2_PCP_default_zrg_error[runn] = root_mean_squared_error(new_default_data_small.precip_total_surf_mass_flux[0], DY2_ppe_dataset_small.sel(run_label=run_lab).DY2_precip_total_surf_mass_flux)
    DY2_TLWP_default_zrg_error[runn] = root_mean_squared_error(new_default_data_small.TotalLiqWaterPath[0], DY2_ppe_dataset_small.sel(run_label=run_lab).DY2_TotalLiqWaterPath)
    DY2_OSR_default_zrg_error[runn] = root_mean_squared_error(new_default_data_small.SW_flux_up_at_model_top[0], DY2_ppe_dataset_small.sel(run_label=run_lab).DY2_SW_flux_up_at_model_top)
    DY2_OLR_default_zrg_error[runn] = root_mean_squared_error(new_default_data_small.LW_flux_up_at_model_top[0], DY2_ppe_dataset_small.sel(run_label=run_lab).DY2_LW_flux_up_at_model_top)
    
DY2_final_default_error = DY2_PCP_default_error+DY2_TLWP_default_error+DY2_OSR_default_error+DY2_OLR_default_error

print('Argmin PCP:', sim_names[np.argmin(DY2_PCP_default_error)])
print('Argmin TLWP:', sim_names[np.argmin(DY2_TLWP_default_error)])
print('Argmin OSR:', sim_names[np.argmin(DY2_OSR_default_error)])
print('Argmin OLR:', sim_names[np.argmin(DY2_OLR_default_error)])

print('Final Argmin:', sim_names[np.argmin(DY2_final_default_error)])


# In[213]:


#This is just DY2!
optimal_run_data = xr.open_dataset('/global/cfs/cdirs/e3smdata/simulations/ecp-autotune/sims-dec3-2024/hh1024/optdec3/SCREAM.2024-autocal-00.ne1024pg2/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc')


# In[217]:


#This is just DY2
optimal_run_data
optimal_run_data_small = optimal_run_data.drop_vars(to_leave)
optimal_run_data_small['TotalLiqWaterPath'] = (optimal_run_data_small.LiqWaterPath + optimal_run_data_small.RainWaterPath)
optimal_run_data_small.squeeze('time')


# In[269]:


#These are just for DY2
PCP_opt = optimal_run_data_small.precip_total_surf_mass_flux
TLWP_opt = optimal_run_data_small.TotalLiqWaterPath
OSR_opt = optimal_run_data_small.SW_flux_up_at_model_top
OLR_opt = optimal_run_data_small.LW_flux_up_at_model_top

help im actually so scaredcost_function_print(preds, obs, var_weights_dict, zrg_weights_dict) #area_weights,
PCP_proj_c = preds[:, :, 0] #this is a numpy array
TLWP_proj_c = preds[:, :, 1]
OSR_proj_c = preds[:, :, 2]
OLR_proj_c = preds[:, :, 3]
    
PCP_obs_c = obs[:, :, 0] #this is a numpy array
TLWP_obs_c = obs[:, :, 1]
OSR_obs_c = obs[:, :, 2]
OLR_obs_c = obs[:, :, 3]
# In[270]:


PCP_gridcell = np.array([5.06340494e-10, 4.92116052e-09, 1.58665070e-08, 3.07661418e-08,
       3.55065630e-08, 2.34604879e-08, 3.72106050e-08, 4.24539022e-08,
       4.13584847e-08, 4.32131084e-08, 4.70194917e-09, 1.84775685e-08,
       3.68316809e-08, 3.16717990e-08, 2.43327872e-08, 2.05848905e-08,
       7.63095746e-09, 1.57959125e-09, 1.22796246e-08, 1.26624345e-08,
       3.76302761e-08, 3.69793659e-08, 5.39746614e-08, 1.76070167e-08, 2.858662280619501e-08])
TLWP_gridcell = np.array([0.00063201, 0.01194122, 0.07097525, 0.15663945, 0.15114135,
       0.07089909, 0.08465579, 0.08718092, 0.08760621, 0.09901258,
       0.03696158, 0.05959901, 0.10949862, 0.07296287, 0.0492253 ,
       0.02420988, 0.00819805, 0.00193441, 0.02978515, 0.03968861,
       0.12506567, 0.07257491, 0.11317925, 0.05988763, 0.07905592808622268])
OSR_gridcell = np.array([3.04710822e+02, 2.64960246e+02, 1.85524556e+02, 1.80526288e+02,
       1.54298748e+02, 1.06706866e+02, 1.00276535e+02, 1.00531361e+02,
       9.65788495e+01, 9.26343558e+01, 7.01231257e+01, 7.42437207e+01,
       8.44921102e+01, 7.23116266e+01, 4.56346818e+01, 1.54534630e+01,
       7.90863578e-02, 0.00000000e+00, 1.17083500e+02, 7.33771154e+01,
       1.19228779e+02, 1.10952979e+02, 9.13173017e+01, 7.69703276e+01, 99.19717819454314])
OLR_gridcell = np.array([195.2142339 , 202.38090686, 216.87761257, 219.31806314,
       232.08660453, 260.62521929, 267.65748369, 249.03337894,
       244.33218156, 254.33839864, 281.53439222, 263.2081499 ,
       221.07182832, 206.12662148, 193.96434633, 173.21265757,
       165.78700508, 164.1785421 , 189.63753347, 212.96784759,
       228.65154941, 252.37460508, 248.23090425, 269.17135429, 237.33540478722333])


# In[228]:


#standard_default #= Y_train_norm[0].shape
#for_function_default


# In[271]:


#These are for DY2
PCP_default_values = np.array([6.0759098e-10, 5.0665494e-09, 1.5982058e-08, 3.4422378e-08,
       3.9564799e-08, 2.7233771e-08, 4.8393659e-08, 5.0862113e-08,
       5.6026789e-08, 6.0322904e-08, 1.2639425e-08, 2.1868269e-08,
       4.2785441e-08, 3.6009368e-08, 2.7117203e-08, 2.2583755e-08,
       8.2424600e-09, 1.9501609e-09, 1.3016564e-08, 1.4906114e-08,
       4.2587406e-08, 4.2515854e-08, 7.6965009e-08, 2.4714762e-08, 3.5520223e-08])
TLWP_default_values = np.array([0.00190594, 0.01279696, 0.06972011, 0.12074795, 0.09879082,
       0.04594548, 0.05182424, 0.04672168, 0.05155803, 0.05652613,
       0.01944731, 0.03291932, 0.07174974, 0.06238572, 0.05042445,
       0.02264049, 0.00462449, 0.00140322, 0.02859227, 0.03983202,
       0.08661857, 0.04111739, 0.06555941, 0.03361464, 0.052390814])
OSR_default_values = np.array([3.05692841e+02, 2.67774048e+02, 1.98876389e+02, 1.95783203e+02,
       1.66375381e+02, 1.14257965e+02, 1.04335182e+02, 1.05946388e+02,
       1.05339088e+02, 1.01501572e+02, 6.90990601e+01, 7.42979279e+01,
       8.55386200e+01, 7.49805069e+01, 4.78489914e+01, 1.57214584e+01,
       7.95350969e-02, 0.00000000e+00, 1.21368813e+02, 7.56370544e+01,
       1.27448418e+02, 1.11861763e+02, 1.02085823e+02, 8.01813354e+01, 104.40278])
OLR_default_values = np.array([195.11432, 201.836  , 215.52368, 217.0451 , 229.63655, 259.87613,
       266.5561 , 243.47202, 234.95396, 245.0337 , 282.4472 , 263.93073,
       221.49437, 205.37933, 192.45297, 171.57426, 165.7792 , 163.45021,
       188.67244, 212.4854 , 227.28685, 247.64632, 238.50238, 268.13565, 234.74042])


# ##### New default

# In[446]:


plt.figure(figsize=(8, 4))
point_size = 30

dict1 = zonal_means_native(new_default_data_small.precip_total_surf_mass_flux, area, lat, lon)
dict2 = regional_means_native(new_default_data_small.precip_total_surf_mass_flux, area)
array = np.array([global_means_native(new_default_data_small.precip_total_surf_mass_flux, area)])

# Extract keys and values from dictionaries
keys1, values1 = list(dict1.keys()), list(dict1.values())
keys2, values2 = list(dict2.keys()), list(dict2.values())

# Create x positions
x1 = range(len(keys1))  # x positions for dict1
x2 = range(len(keys1), len(keys1) + len(keys2))  # x positions for dict2
x3 = range(len(keys1) + len(keys2), len(keys1) + len(keys2) + len(array))  # x positions for array
array_labels = ['global'] # label for global array

#Scatter each row of zrg ppe dataset
for row_name, row_values in DY2_PCP_zrg_ppedataset.iterrows():
    plt.scatter(range(all_num), (row_values-DY2_PCP_zrg_ppedataset.iloc[0,:]), color='gray', alpha=0.25, s=point_size-10)
    
plt.axhline(y=0, color="black", linewidth=1, zorder=1, alpha=0.7) #linestyle=":",

plt.plot(x1, (values1-DY2_PCP_zrg_ppedataset.iloc[0,0:z_num]), '-o', label="New default simulation", color="blue")
plt.plot(x2, (values2-DY2_PCP_zrg_ppedataset.iloc[0,(z_num):(z_num+r_num)]), '-o', color="blue")
plt.plot(x3, (array[0]-DY2_PCP_zrg_ppedataset.iloc[0,-1]), '-o', color="blue")

# Create combined x-tick labels and positions
all_keys = keys1 + keys2 + array_labels
all_positions = list(x1) + list(x2) + list(x3)

# Customize x-axis
plt.xticks(all_positions, all_keys, rotation=45, ha='right')

plt.xlabel("           Center of lattitude band                             Region               Global")
plt.ylabel("Target variable error")
plt.title("PCP comparison plot")
plt.legend(fontsize=8)
#plt.tight_layout()
plt.show()


# In[448]:


plt.figure(figsize=(8, 4))
point_size = 30

dict1 = zonal_means_native(new_default_data_small.TotalLiqWaterPath, area, lat, lon)
dict2 = regional_means_native(new_default_data_small.TotalLiqWaterPath, area)
array = np.array([global_means_native(new_default_data_small.TotalLiqWaterPath, area)])

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
    plt.scatter(range(all_num), (row_values-DY2_TLWP_zrg_ppedataset.iloc[0,:]), color='gray', alpha=0.25, s=point_size-10)
    
plt.axhline(y=0, color="black", linewidth=1, zorder=1, alpha=0.7) #linestyle=":",

plt.plot(x1, (values1-DY2_TLWP_zrg_ppedataset.iloc[0,0:z_num]), '-o', label="New default simulation", color="blue")
plt.plot(x2, (values2-DY2_TLWP_zrg_ppedataset.iloc[0,(z_num):(z_num+r_num)]), '-o', color="blue")
plt.plot(x3, (array[0]-DY2_TLWP_zrg_ppedataset.iloc[0,-1]), '-o', color="blue")

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


# In[450]:


plt.figure(figsize=(8, 4))
point_size = 30

dict1 = zonal_means_native(new_default_data_small.SW_flux_up_at_model_top, area, lat, lon)
dict2 = regional_means_native(new_default_data_small.SW_flux_up_at_model_top, area)
array = np.array([global_means_native(new_default_data_small.SW_flux_up_at_model_top, area)])

# Extract keys and values from dictionaries
keys1, values1 = list(dict1.keys()), list(dict1.values())
keys2, values2 = list(dict2.keys()), list(dict2.values())

# Create x positions
x1 = range(len(keys1))  # x positions for dict1
x2 = range(len(keys1), len(keys1) + len(keys2))  # x positions for dict2
x3 = range(len(keys1) + len(keys2), len(keys1) + len(keys2) + len(array))  # x positions for array
array_labels = ['global'] # label for global array

#Scatter each row of zrg ppe dataset
for row_name, row_values in DY2_OSR_zrg_ppedataset.iterrows():
    plt.scatter(range(all_num), (row_values-DY2_OSR_zrg_ppedataset.iloc[0,:]), color='gray', alpha=0.25, s=point_size-10)
    
plt.axhline(y=0, color="black", linewidth=1, zorder=1, alpha=0.7) #linestyle=":",

plt.plot(x1, (values1-DY2_OSR_zrg_ppedataset.iloc[0,0:z_num]), '-o', label="New default simulation", color="blue")
plt.plot(x2, (values2-DY2_OSR_zrg_ppedataset.iloc[0,(z_num):(z_num+r_num)]), '-o', color="blue")
plt.plot(x3, (array[0]-DY2_OSR_zrg_ppedataset.iloc[0,-1]), '-o', color="blue")

# Create combined x-tick labels and positions
all_keys = keys1 + keys2 + array_labels
all_positions = list(x1) + list(x2) + list(x3)

# Customize x-axis
plt.xticks(all_positions, all_keys, rotation=45, ha='right')

plt.xlabel("           Center of lattitude band                             Region               Global")
plt.ylabel("Target variable error")
plt.title("OSR comparison plot")
plt.legend(fontsize=8)
#plt.tight_layout()
plt.show()


# In[452]:


plt.figure(figsize=(8, 4))
point_size = 30

dict1 = zonal_means_native(new_default_data_small.LW_flux_up_at_model_top, area, lat, lon)
dict2 = regional_means_native(new_default_data_small.LW_flux_up_at_model_top, area)
array = np.array([global_means_native(new_default_data_small.LW_flux_up_at_model_top, area)])

# Extract keys and values from dictionaries
keys1, values1 = list(dict1.keys()), list(dict1.values())
keys2, values2 = list(dict2.keys()), list(dict2.values())

# Create x positions
x1 = range(len(keys1))  # x positions for dict1
x2 = range(len(keys1), len(keys1) + len(keys2))  # x positions for dict2
x3 = range(len(keys1) + len(keys2), len(keys1) + len(keys2) + len(array))  # x positions for array
array_labels = ['global'] # label for global array

#Scatter each row of zrg ppe dataset
for row_name, row_values in DY2_OLR_zrg_ppedataset.iterrows():
    plt.scatter(range(all_num), (row_values-DY2_OLR_zrg_ppedataset.iloc[0,:]), color='gray', alpha=0.25, s=point_size-10)
    
plt.axhline(y=0, color="black", linewidth=1, zorder=1, alpha=0.7) #linestyle=":",

plt.plot(x1, (values1-DY2_OLR_zrg_ppedataset.iloc[0,0:z_num]), '-o', label="New default simulation", color="blue")
plt.plot(x2, (values2-DY2_OLR_zrg_ppedataset.iloc[0,(z_num):(z_num+r_num)]), '-o', color="blue")
plt.plot(x3, (array[0]-DY2_OLR_zrg_ppedataset.iloc[0,-1]), '-o', color="blue")

# Create combined x-tick labels and positions
all_keys = keys1 + keys2 + array_labels
all_positions = list(x1) + list(x2) + list(x3)

# Customize x-axis
plt.xticks(all_positions, all_keys, rotation=45, ha='right')

plt.xlabel("           Center of lattitude band                             Region               Global")
plt.ylabel("Target variable error")
plt.title("OLR comparison plot")
plt.legend(fontsize=8)
#plt.tight_layout()
plt.show()


# ##### Comparison plots

# In[454]:


plt.figure(figsize=(8, 4))
point_size = 30

dict1 = zonal_means_native(PCP_opt, area, lat, lon)
dict2 = regional_means_native(PCP_opt, area)
array = np.array([global_means_native(PCP_opt, area)])

# Extract keys and values from dictionaries
keys1, values1 = list(dict1.keys()), list(dict1.values())
keys2, values2 = list(dict2.keys()), list(dict2.values())

# Create x positions
x1 = range(len(keys1))  # x positions for dict1
x2 = range(len(keys1), len(keys1) + len(keys2))  # x positions for dict2
x3 = range(len(keys1) + len(keys2), len(keys1) + len(keys2) + len(array))  # x positions for array
array_labels = ['global'] # label for global array

#Scatter each row of zrg ppe dataset
for row_name, row_values in DY2_PCP_zrg_ppedataset.iterrows():
    plt.scatter(range(all_num), (row_values-DY2_PCP_zrg_obs), color='gray', alpha=0.25, s=point_size-10)
    
plt.axhline(y=0, color="black", linewidth=1, zorder=1, alpha=0.7) #linestyle=":",

#plt.scatter(range(all_num), (PCP_default_values-DY2_PCP_zrg_obs.iloc[:,0:all_num]), label="Default", marker = '^', color='orange', s=point_size)
plt.plot(range(all_num), (PCP_default_values-DY2_PCP_zrg_obs.iloc[:,0:all_num]).to_numpy()[0], '-o', color='orange', label="Default")

# Plot scatter points - simulation (we want DY2, so the second half of the dataset
#plt.scatter(x1, (values1-DY2_PCP_zrg_obs.iloc[:,0:z_num]), label="Opt simulation", edgecolor="green", facecolors="green", s=point_size)
#plt.scatter(x2, (values2-DY2_PCP_zrg_obs.iloc[:,(z_num):(z_num+r_num)]), edgecolor="green", facecolors="green", s=point_size)
#plt.scatter(x3, (array-DY2_PCP_zrg_obs.iloc[:,-1]), edgecolor="green", facecolors="green", s=point_size)

plt.plot(x1, (values1-DY2_PCP_zrg_obs.iloc[:,0:z_num]).to_numpy()[0], '-^', label="Opt simulation", color="green")
plt.plot(x2, (values2-DY2_PCP_zrg_obs.iloc[:,(z_num):(z_num+r_num)]).to_numpy()[0], '-^', color="green")
plt.plot(x3, (array-DY2_PCP_zrg_obs.iloc[:,-1]).to_numpy()[0], '-^', color="green")

#plt.scatter(range(len(PCP_final_preds)), PCP_final_preds, label="Opt ZRG pred", marker = 's', edgecolors="blue", facecolors="none", s=point_size)
#plt.scatter(range(len(PCP_final_preds)), PCP_gridcell, label="Opt gridcell pred", color='purple', s=point_size)
#plt.scatter(range(all_num), PCP_default_values, label="Default", marker = 'x', color="orange", s=point_size)
#plt.scatter(range(all_num), DY2_PCP_zrg_obs, label="Obs", marker = '^', edgecolor='red', facecolors="none", s=point_size)

# Create combined x-tick labels and positions
all_keys = keys1 + keys2 + array_labels
all_positions = list(x1) + list(x2) + list(x3)

# Customize x-axis
plt.xticks(all_positions, all_keys, rotation=45, ha='right')

plt.xlabel("           Center of lattitude band                             Region               Global")
plt.ylabel("Target variable error")
plt.title("PCP comparison plot")
plt.legend(fontsize=8)
#plt.tight_layout()
plt.show()


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


# In[458]:


plt.figure(figsize=(8, 4))
point_size = 30

dict1 = zonal_means_native(OSR_opt, area, lat, lon)
dict2 = regional_means_native(OSR_opt, area)
array = np.array([global_means_native(OSR_opt, area)])

# Extract keys and values from dictionaries
keys1, values1 = list(dict1.keys()), list(dict1.values())
keys2, values2 = list(dict2.keys()), list(dict2.values())

# Create x positions
x1 = range(len(keys1))  # x positions for dict1
x2 = range(len(keys1), len(keys1) + len(keys2))  # x positions for dict2
x3 = range(len(keys1) + len(keys2), len(keys1) + len(keys2) + len(array))  # x positions for array
array_labels = ['global'] # label for global array

#Scatter each row of zrg ppe dataset
for row_name, row_values in DY2_OSR_zrg_ppedataset.iterrows():
    plt.scatter(range(all_num), (row_values-DY2_OSR_zrg_obs), color='gray', alpha=0.25, s=point_size-10)
    
plt.axhline(y=0, color="black", linewidth=1, zorder=1, alpha=0.7) #linestyle=":",

#plt.scatter(range(all_num), (OSR_default_values-DY2_OSR_zrg_obs.iloc[:,0:all_num]), label="Default", marker = '^', color='orange', s=point_size)
plt.plot(range(all_num), (OSR_default_values-DY2_OSR_zrg_obs.iloc[:,0:all_num]).to_numpy()[0], '-o', color='orange', label="Default")

# Plot scatter points - simulation (we want DY2, so the second half of the dataset
#plt.scatter(x1, (values1-DY2_OSR_zrg_obs.iloc[:,0:z_num]), label="Opt simulation", edgecolor="green", facecolors="green", s=point_size)
#plt.scatter(x2, (values2-DY2_OSR_zrg_obs.iloc[:,(z_num):(z_num+r_num)]), edgecolor="green", facecolors="green", s=point_size)
#plt.scatter(x3, (array-DY2_OSR_zrg_obs.iloc[:,-1]), edgecolor="green", facecolors="green", s=point_size)
plt.plot(x1, (values1-DY2_OSR_zrg_obs.iloc[:,0:z_num]).to_numpy()[0], '-^', label="Opt simulation", color="green")
plt.plot(x2, (values2-DY2_OSR_zrg_obs.iloc[:,(z_num):(z_num+r_num)]).to_numpy()[0], '-^', color="green")
plt.plot(x3, (array-DY2_OSR_zrg_obs.iloc[:,-1]).to_numpy()[0], '-^', color="green")

#plt.scatter(range(len(OSR_final_preds)), OSR_final_preds, label="Opt ZRG pred", marker = 's', edgecolors="blue", facecolors="none", s=point_size)
#plt.scatter(range(len(OSR_final_preds)), OSR_gridcell, label="Opt gridcell pred", color='purple', s=point_size)
#plt.scatter(range(all_num), OSR_default_values, label="Default", marker = 'x', color="orange", s=point_size)
#plt.scatter(range(all_num), DY2_OSR_zrg_obs, label="Obs", marker = '^', edgecolor='red', facecolors="none", s=point_size)

# Create combined x-tick labels and positions
all_keys = keys1 + keys2 + array_labels
all_positions = list(x1) + list(x2) + list(x3)

# Customize x-axis
plt.xticks(all_positions, all_keys, rotation=45, ha='right')

plt.xlabel("           Center of lattitude band                             Region               Global")
plt.ylabel("Target variable error")
plt.title("OSR comparison plot")
plt.legend(fontsize=8)
#plt.tight_layout()
plt.show()


# In[460]:


plt.figure(figsize=(8, 4))
point_size = 30

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

#Scatter each row of zrg ppe dataset
for row_name, row_values in DY2_OLR_zrg_ppedataset.iterrows():
    plt.scatter(range(all_num), (row_values-DY2_OLR_zrg_obs), color='gray', alpha=0.25, s=point_size-10)
    
plt.axhline(y=0, color="black", linewidth=1, zorder=1, alpha=0.7) #linestyle=":",

#plt.scatter(range(all_num), (OLR_default_values-DY2_OLR_zrg_obs.iloc[:,0:all_num]), label="Default", marker = '^', color='orange', s=point_size)
plt.plot(range(all_num), (OLR_default_values-DY2_OLR_zrg_obs.iloc[:,0:all_num]).to_numpy()[0], '-o', color='orange', label="Default")

# Plot scatter points - simulation (we want DY2, so the second half of the dataset
#plt.scatter(x1, (values1-DY2_OLR_zrg_obs.iloc[:,0:z_num]), label="Opt simulation", edgecolor="green", facecolors="green", s=point_size)
#plt.scatter(x2, (values2-DY2_OLR_zrg_obs.iloc[:,(z_num):(z_num+r_num)]), edgecolor="green", facecolors="green", s=point_size)
#plt.scatter(x3, (array-DY2_OLR_zrg_obs.iloc[:,-1]), edgecolor="green", facecolors="green", s=point_size)
plt.plot(x1, (values1-DY2_OLR_zrg_obs.iloc[:,0:z_num]).to_numpy()[0], '-^', label="Opt simulation", color="green")
plt.plot(x2, (values2-DY2_OLR_zrg_obs.iloc[:,(z_num):(z_num+r_num)]).to_numpy()[0], '-^', color="green")
plt.plot(x3, (array-DY2_OLR_zrg_obs.iloc[:,-1]).to_numpy()[0], '-^', color="green")

#plt.scatter(range(len(OLR_final_preds)), OLR_final_preds, label="Opt ZRG pred", marker = 's', edgecolors="blue", facecolors="none", s=point_size)
#plt.scatter(range(len(OLR_final_preds)), OLR_gridcell, label="Opt gridcell pred", color='purple', s=point_size)
#plt.scatter(range(all_num), OLR_default_values, label="Default", marker = 'x', color="orange", s=point_size)
#plt.scatter(range(all_num), DY2_OLR_zrg_obs, label="Obs", marker = '^', edgecolor='red', facecolors="none", s=point_size)

# Create combined x-tick labels and positions
all_keys = keys1 + keys2 + array_labels
all_positions = list(x1) + list(x2) + list(x3)

# Customize x-axis
plt.xticks(all_positions, all_keys, rotation=45, ha='right')

plt.xlabel("           Center of lattitude band                             Region               Global")
plt.ylabel("Target variable error")
plt.title("OLR comparison plot")
plt.legend(fontsize=8)
#plt.tight_layout()
plt.show()


# In[345]:


plt.figure(figsize=(8, 4))
point_size = 20

dict1 = zonal_means_native(PCP_opt, area, lat, lon)
dict2 = regional_means_native(PCP_opt, area)
array = np.array([global_means_native(PCP_opt, area)])

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

#plt.scatter(range(len(PCP_final_preds)), PCP_final_preds, label="Opt ZRG pred", marker = 's', edgecolors="blue", facecolors="none", s=point_size)
#plt.scatter(range(len(PCP_final_preds)), PCP_gridcell, label="Opt gridcell pred", color='purple', s=point_size)
plt.scatter(range(all_num), PCP_default_values, label="Default", marker = 'x', color="orange", s=point_size)
plt.scatter(range(all_num), DY2_PCP_zrg_obs, label="Obs", marker = '^', edgecolor='red', facecolors="none", s=point_size)

# Create combined x-tick labels and positions
all_keys = keys1 + keys2 + array_labels
all_positions = list(x1) + list(x2) + list(x3)

# Customize x-axis
plt.xticks(all_positions, all_keys, rotation=45, ha='right')

plt.xlabel("            Center of lattitude band                             Region               Global")
plt.ylabel("Target variable value")
plt.title("PCP comparison plot")
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


# In[192]:


plt.figure(figsize=(8, 4))
point_size = 20

dict1 = zonal_means_native(OSR_opt, area, lat, lon)
dict2 = regional_means_native(OSR_opt, area)
array = np.array([global_means_native(OSR_opt, area)])

# Extract keys and values from dictionaries
keys1, values1 = list(dict1.keys()), list(dict1.values())
keys2, values2 = list(dict2.keys()), list(dict2.values())

# Create x positions
x1 = range(len(keys1))  # x positions for dict1
x2 = range(len(keys1), len(keys1) + len(keys2))  # x positions for dict2
x3 = range(len(keys1) + len(keys2), len(keys1) + len(keys2) + len(array))  # x positions for array
array_labels = ['global'] # label for global array

# Create x positions
x1 = range(len(keys1))  # x positions for dict1
x2 = range(len(keys1), len(keys1) + len(keys2))  # x positions for dict2
x3 = range(len(keys1) + len(keys2), len(keys1) + len(keys2) + len(array))  # x positions for array
array_labels = ['global'] # label for global array

# Plot scatter points - simulation 
plt.scatter(x1, values1, label="Opt simulation", edgecolor="green", facecolors="none", s=point_size)
plt.scatter(x2, values2, edgecolor="green", facecolors="none", s=point_size)
plt.scatter(x3, array, edgecolor="green", facecolors="none", s=point_size)

plt.scatter(range(len(OSR_final_preds)), OSR_final_preds, label="Opt ZRG pred", marker = 's', edgecolors="blue", facecolors="none", s=point_size)
#plt.scatter(range(len(OSR_final_preds)), OSR_gridcell, label="Opt gridcell pred", color='purple', s=point_size)
#plt.scatter(range(len(OSR_final_preds)), OSR_default_values, label="Default", marker = 'x', color="orange", s=point_size)

plt.scatter(range(len(OSR_final_preds)), zrg_obs.iloc[:, 50:75], label="Obs", marker = '^', edgecolor='red', facecolors="none", s=point_size)

# Create combined x-tick labels and positions
all_keys = keys1 + keys2 + array_labels
all_positions = list(x1) + list(x2) + list(x3)

# Customize x-axis
plt.xticks(all_positions, all_keys, rotation=45, ha='right')

plt.xlabel("            Center of lattitude band                             Region               Global")
plt.ylabel("Target variable value")
plt.title("OSR comparison plot")
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


# In[ ]:


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
plt.scatter(x1, values1, label="Opt simulation", color="red", s=point_size)
plt.scatter(x2, values2, color="red", s=point_size)
plt.scatter(x3, array, color="red", s=point_size)

plt.scatter(range(len(PCP_final_preds)), TLWP_final_preds, label="Opt predicted", color='blue', s=point_size)
plt.scatter(range(len(PCP_final_preds)), zrg_obs.iloc[:, 25:50], label="Obs", color='green', s=point_size)

# Create combined x-tick labels and positions
all_keys = keys1 + keys2 + array_labels
all_positions = list(x1) + list(x2) + list(x3)

# Customize x-axis
plt.xticks(all_positions, all_keys, rotation=45, ha='right')

plt.xlabel("            Center of lattitude band                             Region               Global")
plt.ylabel("Target variable value")
plt.title("TLWP comparison plot")
plt.legend()
#plt.tight_layout()
plt.show()


# In[46]:


data = zonal_means_native(PCP_opt, area, lat, lon)
keys = list(data.keys())
values = list(data.values())

plt.scatter(keys, values)
plt.scatter(keys, PCP_final_preds[0:18])
plt.scatter(keys, zrg_obs.iloc[:, 0:18])
plt.xlabel('Center Latitiude of Band')
plt.title('PCP')
plt.legend({'predicted', 'simulated', 'obs'})
plt.show()


# In[94]:


data = zonal_means_native(TLWP_opt, area, lat, lon)
keys = list(data.keys())
values = list(data.values())

plt.scatter(keys, values)
plt.scatter(keys, TLWP_final_preds[0:18])
plt.scatter(keys, zrg_obs.iloc[:, 25:43])
plt.xlabel('Center Latitiude of Band')
plt.title('TLWP')
plt.legend({'predicted', 'simulated', 'obs'})
plt.show()


# In[101]:


data = zonal_means_native(OSR_opt, area, lat, lon)
keys = list(data.keys())
values = list(data.values())

plt.scatter(keys, values)
plt.scatter(keys, OSR_final_preds[0:18])
plt.scatter(keys, zrg_obs.iloc[:, 50:68])
plt.xlabel('Center Latitiude of Band')
plt.title('OSR')
plt.legend({'predicted', 'simulated', 'obs'})
plt.show()


# In[102]:


data = zonal_means_native(OLR_opt, area, lat, lon)
keys = list(data.keys())
values = list(data.values())

plt.scatter(keys, values)
plt.scatter(keys, OLR_final_preds[0:18])
plt.scatter(keys, zrg_obs.iloc[:, 75:93])
plt.xlabel('Center Latitiude of Band')
plt.title('OLR')
plt.legend({'predicted', 'simulated', 'obs'})
plt.show()


# ##### Correlations

# In[142]:


PCP_preds_flat = PCP_proj_gp.values.flatten()
PCP_scream_flat = PCP_test.values.flatten()
PCP_r_squared = r2_score(PCP_scream_flat, PCP_preds_flat)
PCP_rmse_squared = root_mean_squared_error(PCP_scream_flat, PCP_preds_flat)
PCP_r_squared_unflat = r2_score(PCP_test, PCP_proj_gp)

TLWP_preds_flat = TLWP_proj_gp.values.flatten()
TLWP_scream_flat = TLWP_test.values.flatten()
TLWP_r_squared = r2_score(TLWP_scream_flat, TLWP_preds_flat)
TLWP_rmse_squared = root_mean_squared_error(TLWP_scream_flat, TLWP_preds_flat)
TLWP_r_squared_unflat = r2_score(TLWP_test, TLWP_proj_gp)

OSR_preds_flat = OSR_proj_gp.values.flatten()
OSR_scream_flat = OSR_test.values.flatten()
OSR_r_squared = r2_score(OSR_scream_flat, OSR_preds_flat)
OSR_rmse_squared = root_mean_squared_error(OSR_scream_flat, OSR_preds_flat)
OSR_r_squared_unflat = r2_score(OSR_test, OSR_proj_gp)

OLR_preds_flat = OLR_proj_gp.values.flatten()
OLR_scream_flat = OLR_test.values.flatten()
OLR_r_squared = r2_score(OLR_scream_flat, OLR_preds_flat)
OLR_rmse_squared = root_mean_squared_error(OLR_scream_flat, OLR_preds_flat)
OLR_r_squared_unflat = r2_score(OLR_test, OLR_proj_gp)

print('r squared:', 'PCP:', PCP_r_squared, 'TLWP:', TLWP_r_squared, 'OSR:', OSR_r_squared, 'OLR:', OLR_r_squared)
print('     rmse:', 'PCP:', PCP_rmse_squared, 'TLWP:', TLWP_rmse_squared, 'OSR:', OSR_rmse_squared, 'OLR:', OLR_rmse_squared)

print('PCP:', PCP_r_squared_unflat, 'TLWP:', TLWP_r_squared_unflat, 'OSR:', OSR_r_squared_unflat, 'OLR:', OLR_r_squared_unflat)


# In[143]:


PCP_r_squared = r2_score(PCP_test.T, PCP_proj_gp.T)
PCP_rmse_squared = root_mean_squared_error(PCP_test.T, PCP_proj_gp.T)

TLWP_r_squared = r2_score(TLWP_test.T, TLWP_proj_gp.T)
TLWP_rmse_squared = root_mean_squared_error(TLWP_test.T, TLWP_proj_gp.T)

OSR_r_squared = r2_score(OSR_test.T, OSR_proj_gp.T)
OSR_rmse_squared = root_mean_squared_error(OSR_test.T, OSR_proj_gp.T)

OLR_r_squared = r2_score(OLR_test.T, OLR_proj_gp.T)
OLR_rmse_squared = root_mean_squared_error(OLR_test.T, OLR_proj_gp.T)

print('r squared:', 'PCP:', PCP_r_squared, 'TLWP:', TLWP_r_squared, 'OSR:', OSR_r_squared, 'OLR:', OLR_r_squared)
print('     rmse:', 'PCP:', PCP_rmse_squared, 'TLWP:', TLWP_rmse_squared, 'OSR:', OSR_rmse_squared, 'OLR:', OLR_rmse_squared)

k = 0
r squared: PCP: 0.9062514338857577 TLWP: 0.9407431482384747 OSR: 0.9712357082680589 OLR: 0.9765864016743739
     rmse: PCP: 2.5199177521915755e-08 TLWP: 0.03140594031862504 OSR: 12.557224112833428 OLR: 6.241991414593105
k = 1
r squared: PCP: 0.8476427814369366 TLWP: 0.8748461329967885 OSR: 0.947262060397879 OLR: 0.9193235211882576
     rmse: PCP: 3.429189969770179e-08 TLWP: 0.04231762249424831 OSR: 17.433392266266438 OLR: 12.132133447183799
k = 2
r squared: PCP: 0.9126713624089907 TLWP: 0.942302700692743 OSR: 0.975808449461033 OLR: 0.9814216697495927
     rmse: PCP: 2.4244083316375917e-08 TLWP: 0.02900550319236126 OSR: 11.37151519766062 OLR: 5.551501524576244
k = 3
r squared: PCP: 0.8977019914618889 TLWP: 0.9180134669387755 OSR: 0.9681500779912383 OLR: 0.9721014792067822
     rmse: PCP: 2.6539004234824845e-08 TLWP: 0.03624388204439803 OSR: 13.341499079616732 OLR: 6.916254481852578
k = 4
r squared: PCP: 0.8614315888155565 TLWP: 0.9107708774746086 OSR: 0.9604478062704074 OLR: 0.9558517415766998
     rmse: PCP: 3.102985734548764e-08 TLWP: 0.0360364986483419 OSR: 14.693133202314733 OLR: 8.647590815961077
# In[ ]:





# In[144]:


len(r2_score(OLR_test, OLR_proj_gp, multioutput = 'raw_values'))
len(r2_score(OLR_test.T, OLR_proj_gp.T, multioutput = 'raw_values'))
#multioutput{‘raw_values’, ‘uniform_average’, ‘variance_weighted’}, array-like of shape (n_outputs,) or None, default=’uniform_average’


# In[67]:


poles_ithink.shape


# In[124]:


r2_score(OLR_test.T, OLR_proj_gp.T)


# In[147]:


print(r2_score(OLR_test.values.T, OLR_proj_gp.T))

plt.scatter(OLR_test.T, OLR_proj_gp.T)


# In[162]:


i=5
file = 15
print(regions_list[i])

poles_ithink = regions_file.variables[regions_list[i]][0, :]
vectorpoles_test = (poles_ithink * OLR_test.T[:,file])
filtered_vectorpoles_test =  vectorpoles_test[vectorpoles_test != 0.]

vectorpoles_proj = (poles_ithink * (OLR_proj_gp.iloc[file]).to_numpy())
filtered_vectorpoles_proj =  vectorpoles_proj[vectorpoles_proj != 0.]
# = np.dot((poles_ithink*np.ones((21600,21600))), OLR_test.T)

print(r2_score(filtered_vectorpoles_test, filtered_vectorpoles_proj))

plt.scatter(filtered_vectorpoles_test, filtered_vectorpoles_proj)


# In[188]:


print('TLWP')
for i in range(6):
    print(regions_list[i])
    avg = np.zeros((30,1))
    for file in range(30):
        poles_ithink = regions_file.variables[regions_list[i]][0, :]
        vectorpoles_test = (poles_ithink * TLWP_test.T[:,file])
        filtered_vectorpoles_test =  vectorpoles_test[vectorpoles_test != 0.]

        vectorpoles_proj = (poles_ithink * (TLWP_proj_gp.iloc[file]).to_numpy())
        filtered_vectorpoles_proj =  vectorpoles_proj[vectorpoles_proj != 0.]
        # = np.dot((poles_ithink*np.ones((21600,21600))), OLR_test.T)

        #print(r2_score(filtered_vectorpoles_test, filtered_vectorpoles_proj))
        avg[file] = r2_score(filtered_vectorpoles_test, filtered_vectorpoles_proj)

    print(np.mean(avg))


# In[186]:


print('OSR')
for i in range(6):
    print(regions_list[i])
    avg = np.zeros((30,1))
    for file in range(30):
        poles_ithink = regions_file.variables[regions_list[i]][0, :]
        vectorpoles_test = (poles_ithink * OSR_test.T[:,file])
        filtered_vectorpoles_test =  vectorpoles_test[vectorpoles_test != 0.]

        vectorpoles_proj = (poles_ithink * (OSR_proj_gp.iloc[file]).to_numpy())
        filtered_vectorpoles_proj =  vectorpoles_proj[vectorpoles_proj != 0.]
        # = np.dot((poles_ithink*np.ones((21600,21600))), OLR_test.T)

        #print(r2_score(filtered_vectorpoles_test, filtered_vectorpoles_proj))
        avg[file] = r2_score(filtered_vectorpoles_test, filtered_vectorpoles_proj)

    print(np.mean(avg))


# In[187]:


print('OLR')
for i in range(6):
    print(regions_list[i])
    avg = np.zeros((30,1))
    for file in range(30):
        poles_ithink = regions_file.variables[regions_list[i]][0, :]
        vectorpoles_test = (poles_ithink * OLR_test.T[:,file])
        filtered_vectorpoles_test =  vectorpoles_test[vectorpoles_test != 0.]

        vectorpoles_proj = (poles_ithink * (OLR_proj_gp.iloc[file]).to_numpy())
        filtered_vectorpoles_proj =  vectorpoles_proj[vectorpoles_proj != 0.]
        # = np.dot((poles_ithink*np.ones((21600,21600))), OLR_test.T)

        #print(r2_score(filtered_vectorpoles_test, filtered_vectorpoles_proj))
        avg[file] = r2_score(filtered_vectorpoles_test, filtered_vectorpoles_proj)

    print(np.mean(avg))


# In[ ]:





# In[ ]:





# In[ ]:


GP_dict_vars_regional_test_rf = dict()
Y_test_shape = PCP_test.shape

area = np.zeros(Y_test_shape)
for i in range(Y_test_shape[0]):  # this is the same as shape[0]
    area[i, :] = ar  
time_days = np.array(range(tc * Y_test_shape[0]))

for i in range(len(regions_list)):  # for each region
    reg = regions_file.variables[regions_list[i]][0, :]
    reg = np.array(reg, dtype=float)  # extract region mask
    region = np.zeros(Y_test_shape)  # this is number of files by ncol
    ocean = np.zeros(Y_test_shape)
    oc = 1 - landfrac 
    oc[oc < 0.5] = 0
    oc[oc >= 0.5] = 1
    oc[oc == 0] = np.nan  # oc is every cell that is less than half land (from landfrac)
    for j in range(Y_test_shape[0]):
        region[j, :] = np.array(reg, dtype=float)
        ocean[j, :] = 1 - np.array(landfrac, dtype=float)
    ocean[ocean > 0.5] = 1
    ocean[ocean < 0.5] = 0
    ocean[ocean == 0] = np.nan
    region[region == 0] = np.nan
    reg[reg == 0] = np.nan
    
    regional_area_weighting[i] = np.nansum(ar * reg) 
    regional_area_weighting_ocean[i] = np.nansum(ar * reg * oc)

    ## Test variables
    # dict_vars_regions[regions_list[i]+'PCP'] = 1e3*24*3600*np.nanmean(PCP*area*region,(1))/np.nanmean(area*region,(1))  # why is this multiplied by a coefficient?
    GP_dict_vars_regional_r2[regions_list[i] + 'PCP'] = 24 * 3600 * np.nanmean(PCP_test_df * area * region, (1)) / np.nanmean(area * region, (1))
    GP_dict_vars_regional_r2[regions_list[i] + 'OSR'] = np.nanmean((OSR_test_df) * area * region, (1)) / np.nanmean(area * region, (1))
    GP_dict_vars_regional_r2[regions_list[i] + 'OLR'] = np.nanmean((OLR_test_df) * area * region, (1)) / np.nanmean(area * region, (1))
    GP_dict_vars_regional_test_rf[regions_list[i] + 'TLWP'] = np.nanmean((TLWP_test_df) * area * region, (1)) / np.nanmean(area * region, (1))

    ## Projected variables
    # dict_vars_regions[regions_list[i]+'PCP'] = 1e3*24*3600*np.nanmean(PCP*area*region,(1))/np.nanmean(area*region,(1))  # why is this multiplied by a coefficient?
    dict_vars_regional_proj_rf[regions_list[i] + 'PCP'] = 24 * 3600 * np.nanmean(PCP_proj_rf * area * region, (1)) / np.nanmean(area * region, (1))
    dict_vars_regional_proj_rf[regions_list[i] + 'OSR'] = np.nanmean(OSR_proj_rf * area * region, (1)) / np.nanmean(area * region, (1))
    dict_vars_regional_proj_rf[regions_list[i] + 'OLR'] = np.nanmean(OLR_proj_rf * area * region, (1)) / np.nanmean(area * region, (1))
    dict_vars_regional_proj_rf[regions_list[i] + 'TLWP'] = np.nanmean(TLWP_proj_rf * area * region, (1)) / np.nanmean(area * region, (1))


# In[ ]:





# In[80]:


PCP_gp_R2 = regional_correlation_squared(PCP_regions_test, PCP_regions_proj_gp)
TLWP_gp_R2 = regional_correlation_squared(TLWP_regions_test, TLWP_regions_proj_gp)
OSR_gp_R2 = regional_correlation_squared(OSR_regions_test, OSR_regions_proj_gp)
OLR_gp_R2 = regional_correlation_squared(OLR_regions_test, OLR_regions_proj_gp)
PCP_gp_R2, TLWP_gp_R2, OSR_gp_R2, OLR_gp_R2


# In[ ]:





# In[ ]:





# In[ ]:


from esem import rf_model
from esem.utils import plot_results, prettify_plot, add_121_line, leave_one_out

#from the example
precip = df_crm['precip'].to_numpy().reshape(-1 ,1)
# In[ ]:


precip_data = ppe_dataset_small.precip_total_surf_mass_flux.values


# Leave one out cross validation ... ?
# 
# This took too long

# In[ ]:


# Ignore the mountain of warnings
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
outp = np.asarray(leave_one_out(Xdata=ppe_params, Ydata=precip_data, model='RandomForest', n_estimators=50, random_state=0))


# In[ ]:





# In[ ]:


X_train_rf = X_train
Y_train_rf = np.stack((PCP_train.values, TLWP_train.values, OSR_train.values, OLR_train.values), axis = 0)
Y_train_rf = np.transpose(Y_train_rf, (1, 2, 0))
print(X_train_norm.shape, Y_train_norm.shape)


# In[ ]:


model_rf = rf_model(X_train_rf, Y_train_rf)


# In[ ]:


model_rf.train()


# In[ ]:


preds_rf,_ = model_rf.predict(X_test)
preds_rf.shape


# In[ ]:


PCP_proj_rf = pd.DataFrame(preds_rf[:, :, 0], index = X_test.index)
TLWP_proj_rf = pd.DataFrame(preds_rf[:, :, 1], index = X_test.index)
OSR_proj_rf = pd.DataFrame(preds_rf[:, :, 2], index = X_test.index)
OLR_proj_rf = pd.DataFrame(preds_rf[:, :, 3], index = X_test.index)


# In[ ]:


PCP_test_df = pd.DataFrame(PCP_test, index=PCP_proj_rf.index)
TLWP_test_df = pd.DataFrame(TLWP_test, index=TLWP_proj_rf.index)
OSR_test_df = pd.DataFrame(OSR_test, index=OSR_proj_rf.index)
OLR_test_df = pd.DataFrame(OLR_test, index=OLR_proj_rf.index)


# ##### Regional Work

# In[ ]:


dict_vars_regional_test_rf = dict()
dict_vars_regional_proj_rf = dict()


# In[ ]:


lat_bands = np.linspace(-90,90,18) #this is where to change the refinement of the zones; start, end, # of partitions #Peter suggested 10 degree bands
regions_file = S.Dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/regions.nc')
regions_list = ['poles','extratropical_land','extratropical_ocean','tropical_land','ascending_tropical_ocean','descending_tropical_ocean']


# In[ ]:


regional_area_weighting = np.zeros(len(regions_list))
regional_area_weighting_ocean = np.zeros(len(regions_list))
landfrac = regions_file.variables['landfrac'][0,:]


# In[ ]:


Y_test_shape = PCP_test.shape

area = np.zeros(Y_test_shape)
for i in range(Y_test_shape[0]):  # this is the same as shape[0]
    area[i, :] = ar  
time_days = np.array(range(tc * Y_test_shape[0]))

for i in range(len(regions_list)):  # for each region
    reg = regions_file.variables[regions_list[i]][0, :]
    reg = np.array(reg, dtype=float)  # extract region mask
    region = np.zeros(Y_test_shape)  # this is number of files by ncol
    ocean = np.zeros(Y_test_shape)
    oc = 1 - landfrac 
    oc[oc < 0.5] = 0
    oc[oc >= 0.5] = 1
    oc[oc == 0] = np.nan  # oc is every cell that is less than half land (from landfrac)
    for j in range(Y_test_shape[0]):
        region[j, :] = np.array(reg, dtype=float)
        ocean[j, :] = 1 - np.array(landfrac, dtype=float)
    ocean[ocean > 0.5] = 1
    ocean[ocean < 0.5] = 0
    ocean[ocean == 0] = np.nan
    region[region == 0] = np.nan
    reg[reg == 0] = np.nan
    
    regional_area_weighting[i] = np.nansum(ar * reg) 
    regional_area_weighting_ocean[i] = np.nansum(ar * reg * oc)

    ## Test variables
    # dict_vars_regions[regions_list[i]+'PCP'] = 1e3*24*3600*np.nanmean(PCP*area*region,(1))/np.nanmean(area*region,(1))  # why is this multiplied by a coefficient?
    dict_vars_regional_test_rf[regions_list[i] + 'PCP'] = 24 * 3600 * np.nanmean(PCP_test_df * area * region, (1)) / np.nanmean(area * region, (1))
    dict_vars_regional_test_rf[regions_list[i] + 'OSR'] = np.nanmean((OSR_test_df) * area * region, (1)) / np.nanmean(area * region, (1))
    dict_vars_regional_test_rf[regions_list[i] + 'OLR'] = np.nanmean((OLR_test_df) * area * region, (1)) / np.nanmean(area * region, (1))
    dict_vars_regional_test_rf[regions_list[i] + 'TLWP'] = np.nanmean((TLWP_test_df) * area * region, (1)) / np.nanmean(area * region, (1))

    ## Projected variables
    # dict_vars_regions[regions_list[i]+'PCP'] = 1e3*24*3600*np.nanmean(PCP*area*region,(1))/np.nanmean(area*region,(1))  # why is this multiplied by a coefficient?
    dict_vars_regional_proj_rf[regions_list[i] + 'PCP'] = 24 * 3600 * np.nanmean(PCP_proj_rf * area * region, (1)) / np.nanmean(area * region, (1))
    dict_vars_regional_proj_rf[regions_list[i] + 'OSR'] = np.nanmean(OSR_proj_rf * area * region, (1)) / np.nanmean(area * region, (1))
    dict_vars_regional_proj_rf[regions_list[i] + 'OLR'] = np.nanmean(OLR_proj_rf * area * region, (1)) / np.nanmean(area * region, (1))
    dict_vars_regional_proj_rf[regions_list[i] + 'TLWP'] = np.nanmean(TLWP_proj_rf * area * region, (1)) / np.nanmean(area * region, (1))


# ##### Correlations

# In[ ]:


regional_corr_matrix_rf = pd.DataFrame(columns = regions_list, index = vars_list)
for i in (vars_list): #for all variables
    for j in (regions_list): #for all regions
        slope, intercept, r, p, se = stats.linregress(dict_vars_regional_test_rf[j+i],dict_vars_regional_proj_rf[j+i]) #r is the Pearson correlation coefficient. The square of rvalue is equal to the coefficient of determination.
        regional_corr_matrix_rf.at[i,j] = r
regional_corr_matrix_rf.pow(2)


# In[ ]:





# In[ ]:


PCP_rf_R2 = regional_correlation_squared(PCP_test_df, PCP_proj_rf)
TLWP_rf_R2 = regional_correlation_squared(TLWP_test_df, TLWP_proj_rf)
OSR_rf_R2 = regional_correlation_squared(OSR_test_df, OSR_proj_rf)
OLR_rf_R2 = regional_correlation_squared(OLR_test_df, OLR_proj_rf)
PCP_rf_R2, TLWP_rf_R2, OSR_rf_R2, OLR_rf_R2


# In[ ]:


# Now, make grid for plotting RF predictions
# more n_points means higher resolution, but takes exponentially longer
n_points = 10

min_vals = ppe_params.min()
max_vals = ppe_params.max()

# For uniform prediction over full params space
space=np.linspace(min_vals, max_vals, n_points)

# Reshape to (N,D)
reshape_to_ND = np.transpose(space)
Xs_uniform = np.meshgrid(*reshape_to_ND)
test = np.array([_.flatten() for _ in Xs_uniform]).T

# Predict
predictions,_ = model_rf.predict(test)
predictions   = predictions.reshape(Xs_uniform[0].shape)

# Now, take mean over all parameters except [LWP, z700], assumed to be first 2 indices
predictions_reduced = np.mean(predictions, axis=tuple(range(2, predictions.ndim)))


# In[ ]:





# In[ ]:





# In[7]:


# Load back

with open(GP_proj_filename, 'rb') as f:
    loaded = pickle.load(f)

# Access data
restored_pipeline = loaded['X_pipeline']
restored_df1 = loaded['proj_gp']


# In[ ]:


global/cfs/cdirs/e3sm/jpaige3/ESEm/cv5_final_predictions/optmar26e_preds_remapped.ec


# In[5]:


template_xr = xr.open_dataset('/global/cfs/cdirs/e3sm/jpaige3/ESEm/cv5_final_predictions/template.nc')


# In[2]:


test_ordered_list = ['m0020', 'm0027', 'm0033', 'm0037', 'm0038', 'm0053', 'm0060', 'm0062', 'm0068', 'm0085', 'm0110', 'm0117', 'm0125', 'm0131', 'm0132', 'm0133', 'm0161', 'm0164', 'm0165', 'm0185', 'm0216', 'm0225', 'm0226', 'm0228', 'm0252', 'm0255', 'm0283', 'm0284', 'optfeb20', 'optmar15', 'optmar26c',
 'm0001', 'm0006', 'm0023', 'm0026', 'm0034', 'm0042', 'm0050', 'm0054', 'm0057', 'm0058', 'm0063', 'm0079', 'm0089', 'm0096', 'm0109', 'm0128', 'm0140', 'm0147', 'm0153', 'm0180', 'm0183', 'm0217', 'm0218', 'm0224', 'm0227', 'm0244', 'm0251', 'm0254', 'm0260', 'optmar22ga', 'optmar26d',
 'm0009', 'm0017', 'm0049', 'm0051', 'm0065', 'm0066', 'm0067', 'm0074', 'm0088', 'm0091', 'm0094', 'm0101', 'm0121', 'm0124', 'm0137', 'm0148', 'm0159', 'm0160', 'm0163', 'm0181', 'm0186', 'm0190', 'm0199', 'm0203', 'm0205', 'm0231', 'm0232', 'optmar22hb', 'optmar22hc', 'optmar26f',
 'm0004', 'm0005', 'm0015', 'm0018', 'm0028', 'm0035', 'm0073', 'm0087', 'm0090', 'm0095', 'm0106', 'm0119', 'm0123', 'm0134', 'm0136', 'm0162', 'm0168', 'm0197', 'm0213', 'm0238', 'm0240', 'm0246', 'm0248', 'm0280', 'm0281', 'm0285', 'm0286', 'optmar22gb', 'optmar26a', 'optmar26e',
 'm0003', 'm0032', 'm0039', 'm0040', 'm0069', 'm0080', 'm0092', 'm0099', 'm0111', 'm0114', 'm0115', 'm0135', 'm0138', 'm0154', 'm0166', 'm0167', 'm0170', 'm0174', 'm0176', 'm0193', 'm0200', 'm0201', 'm0207', 'm0229', 'm0243', 'm0257', 'm0259', 'optmar22ha', 'optmar26b', 'optmar5']


# In[3]:


predictions_xr = xr.open_dataset('/global/cfs/cdirs/e3sm/jpaige3/ESEm/cv5_final_predictions/esem_predictions_output.nc')
predictions_xr['run_label'] = test_ordered_list


# In[51]:


PCP_predictions = pd.DataFrame(predictions_xr.precip_total_surf_mass_flux, index = predictions_xr.run_label)
PCP_scream = pd.DataFrame(ppe_dataset_small.precip_total_surf_mass_flux, index = ppe_dataset_small.run_label)


# In[57]:


PCP_scream_reorder = PCP_scream.loc[PCP_predictions.index]


# In[60]:


from sklearn.metrics import explained_variance_score
explained_variance = explained_variance_score(PCP_scream_reorder.T, PCP_predictions.T)
print(f'Explained Variance Score: {explained_variance}')


# ## Training the final model

# In[34]:


X_train = ppe_params
#X_test = ppe_params.iloc[test_index]
train_run_labels = X_train.index.to_list()
#test_run_labels = X_test.index.to_list()
#print('k =', i, test_run_labels)
        
Y_train = ppe_dataset_small
#Y_test = ppe_dataset_small
Y_train_array = ppe_dataset_small.to_array()
#Y_test_array = ppe_dataset_small.to_array()

#print("X_test shape:", X_test.shape, "type:", type(X_test))
print("X_train shape:", X_train.shape, "type:", type(X_train))
#print("Y_test shape:", Y_test_array.shape, "type:", type(Y_test_array))
print("Y_train shape:", Y_train_array.shape, "type:", type(Y_train_array))


# In[38]:


PCP_train = PCP_zrg_ppedataset.loc[train_run_labels] #**(1/8)
TLWP_train = TLWP_zrg_ppedataset.loc[train_run_labels] #**(1/4)
OSR_train = OSR_zrg_ppedataset.loc[train_run_labels] #**(1/4)
OLR_train = OLR_zrg_ppedataset.loc[train_run_labels] #**(1/8)

PCP_train.columns = PCP_zrg_ppedataset.columns.astype(str)
TLWP_train.columns = TLWP_zrg_ppedataset.columns.astype(str)
OSR_train.columns = OSR_zrg_ppedataset.columns.astype(str)
OLR_train.columns = OLR_zrg_ppedataset.columns.astype(str)

vars_train_list = [PCP_train, TLWP_train, OSR_train, OLR_train]


# In[44]:


#transform data
X_pipe_sk_minmax = preprocessing.MinMaxScaler()
X_pipe_sk_minmax.fit(X_train)
X_train_norm = X_pipe_sk_minmax.transform(X_train)

#from scikitlearn
Y_pipe_sk_ss_PCP = preprocessing.StandardScaler()
Y_pipe_sk_ss_PCP.fit(PCP_train)
PCP_train_norm = Y_pipe_sk_ss_PCP.transform(PCP_train)

Y_pipe_sk_ss_TLWP = preprocessing.StandardScaler() #lots of other options for this: RobustScaler(), etc.
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


# In[10]:


print(X_train_norm.shape, Y_train_norm.shape) #the first dimension of these must match


# In[11]:


model_gp = gp_model(X_train_norm, Y_train_norm) #creates the model form, but is not trained yet
#this default kernel has been useful (combination of linear, polynomial, and RBF) and outperformed other options individually


# In[ ]:




