#!/usr/bin/env python
# coding: utf-8

# # Preprocessing surrogate training workflow
# 

# Load in packages - Tensorflow warnings may affect ESEM GP performance

# In[1]:


import os
import json
import math
import pandas as pd
import numpy as np
import xarray as xr
import glob
import pickle
from esem import gp_model
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

#removing files that don't have all the data--could be doing this better
DY2_folders.remove('m0230') #this file is missing 3hr averages
DY2_folders.remove('optmar20day5') #this file is missing 2nd day daily averages

#not in DY1
DY2_folders.remove('m0024')
DY2_folders.remove('m0025')
DY2_folders.remove('m0061')
DY2_folders.remove('optmar22hd')
DY2_folders.remove('optmar15')
DY2_folders.remove('optmar20day2')

print(len(DY2_folders)) 
DY2_folders = sorted(DY2_folders)

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
DY1_folders = sorted(DY1_folders)

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

#Check for intersection between DY1 and DY2, take only runs in both
print(list(set(DY1_sim_names) - set(DY2_sim_names))) # should be empty
print(list(set(DY2_sim_names) - set(DY1_sim_names)))
sim_names = [sim for sim in DY1_sim_names if sim in DY2_sim_names] 

ppe_params = ppe_params_all[ppe_params_all.index.isin(sim_names)]
ppe_params
# In[6]:


#Check for intersection between DY1 and DY2, take only runs in both
print(list(set(DY1_sim_names) - set(DY2_sim_names))) # should be empty
print(list(set(DY2_sim_names) - set(DY1_sim_names)))
sim_names = [sim for sim in DY1_sim_names if sim in DY2_sim_names] 

ppe_params = ppe_params_all.loc[sim_names]
ppe_params


# In[25]:


# Open and concatenate the datasets along the new 'run_number' dimension
DY1_concat_dataset = xr.open_mfdataset(DY1_filename_list_filtered, concat_dim='run_label', combine='nested')
DY1_sim_names_from_paths = [path.split('/')[-4] for path in DY1_filename_list_filtered]
DY1_ppe_dataset_time = DY1_concat_dataset.assign_coords(run_label=('run_label', DY1_sim_names_from_paths))
#DY1_ppe_dataset_time = DY1_concat_dataset.assign_coords(run_label=('run_label', DY1_sim_names)) # Assign the 'run_number' coordinate
DY1_ppe_dataset = DY1_ppe_dataset_time.squeeze('time')


# In[27]:


# Open and concatenate the datasets along the new 'run_number' dimension
DY2_concat_dataset = xr.open_mfdataset(DY2_filename_list_filtered, concat_dim='run_label', combine='nested')
DY2_sim_names_from_paths = [path.split('/')[-4] for path in DY2_filename_list_filtered]
DY2_ppe_dataset_time = DY2_concat_dataset.assign_coords(run_label=('run_label', DY2_sim_names_from_paths))
#DY2_ppe_dataset_time = DY2_concat_dataset.assign_coords(run_label=('run_label', DY2_sim_names)) # Assign the 'run_number' coordinate
DY2_ppe_dataset = DY2_ppe_dataset_time.squeeze('time')


# ##### Observations

# In[28]:


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


# In[29]:


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

# In[30]:


#filter variables in output - keep only the 4 of interest
to_keep = ['precip_total_surf_mass_flux','LiqWaterPath','RainWaterPath','SW_flux_up_at_model_top','LW_flux_up_at_model_top']
           
to_leave = ['SW_flux_dn','SW_flux_dn_at_model_bot','SW_flux_up','SW_flux_up_at_model_bot','SW_flux_dn_at_model_top', 'T_2m',
'T_mid', 'precip_ice_surf_mass_flux','precip_liq_surf_mass_flux','ps','qc','qi','qm','qr','qv','qv_2m','LW_flux_up','LW_flux_up_at_model_bot',
'IceWaterPath', 'LW_flux_dn','LW_flux_dn_at_model_bot','time_bnds', 'LongwaveCloudForcing', 'MeridionalVapFlux','ShortwaveCloudForcing','U','V', 
'VapWaterPath', 'ZonalVapFlux','bm','eddy_diff_mom', 'eff_radius_qc_at_cldtop','eff_radius_qi_at_cldtop', 'homme_T_mid_tend', 'homme_qv_tend',
'horiz_winds_at_model_bot', 'nc', 'ni', 'nr', 'omega', 'p3_T_mid_tend', 'p3_qv_tend', 'rrtmgp_T_mid_tend', 'sgs_buoy_flux', 'shoc_T_mid_tend',
'shoc_qv_tend', 'surf_evap', 'surf_mom_flux', 'surf_radiative_T','surf_sens_flux', 'surface_upward_latent_heat_flux', 'avg_count_ncol', 
'avg_count_ncol_lev', 'avg_count_ncol_dim', 'area', 'lat', 'lon']


# In[31]:


DY1_ppe_dataset_small = DY1_ppe_dataset.drop_vars(to_leave)
DY1_ppe_dataset_small['TotalLiqWaterPath'] = (DY1_ppe_dataset_small.LiqWaterPath + DY1_ppe_dataset_small.RainWaterPath)
#ppe_dataset_small['precip_total_surf_mass_flux'] = ppe_dataset_small['precip_total_surf_mass_flux']*1e-3*24*3600
DY1_ppe_dataset_small = DY1_ppe_dataset_small.drop_vars('p_levs')
DY1_ppe_dataset_small = DY1_ppe_dataset_small.rename({var: f"DY1_{var}" for var in DY1_ppe_dataset_small.data_vars})


# In[32]:


DY2_ppe_dataset_small = DY2_ppe_dataset.drop_vars(to_leave)
DY2_ppe_dataset_small['TotalLiqWaterPath'] = (DY2_ppe_dataset_small.LiqWaterPath + DY2_ppe_dataset_small.RainWaterPath)
#ppe_dataset_small['precip_total_surf_mass_flux'] = ppe_dataset_small['precip_total_surf_mass_flux']*1e-3*24*3600]
#DY2_ppe_dataset_small = DY2_ppe_dataset_small.drop_vars('p_levs')
DY2_ppe_dataset_small = DY2_ppe_dataset_small.rename({var: f"DY2_{var}" for var in DY2_ppe_dataset_small.data_vars})


# In[33]:


assert list(ppe_params.index) == sim_names
assert list(DY1_ppe_dataset_small.run_label.values) == sim_names
assert list(DY2_ppe_dataset_small.run_label.values) == sim_names

#ppe_dataset_small = xr.concat([DY1_ppe_dataset_small, DY2_ppe_dataset_small], dim='run_label')
ppe_dataset_small = DY1_ppe_dataset_small.sel(run_label=sim_names).combine_first(DY2_ppe_dataset_small.sel(run_label=sim_names))
ppe_dataset_smallUnits
SCREAM
float precip_total_surf_mass_flux(time, ncol) ;
		precip_total_surf_mass_flux:units = "m s^-1" ;  #*1e3*24*3600 --- this is a conversion Hassan had; typically in mm/day - (precip/100*3600*24)
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

# In[34]:


##two ways to get these masks
mask_path ='/global/cfs/cdirs/e3sm/jpaige3/mahf708_newvs_surrogate/Autotuning-NGD/src/scream-jp/surrogate_and_optimization/'

DY1_PCP_mask = xr.open_dataset(mask_path+'/masks/precip_1_output.nc').squeeze('time', drop=True)
DY2_PCP_mask = xr.open_dataset(mask_path+'/masks/precip_2_output.nc').squeeze('time', drop=True)
DY1_TLWP_mask = xr.open_dataset(mask_path+'/masks/tlwp_1_output.nc').squeeze('time', drop=True)
DY2_TLWP_mask = xr.open_dataset(mask_path+'/masks/tlwp_2_output.nc').squeeze('time', drop=True)

#len(np.where(((DY1_TLWP_mask.to_dataarray()[0]) == True).values)[0]) - ncol drops from 21600 to 15313


# In[35]:


DY1_ppe_dataset_mask = DY1_ppe_dataset_small.copy(deep=True)
DY2_ppe_dataset_mask = DY2_ppe_dataset_small.copy(deep=True)

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


# In[36]:


regions_file = xr.open_dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/regions.nc')
regions_list = ['poles','extratropical_land','extratropical_ocean','tropical_land','ascending_tropical_ocean','descending_tropical_ocean']
#area = ppe_dataset.area[1,:] #only taking the first row, because all rows should have the same values
control = xr.open_dataset('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/SCREAM.2024-autocal-00.ne1024pg2/m0000/SCREAM.2024-autocal-00.ne1024pg2/run/output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc')
area = control.variables['area'][:]
lat = control.variables['lat'][:]
lon = control.variables['lon'][:]


# In[37]:


def zonal_means_native(data, area, lat, lon):
    data = data.squeeze()
    area = area.squeeze()
    lat = lat.squeeze()
    lat_bands = np.linspace(-90, 90, 19) #currently dividing globe in 18 zones via 19 borders - 10 degree bands
    zonal_means = dict()
    for i in range(len(lat_bands) - 1):
        mask_zone = (lat >= lat_bands[i]) & (lat < lat_bands[i+1]) #includes lower bound in the band (! might cause missing north pole)
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


# Below we are just taking the averages, in a likely not very efficient way... :(

# In[38]:


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
    precip = DY1_ppe_dataset_mask.sel(run_label = run)['DY1_precip_total_surf_mass_flux']
    tlwp = DY1_ppe_dataset_mask.sel(run_label = run)['DY1_TotalLiqWaterPath']
    swflux = DY1_ppe_dataset_mask.sel(run_label = run)['DY1_SW_flux_up_at_model_top']
    lwflux = DY1_ppe_dataset_mask.sel(run_label = run)['DY1_LW_flux_up_at_model_top']

    DY1_PCP_zonal_data[run] = zonal_means_native(precip, area, lat, lon)
    DY1_TLWP_zonal_data[run] = zonal_means_native(tlwp, area, lat, lon)
    DY1_OSR_zonal_data[run] = zonal_means_native(swflux, area, lat, lon)
    DY1_OLR_zonal_data[run] = zonal_means_native(lwflux, area, lat, lon)
    
    DY1_PCP_regional_data[run] = regional_means_native(precip, area, regions_file)
    DY1_TLWP_regional_data[run] = regional_means_native(tlwp, area, regions_file)
    DY1_OSR_regional_data[run] = regional_means_native(swflux, area, regions_file)
    DY1_OLR_regional_data[run] = regional_means_native(lwflux, area, regions_file)    
    
    DY1_PCP_global_data.append(global_means_native(precip, area))
    DY1_TLWP_global_data.append(global_means_native(tlwp, area))
    DY1_OSR_global_data.append(global_means_native(swflux, area))
    DY1_OLR_global_data.append(global_means_native(lwflux, area))


# In[39]:


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
    precip = DY2_ppe_dataset_mask.sel(run_label = run)['DY2_precip_total_surf_mass_flux']
    tlwp = DY2_ppe_dataset_mask.sel(run_label = run)['DY2_TotalLiqWaterPath']
    swflux = DY2_ppe_dataset_mask.sel(run_label = run)['DY2_SW_flux_up_at_model_top']
    lwflux = DY2_ppe_dataset_mask.sel(run_label = run)['DY2_LW_flux_up_at_model_top']

    DY2_PCP_zonal_data[run] = zonal_means_native(precip, area, lat, lon)
    DY2_TLWP_zonal_data[run] = zonal_means_native(tlwp, area, lat, lon)
    DY2_OSR_zonal_data[run] = zonal_means_native(swflux, area, lat, lon)
    DY2_OLR_zonal_data[run] = zonal_means_native(lwflux, area, lat, lon)
    
    DY2_PCP_regional_data[run] = regional_means_native(precip, area, regions_file)
    DY2_TLWP_regional_data[run] = regional_means_native(tlwp, area, regions_file)
    DY2_OSR_regional_data[run] = regional_means_native(swflux, area, regions_file)
    DY2_OLR_regional_data[run] = regional_means_native(lwflux, area, regions_file)    
    
    DY2_PCP_global_data.append(global_means_native(precip, area))
    DY2_TLWP_global_data.append(global_means_native(tlwp, area))
    DY2_OSR_global_data.append(global_means_native(swflux, area))
    DY2_OLR_global_data.append(global_means_native(lwflux, area))


# In[40]:


DY1_PCP_z_df = pd.DataFrame.from_dict(DY1_PCP_zonal_data, orient="index")
DY1_PCP_r_df = pd.DataFrame.from_dict(DY1_PCP_regional_data, orient="index")
DY1_PCP_zrg_ppedataset = pd.concat([DY1_PCP_z_df, DY1_PCP_r_df], axis=1)
DY1_PCP_zrg_ppedataset["DY1_global"] = DY1_PCP_global_data

DY2_PCP_z_df = pd.DataFrame.from_dict(DY2_PCP_zonal_data, orient="index")
DY2_PCP_r_df = pd.DataFrame.from_dict(DY2_PCP_regional_data, orient="index")
DY2_PCP_zrg_ppedataset = pd.concat([DY2_PCP_z_df, DY2_PCP_r_df], axis=1)
DY2_PCP_zrg_ppedataset["DY2_global"] = DY2_PCP_global_data

PCP_zrg_ppedataset = pd.concat([DY1_PCP_zrg_ppedataset, DY2_PCP_zrg_ppedataset], axis=1)


# In[41]:


DY1_TLWP_z_df = pd.DataFrame.from_dict(DY1_TLWP_zonal_data, orient="index")
DY1_TLWP_r_df = pd.DataFrame.from_dict(DY1_TLWP_regional_data, orient="index")
DY1_TLWP_zrg_ppedataset = pd.concat([DY1_TLWP_z_df, DY1_TLWP_r_df], axis=1)
DY1_TLWP_zrg_ppedataset["DY1_global"] = DY1_TLWP_global_data

DY2_TLWP_z_df = pd.DataFrame.from_dict(DY2_TLWP_zonal_data, orient="index")
DY2_TLWP_r_df = pd.DataFrame.from_dict(DY2_TLWP_regional_data, orient="index")
DY2_TLWP_zrg_ppedataset = pd.concat([DY2_TLWP_z_df, DY2_TLWP_r_df], axis=1)
DY2_TLWP_zrg_ppedataset["DY2_global"] = DY2_TLWP_global_data

TLWP_zrg_ppedataset = pd.concat([DY1_TLWP_zrg_ppedataset, DY2_TLWP_zrg_ppedataset], axis=1)


# In[42]:


DY1_OSR_z_df = pd.DataFrame.from_dict(DY1_OSR_zonal_data, orient="index")
DY1_OSR_r_df = pd.DataFrame.from_dict(DY1_OSR_regional_data, orient="index")
DY1_OSR_zrg_ppedataset = pd.concat([DY1_OSR_z_df, DY1_OSR_r_df], axis=1)
DY1_OSR_zrg_ppedataset["DY1_global"] = DY1_OSR_global_data

DY2_OSR_z_df = pd.DataFrame.from_dict(DY2_OSR_zonal_data, orient="index")
DY2_OSR_r_df = pd.DataFrame.from_dict(DY2_OSR_regional_data, orient="index")
DY2_OSR_zrg_ppedataset = pd.concat([DY2_OSR_z_df, DY2_OSR_r_df], axis=1)
DY2_OSR_zrg_ppedataset["DY2_global"] = DY2_OSR_global_data

OSR_zrg_ppedataset = pd.concat([DY1_OSR_zrg_ppedataset, DY2_OSR_zrg_ppedataset], axis=1)


# In[43]:


DY1_OLR_z_df = pd.DataFrame.from_dict(DY1_OLR_zonal_data, orient="index")
DY1_OLR_r_df = pd.DataFrame.from_dict(DY1_OLR_regional_data, orient="index")
DY1_OLR_zrg_ppedataset = pd.concat([DY1_OLR_z_df, DY1_OLR_r_df], axis=1)
DY1_OLR_zrg_ppedataset["DY1_global"] = DY1_OLR_global_data

DY2_OLR_z_df = pd.DataFrame.from_dict(DY2_OLR_zonal_data, orient="index")
DY2_OLR_r_df = pd.DataFrame.from_dict(DY2_OLR_regional_data, orient="index")
DY2_OLR_zrg_ppedataset = pd.concat([DY2_OLR_z_df, DY2_OLR_r_df], axis=1)
DY2_OLR_zrg_ppedataset["DY2_global"] = DY2_OLR_global_data

OLR_zrg_ppedataset = pd.concat([DY1_OLR_zrg_ppedataset, DY2_OLR_zrg_ppedataset], axis=1)


# In[44]:


#### Ordering matters here
zrg_ppedataset = pd.concat([PCP_zrg_ppedataset, TLWP_zrg_ppedataset, OSR_zrg_ppedataset, OLR_zrg_ppedataset], axis=1) 


# In[45]:


zrg_ppedataset #for both DY1 and DY2, should be 200 wide (2 seasons of 4 variables of geographic averages) and number of runs tall


# In[46]:


assert list(ppe_params.index) == sim_names
assert list(zrg_ppedataset.index) == sim_names
assert list(DY1_ppe_dataset_small.run_label.values) == sim_names
assert list(DY2_ppe_dataset_small.run_label.values) == sim_names


# In[47]:


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
    
DY1_PCP_regional_obs['obs'] = regional_means_native(DY1_PCP_obs, area, regions_file)
DY1_TLWP_regional_obs['obs'] = regional_means_native(DY1_TLWP_obs, area, regions_file)
DY1_OSR_regional_obs['obs'] = regional_means_native(DY1_OSR_obs, area, regions_file)
DY1_OLR_regional_obs['obs'] = regional_means_native(DY1_OLR_obs, area, regions_file)    
    
DY1_PCP_global_obs.append(global_means_native(DY1_PCP_obs, area))
DY1_TLWP_global_obs.append(global_means_native(DY1_TLWP_obs, area))
DY1_OSR_global_obs.append(global_means_native(DY1_OSR_obs, area))
DY1_OLR_global_obs.append(global_means_native(DY1_OLR_obs, area))


# In[48]:


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
    
DY2_PCP_regional_obs['obs'] = regional_means_native(DY2_PCP_obs, area, regions_file)
DY2_TLWP_regional_obs['obs'] = regional_means_native(DY2_TLWP_obs, area, regions_file)
DY2_OSR_regional_obs['obs'] = regional_means_native(DY2_OSR_obs, area, regions_file)
DY2_OLR_regional_obs['obs'] = regional_means_native(DY2_OLR_obs, area, regions_file)    
    
DY2_PCP_global_obs.append(global_means_native(DY2_PCP_obs, area))
DY2_TLWP_global_obs.append(global_means_native(DY2_TLWP_obs, area))
DY2_OSR_global_obs.append(global_means_native(DY2_OSR_obs, area))
DY2_OLR_global_obs.append(global_means_native(DY2_OLR_obs, area))


# In[49]:


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


# In[50]:


zrg_obs #should just be one row of the observations by 200


# ### Training preparation

# In[51]:


X_train = ppe_params


# In[52]:


train_run_labels = zrg_ppedataset.index.to_list()


# In[53]:


PCP_train = PCP_zrg_ppedataset #.loc[train_run_labels] #**(1/8)
TLWP_train = TLWP_zrg_ppedataset #.loc[train_run_labels] #**(1/4)
OSR_train = OSR_zrg_ppedataset #.loc[train_run_labels] #**(1/4)
OLR_train = OLR_zrg_ppedataset #.loc[train_run_labels] #**(1/8)

PCP_train.columns = PCP_zrg_ppedataset.columns.astype(str)
TLWP_train.columns = TLWP_zrg_ppedataset.columns.astype(str)
OSR_train.columns = OSR_zrg_ppedataset.columns.astype(str)
OLR_train.columns = OLR_zrg_ppedataset.columns.astype(str)

vars_train_list = [PCP_train, TLWP_train, OSR_train, OLR_train]

PCP_test = PCP_zrg_ppedataset.loc[test_run_labels] #**(1/8)
TLWP_test = TLWP_zrg_ppedataset.loc[test_run_labels] #**(1/4)
OSR_test = OSR_zrg_ppedataset.loc[test_run_labels] #**(1/4)
OLR_test = OLR_zrg_ppedataset.loc[test_run_labels] #**(1/8)

PCP_test.columns = PCP_zrg_ppedataset.columns.astype(str)
TLWP_test.columns = TLWP_zrg_ppedataset.columns.astype(str)
OSR_test.columns = OSR_zrg_ppedataset.columns.astype(str)
OLR_test.columns = OLR_zrg_ppedataset.columns.astype(str)

vars_test_list = [PCP_test, TLWP_test, OSR_test, OLR_test]
# In[54]:


#for random forest, no preprocessing will be used
Y_train_ZRG = np.stack((PCP_train, TLWP_train, OSR_train, OLR_train), axis = 0)
Y_train_ZRG = np.transpose(Y_train_ZRG, (1, 2, 0))
Y_train_ZRG.shape


# In[55]:


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

#Using full bounds of parameters
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

param_bounds2 = np.array([range_thl2tune,
                   range_qw2tune,
                   range_length_fac,
                   range_c_diag_3rd_mom, 
                   range_Ckh,
                   range_Ckm,
                   range_lambda_low,
                   range_lambda_high,
                   range_spa_to_nc, 
                   range_p3_eci,
                   range_p3_eri,
                   range_k_acc,
                   range_p3_dep_nucleation_exponent,
                   range_max_total_ni,
                   range_ice_sed_knob,
                   range_p3_d_breakup_cutoff])
# In[56]:


#Create bounds for minmax scaler -- using full bounds of parameters
param_bounds = np.array([dict_range_pars[param] for param in ppe_params.columns])


# In[57]:


#transform data
X_pipe_sk_minmax = preprocessing.MinMaxScaler()
X_pipe_sk_minmax.fit(param_bounds.T)  # fit on lower and upper bounds (2,16)
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


# #### Employing models

# In[58]:


print(X_train_norm.shape, Y_train_norm.shape)


# ##### Model

# In[59]:


model_gp = gp_model(X_train_norm, Y_train_norm)
#model_cnn = cnn_model(X_train_norm, Y_train_norm)
#model_rf = rf_model(X_train.to_numpy(), Y_train_ZRG) #not preprocessed


# In[60]:


model_gp.train()


# #### Saving transforms

# In[61]:


type(Y_train_norm)


# In[62]:


### Saving projections both normed and not and R2s
timestamp_day = datetime.now().strftime("%Y-%m-%d")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#path - /global/cfs/cdirs/e3sm/jpaige3/ESEm/TF_saving/ --- REPLACE THIS PATH TO SAVE THE PIPELINE WITH THE MODEL
GP_proj_filename = f"/global/cfs/cdirs/e3sm/jpaige3/ESEm/GP_Saved_Model_Data/GP_ZRG_masked_proj_{timestamp}.pkl"

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
    'Y_train_norm': Y_train_norm,
    'PCP_train_norm': PCP_train_norm, 
    'TLWP_train_norm': TLWP_train_norm, 
    'OSR_train_norm': OSR_train_norm, 
    'OLR_train_norm':OLR_train_norm,
    ### unnormalized/untransformed
    'X_train': X_train, #ppe_params #.to_numpy()
    'Y_train': Y_train_ZRG,
    'PCP_train': PCP_train, 
    'TLWP_train': TLWP_train, 
    'OSR_train': OSR_train, 
    'OLR_train': OLR_train,
}

#save using pickle
with open(GP_proj_filename, 'wb') as f:
    pickle.dump(GP_data_to_save, f)


# In[63]:


### Saving obs
#path - /global/cfs/cdirs/e3sm/jpaige3/ESEm/GP_Saved_Model_Data/ --- REPLACE THIS PATH TO SAVE THE PIPELINE WITH THE MODEL
obs_save_filename = f"/global/cfs/cdirs/e3sm/jpaige3/ESEm/GP_Saved_Model_Data/obs_{timestamp}.pkl"

#make dictionary to save
obs_data_to_save = {
    'zrg_obs': zrg_obs, 
    'PCP_zrg_obs': PCP_zrg_obs, 
    'TLWP_zrg_obs': TLWP_zrg_obs, 
    'OSR_zrg_obs': OSR_zrg_obs, 
    'OLR_zrg_obs': OLR_zrg_obs,
}

#save using pickle
with open(obs_save_filename, 'wb') as f:
    pickle.dump(obs_data_to_save, f)


# ### In sample $R^2$

# In[64]:


m_gp, v_gp = model_gp.predict(X_train_norm)


# In[65]:


#vars_list = ['PCP','TLWP', 'OSR', 'OLR']
PCP_proj_norm_gp = pd.DataFrame(m_gp[:, :, 0], index = X_train.index)
TLWP_proj_norm_gp = pd.DataFrame(m_gp[:, :, 1], index = X_train.index)
OSR_proj_norm_gp = pd.DataFrame(m_gp[:, :, 2], index = X_train.index)
OLR_proj_norm_gp = pd.DataFrame(m_gp[:, :, 3], index = X_train.index)

PCP_v_proj_norm_gp = pd.DataFrame(v_gp[:, :, 0], index = X_train.index)
TLWP_v_proj_norm_gp = pd.DataFrame(v_gp[:, :, 1], index = X_train.index)
OSR_v_proj_norm_gp = pd.DataFrame(v_gp[:, :, 2], index = X_train.index)
OLR_v_proj_norm_gp = pd.DataFrame(v_gp[:, :, 3], index = X_train.index)


# In[66]:


#Normalied space
#Variance weighted -- lots of flexibility in the definition of this
PCP_r_squared_vw = r2_score(PCP_train_norm, PCP_proj_norm_gp, multioutput='variance_weighted')
PCP_rmse = root_mean_squared_error(PCP_train_norm, PCP_proj_norm_gp)

TLWP_r_squared_vw = r2_score(TLWP_train_norm, TLWP_proj_norm_gp, multioutput='variance_weighted')
TLWP_rmse = root_mean_squared_error(TLWP_train_norm, TLWP_proj_norm_gp)

OSR_r_squared_vw = r2_score(OSR_train_norm, OSR_proj_norm_gp, multioutput='variance_weighted')
OSR_rmse = root_mean_squared_error(OSR_train_norm, OSR_proj_norm_gp)

OLR_r_squared_vw = r2_score(OLR_train_norm, OLR_proj_norm_gp, multioutput='variance_weighted')
OLR_rmse = root_mean_squared_error(OLR_train_norm, OLR_proj_norm_gp)

print('PCP:', PCP_r_squared_vw, 'TLWP:', TLWP_r_squared_vw, 'OSR:', OSR_r_squared_vw, 'OLR:', OLR_r_squared_vw)
print('PCP:', PCP_rmse, 'TLWP:', TLWP_rmse, 'OSR:', OSR_rmse, 'OLR:', OLR_rmse)


# In[67]:


#Non-normalized space
PCP_proj_gp = pd.DataFrame(Y_pipe_sk_ss_PCP.inverse_transform(PCP_proj_norm_gp))
PCP_v_proj_gp = pd.DataFrame(Y_pipe_sk_ss_PCP.inverse_transform(PCP_v_proj_norm_gp))

TLWP_proj_gp = pd.DataFrame(Y_pipe_sk_ss_TLWP.inverse_transform(TLWP_proj_norm_gp))
TLWP_v_proj_gp = pd.DataFrame(Y_pipe_sk_ss_TLWP.inverse_transform(TLWP_v_proj_norm_gp))

OSR_proj_gp = pd.DataFrame(Y_pipe_sk_ss_OSR.inverse_transform(OSR_proj_norm_gp))
OSR_v_proj_gp = pd.DataFrame(Y_pipe_sk_ss_OSR.inverse_transform(OSR_v_proj_norm_gp))

OLR_proj_gp = pd.DataFrame(Y_pipe_sk_ss_OLR.inverse_transform(OLR_proj_norm_gp))
OLR_v_proj_gp = pd.DataFrame(Y_pipe_sk_ss_OLR.inverse_transform(OLR_v_proj_norm_gp))

#NOTE (unchecked Claude):  inverse_transform on a StandardScaler performs X * std + mean, which is the correct inverse for a mean but not the correct inverse for a variance. 
#For variances, you need to multiply by std**2 (i.e., variance_physical = variance_normalized * scaler.scale_**2). 
#So inverse_transform would give the wrong result for variance.


# In[68]:


PCP_r_squared_vw_phys = r2_score(PCP_train, PCP_proj_gp, multioutput='variance_weighted') # = 'raw_values')
PCP_rmse_phys = root_mean_squared_error(PCP_train, PCP_proj_gp)

TLWP_r_squared_vw_phys = r2_score(TLWP_train, TLWP_proj_gp, multioutput='variance_weighted')
TLWP_rmse_phys = root_mean_squared_error(TLWP_train, TLWP_proj_gp)

OSR_r_squared_vw_phys = r2_score(OSR_train, OSR_proj_gp, multioutput='variance_weighted')
OSR_rmse_phys = root_mean_squared_error(OSR_train, OSR_proj_gp)

OLR_r_squared_vw_phys = r2_score(OLR_train, OLR_proj_gp, multioutput='variance_weighted')
OLR_rmse_phys = root_mean_squared_error(OLR_train, OLR_proj_gp)

print('PCP:', PCP_r_squared_vw_phys, 'TLWP:', TLWP_r_squared_vw_phys, 'OSR:', OSR_r_squared_vw_phys, 'OLR:', OLR_r_squared_vw_phys)
print('PCP:', PCP_rmse_phys, 'TLWP:', TLWP_rmse_phys, 'OSR:', OSR_rmse_phys, 'OLR:', OLR_rmse_phys)

# Saving R2 variance weighted
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
r2output_filename = f"R2_output_varweight_insample/GP_ZRG_r2output_k={k}_{timestamp}.txt" - in sample

with open(r2output_filename, "w") as file:
    print('GP regionally trained, summer and winter, in variable units (untransformed), variance weighted', file=file)
    print('r squared unflat:','PCP:', PCP_r_squared_unflat, 'TLWP:', TLWP_r_squared_unflat, 'OSR:', OSR_r_squared_unflat, 'OLR:', OLR_r_squared_unflat, file=file)
    print('     rmse unflat:','PCP:', PCP_rmse_squared_unflat, 'TLWP:', TLWP_rmse_squared_unflat, 'OSR:', OSR_rmse_squared_unflat, 'OLR:', OLR_rmse_squared_unflat, file=file)
    
# #### Load back in
# Load back
GP_proj_filename = "/global/cfs/cdirs/e3sm/jpaige3/ESEm/TF_saving/GPFlow_save2025-06-12/GP_ZRG_proj_2025-06-12_14-31-26.pkl"

with open(GP_proj_filename, 'rb') as f:
    loaded = pickle.load(f)

# Access data
X_pipe_sk_minmax = loaded['X_pipeline']