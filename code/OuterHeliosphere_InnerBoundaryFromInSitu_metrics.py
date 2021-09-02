# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 08:56:33 2021

@author: mathewjowens
"""

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import pandas as pd
import os
import h5py

import helio_time as htime
import helio_coords as hcoord

# <codecell> input params

outputdir = os.environ['DBOX'] + '\\Apps\\Overleaf\\Solar Wind Inputs\\'
datadir = os.environ['DBOX'] + 'Papers_WIP\\_coauthor\\JonnyNichols\\'
nphi = 16 #number of longitudes from Earth to consider

dv_thres = -10 #speed gradient to identify a HSS
dt = 3 #number of days to classify as a single HSS, and to find max V

# <codecell> Functions and constants
def zerototwopi(angles):
    """
    Function to constrain angles to the 0 - 2pi domain.

    :param angles: a numpy array of angles
    :return: a numpy array of angles
    """
    twopi = 2.0 * np.pi
    angles_out = angles
    a = -np.floor_divide(angles_out, twopi)
    angles_out = angles_out + (a * twopi)
    return angles_out

daysec = 24 * 60 * 60 * u.s
kms = u.km / u.s
synodic_period = 27.2753 * daysec  # Solar Synodic rotation period from Earth.
sidereal_period = 25.38 *daysec 
omega_sidereal = 2*np.pi *u.rad / sidereal_period
omega_synodic = 2*np.pi *u.rad / synodic_period
domega = omega_synodic - omega_sidereal

# <codecell> Data load
    
#load the vinput reconstructions
filepath = outputdir + 'vinputs_TrueStateHGI.h5'
h5f = h5py.File(filepath,'r')
vgrid_HGI_tlin = np.array(h5f['vgrid_HGI_tlin'])
vgrid_HGI_tsin = np.array(h5f['vgrid_HGI_tsin'])
time_edges = np.array(h5f['time_edges'])
lon_edges = np.array(h5f['lon_edges'])
time_grid = np.array(h5f['time_grid'])
lon_grid = np.array(h5f['lon_grid'])
h5f.close()

filepath = outputdir + 'vinputs_corot_HGI.h5'
h5f = h5py.File(filepath,'r')
vgrid_HGI_recon_back_tlin = np.array(h5f['vgrid_HGI_recon_back_tlin'])
vgrid_HGI_recon_back_tsin = np.array(h5f['vgrid_HGI_recon_back_tsin'])
vgrid_HGI_recon_both_tlin = np.array(h5f['vgrid_HGI_recon_both_tlin'])
vgrid_HGI_recon_both_tsin = np.array(h5f['vgrid_HGI_recon_both_tsin'])
h5f.close()

filepath = outputdir + 'vinputs_dtw_HGI.h5'
h5f = h5py.File(filepath,'r')
vgrid_HGI_recon_dtw_tlin = np.array(h5f['vgrid_HGI_recon_dtw_tlin'])
vgrid_HGI_recon_dtw_tsin = np.array(h5f['vgrid_HGI_recon_dtw_tsin'])
h5f.close()

#remove data in the spin-up and spin-down period
smjd = time_grid[0] + 29
fmjd = time_grid[-1] - 29
mask = ((time_grid > smjd) & (time_grid < fmjd))
time_grid = time_grid[mask]
vgrid_HGI_tlin = vgrid_HGI_tlin[:,mask]
vgrid_HGI_tsin = vgrid_HGI_tsin[:,mask]
vgrid_HGI_recon_back_tlin = vgrid_HGI_recon_back_tlin[:,mask]
vgrid_HGI_recon_both_tlin = vgrid_HGI_recon_both_tlin[:,mask]
vgrid_HGI_recon_dtw_tlin = vgrid_HGI_recon_dtw_tlin[:,mask]
vgrid_HGI_recon_back_tsin = vgrid_HGI_recon_back_tsin[:,mask]
vgrid_HGI_recon_both_tsin = vgrid_HGI_recon_both_tsin[:,mask]
vgrid_HGI_recon_dtw_tsin = vgrid_HGI_recon_dtw_tsin[:,mask]
mask = ((time_edges >= smjd) & (time_edges <= fmjd))
time_edges = time_edges[mask]

#Load Earth ephemeris
filepath = datadir + 'Earth_HGI.lst'
pos_Earth = pd.read_csv(filepath,
                     skiprows = 1, delim_whitespace=True,
                     names=['year','doy',
                            'rad_au','HGI_lat','HGI_lon'])
#convert to mjd
pos_Earth['mjd'] = htime.doyyr2mjd(pos_Earth['doy'],pos_Earth['year'])


#create OMNI position files
starttime = htime.mjd2datetime(time_edges[0]).item()
endtime = htime.mjd2datetime(time_edges[-1]).item()

OMNI = pd.DataFrame(index=pd.date_range(starttime, endtime, freq='H'))
OMNI['mjd'] = htime.datetime2mjd(OMNI.index).to_numpy()*u.day

OMNI['r_au'] = np.interp(OMNI['mjd'].to_numpy(),
                              pos_Earth['mjd'].to_numpy(),pos_Earth['rad_au'].to_numpy()) * u.au
OMNI['HGI_lat'] = np.interp(OMNI['mjd'].to_numpy(),
                              pos_Earth['mjd'].to_numpy(),pos_Earth['HGI_lat'].to_numpy()) * u.deg

#for longitude, unwrap to avoid the 0/2*pi issue
pos_Earth_unwrap = np.unwrap(pos_Earth['HGI_lon'].to_numpy()*np.pi/180)
OMNI['HGI_lon'] = np.interp(OMNI['mjd'].to_numpy(),
                              pos_Earth['mjd'].to_numpy(), pos_Earth_unwrap) 
OMNI['HGI_lon'] = zerototwopi(OMNI['HGI_lon'])

#compute the Carrington longitude of Earth
temp = hcoord.carringtonlatlong_earth(OMNI['mjd'])
OMNI['Carr_lon'] = temp[:,1]

#compute the synodic period for OMNI/Earth
OMNI_lon_unwrap = np.unwrap(OMNI['HGI_lon'].to_numpy())
smjd=OMNI['mjd'][0] 
fmjd=OMNI['mjd'][-1] 
omega_omni = omega_sidereal - (OMNI_lon_unwrap[-1] - OMNI_lon_unwrap[0])*u.rad/(fmjd-smjd)/daysec


# <codecell> Generate time series
dphi = 2*np.pi/nphi
dphis = np.arange(dphi/2, 2*np.pi +0.001 - dphi/2, dphi)


nt = len(time_grid)
#extract time series at given longitude relative to Earth
v_lin_dphi_true = np.ones((nt,nphi))*np.nan
v_lin_dphi_back = np.ones((nt,nphi))*np.nan
v_lin_dphi_both = np.ones((nt,nphi))*np.nan
v_lin_dphi_dtw = np.ones((nt,nphi))*np.nan

v_sin_dphi_true = np.ones((nt,nphi))*np.nan
v_sin_dphi_back = np.ones((nt,nphi))*np.nan
v_sin_dphi_both = np.ones((nt,nphi))*np.nan
v_sin_dphi_dtw = np.ones((nt,nphi))*np.nan

for n in range(0,nphi):
    thisphi = dphis[n]
    for t in range(0, len(time_grid)):
        #compute the HGI longitude of interest
        thist = time_grid[t]
        t_ephem = np.argmin(np.abs(OMNI['mjd']-thist))
        
        #linear model
        E_HGI = OMNI['HGI_lon'][t_ephem]
        this_HGI = zerototwopi(E_HGI + thisphi)
        
        #find the solar wind speed at this time/longitude
        vHGI = vgrid_HGI_tlin[:,t]
        this_v = np.interp(this_HGI, lon_grid, vHGI)
        v_lin_dphi_true[t,n] = this_v
        
        vHGI = vgrid_HGI_recon_back_tlin[:,t]
        this_v = np.interp(this_HGI, lon_grid, vHGI)
        v_lin_dphi_back[t,n] = this_v
        
        vHGI = vgrid_HGI_recon_both_tlin[:,t]
        this_v = np.interp(this_HGI, lon_grid, vHGI)
        v_lin_dphi_both[t,n] = this_v
        
        vHGI = vgrid_HGI_recon_dtw_tlin[:,t]
        this_v = np.interp(this_HGI, lon_grid, vHGI)
        v_lin_dphi_dtw[t,n] = this_v
        
        #nonlinear model      
        #find the solar wind speed at this time/longitude
        vHGI = vgrid_HGI_tsin[:,t]
        this_v = np.interp(this_HGI, lon_grid, vHGI)
        v_sin_dphi_true[t,n] = this_v
        
        vHGI = vgrid_HGI_recon_back_tsin[:,t]
        this_v = np.interp(this_HGI, lon_grid, vHGI)
        v_sin_dphi_back[t,n] = this_v
        
        vHGI = vgrid_HGI_recon_both_tsin[:,t]
        this_v = np.interp(this_HGI, lon_grid, vHGI)
        v_sin_dphi_both[t,n] = this_v
        
        vHGI = vgrid_HGI_recon_dtw_tsin[:,t]
        this_v = np.interp(this_HGI, lon_grid, vHGI)
        v_sin_dphi_dtw[t,n] = this_v

# <codecell> Analysis of time series
v_lin_back_mae = np.ones((nphi))*np.nan   
v_lin_both_mae = np.ones((nphi))*np.nan      
v_lin_dtw_mae = np.ones((nphi))*np.nan

v_sin_back_mae = np.ones((nphi))*np.nan   
v_sin_both_mae = np.ones((nphi))*np.nan      
v_sin_dtw_mae = np.ones((nphi))*np.nan

v_lin_back_dt = np.ones((nphi))*np.nan   
v_lin_both_dt = np.ones((nphi))*np.nan      
v_lin_dtw_dt = np.ones((nphi))*np.nan  
v_lin_back_absdt = np.ones((nphi))*np.nan   
v_lin_both_absdt = np.ones((nphi))*np.nan      
v_lin_dtw_absdt = np.ones((nphi))*np.nan 

v_lin_back_dv = np.ones((nphi))*np.nan   
v_lin_both_dv = np.ones((nphi))*np.nan      
v_lin_dtw_dv = np.ones((nphi))*np.nan  
v_lin_back_absdv = np.ones((nphi))*np.nan   
v_lin_both_absdv = np.ones((nphi))*np.nan      
v_lin_dtw_absdv = np.ones((nphi))*np.nan

v_sin_back_dt = np.ones((nphi))*np.nan   
v_sin_both_dt = np.ones((nphi))*np.nan      
v_sin_dtw_dt = np.ones((nphi))*np.nan    
v_sin_back_absdt = np.ones((nphi))*np.nan   
v_sin_both_absdt = np.ones((nphi))*np.nan      
v_sin_dtw_absdt = np.ones((nphi))*np.nan 

v_sin_back_dv = np.ones((nphi))*np.nan   
v_sin_both_dv = np.ones((nphi))*np.nan      
v_sin_dtw_dv = np.ones((nphi))*np.nan  
v_sin_back_absdv = np.ones((nphi))*np.nan   
v_sin_both_absdv = np.ones((nphi))*np.nan      
v_sin_dtw_absdv = np.ones((nphi))*np.nan   



for n in range(0,nphi):
    v_lin_back_mae[n] = np.nanmean(np.abs(v_lin_dphi_back[:,n] - v_lin_dphi_true[:,n]))
    v_lin_both_mae[n] = np.nanmean(np.abs(v_lin_dphi_both[:,n] - v_lin_dphi_true[:,n]))
    v_lin_dtw_mae[n] = np.nanmean(np.abs(v_lin_dphi_dtw[:,n] - v_lin_dphi_true[:,n]))
    
    v_sin_back_mae[n] = np.nanmean(np.abs(v_sin_dphi_back[:,n] - v_sin_dphi_true[:,n]))
    v_sin_both_mae[n] = np.nanmean(np.abs(v_sin_dphi_both[:,n] - v_sin_dphi_true[:,n]))
    v_sin_dtw_mae[n] = np.nanmean(np.abs(v_sin_dphi_dtw[:,n] - v_sin_dphi_true[:,n]))
    
    #find the fast-wind interfaces
    dv_lin_true = np.append(v_lin_dphi_true[:-1,n] -  v_lin_dphi_true[1:,n], 0)
    dv_sin_true = np.append(v_sin_dphi_true[:-1,n] -  v_sin_dphi_true[1:,n], 0)
    dv_lin_back = np.append(v_lin_dphi_back[:-1,n] -  v_lin_dphi_back[1:,n], 0)
    dv_sin_back = np.append(v_sin_dphi_back[:-1,n] -  v_sin_dphi_back[1:,n], 0)
    dv_lin_both = np.append(v_lin_dphi_both[:-1,n] -  v_lin_dphi_both[1:,n], 0)
    dv_sin_both = np.append(v_sin_dphi_both[:-1,n] -  v_sin_dphi_both[1:,n], 0)
    dv_lin_dtw = np.append(v_lin_dphi_dtw[:-1,n] -  v_lin_dphi_dtw[1:,n], 0)
    dv_sin_dtw = np.append(v_sin_dphi_dtw[:-1,n] -  v_sin_dphi_dtw[1:,n], 0)
    
    #get times of high speed gradients
    t_hss_lin_true = time_grid[dv_lin_true < dv_thres]
    t_hss_sin_true = time_grid[dv_sin_true < dv_thres]
    t_hss_lin_back = time_grid[dv_lin_back < dv_thres]
    t_hss_sin_back = time_grid[dv_sin_back < dv_thres]
    t_hss_lin_both = time_grid[dv_lin_both < dv_thres]
    t_hss_sin_both = time_grid[dv_sin_both < dv_thres]
    t_hss_lin_dtw = time_grid[dv_lin_dtw < dv_thres]
    t_hss_sin_dtw = time_grid[dv_sin_dtw < dv_thres]
    
    #check that the identified HSSs are at least dt days apart
    i=0; exitflag = False
    while exitflag == False:
        if t_hss_lin_true[i+1] - t_hss_lin_true[i] < dt:
            t_hss_lin_true = np.delete(t_hss_lin_true,i+1)
        else:
            i=i+1 
        if i == len(t_hss_lin_true) -1:
            exitflag = True
    i=0; exitflag = False
    while exitflag == False:
        if t_hss_sin_true[i+1] - t_hss_sin_true[i] < dt:
            t_hss_sin_true = np.delete(t_hss_sin_true,i+1)
        else:
            i=i+1 
        if i == len(t_hss_sin_true) -1:
            exitflag = True
    i=0; exitflag = False
    while exitflag == False:
        if t_hss_lin_back[i+1] - t_hss_lin_back[i] < dt:
            t_hss_lin_back = np.delete(t_hss_lin_back,i+1)
        else:
            i=i+1 
        if i == len(t_hss_lin_back) -1:
            exitflag = True
    i=0; exitflag = False
    while exitflag == False:
        if t_hss_sin_back[i+1] - t_hss_sin_back[i] < dt:
            t_hss_sin_back = np.delete(t_hss_sin_back,i+1)
        else:
            i=i+1 
        if i == len(t_hss_sin_back) -1:
            exitflag = True
    i=0; exitflag = False
    while exitflag == False:
        if t_hss_lin_both[i+1] - t_hss_lin_both[i] < dt:
            t_hss_lin_both = np.delete(t_hss_lin_both,i+1)
        else:
            i=i+1 
        if i == len(t_hss_lin_both) -1:
            exitflag = True
    i=0; exitflag = False
    while exitflag == False:
        if t_hss_sin_both[i+1] - t_hss_sin_both[i] < dt:
            t_hss_sin_both = np.delete(t_hss_sin_both,i+1)
        else:
            i=i+1 
        if i == len(t_hss_sin_both) -1:
            exitflag = True
    i=0; exitflag = False
    while exitflag == False:
        if t_hss_lin_dtw[i+1] - t_hss_lin_dtw[i] < dt:
            t_hss_lin_dtw = np.delete(t_hss_lin_dtw,i+1)
        else:
            i=i+1 
        if i == len(t_hss_lin_dtw) -1:
            exitflag = True
    i=0; exitflag = False
    while exitflag == False:
        if t_hss_sin_dtw[i+1] - t_hss_sin_dtw[i] < dt:
            t_hss_sin_dtw = np.delete(t_hss_sin_dtw,i+1)
        else:
            i=i+1 
        if i == len(t_hss_sin_dtw) -1:
            exitflag = True
    
    #also require the observed HSS to be more than dt from the start/end
    if t_hss_lin_true[0] < time_grid[0] + dt:
        t_hss_lin_true = np.delete(t_hss_lin_true,0)
    if t_hss_sin_true[-1] > time_grid[-1] - dt:
        t_hss_sin_true = np.delete(t_hss_sin_true,-1)
            
    #find the max speed (within a window dt) associated with each HSS
    vmax_lin_true = np.ones(len(t_hss_lin_true))*np.nan
    for i in range(0,len(t_hss_lin_true)):
        mask = ((time_grid >= t_hss_lin_true[i] - dt) &
                (time_grid <= t_hss_lin_true[i] + dt)) 
        vmax_lin_true[i] = np.nanmax(v_lin_dphi_true[mask,n])
    vmax_sin_true = np.ones(len(t_hss_sin_true))*np.nan
    for i in range(0,len(t_hss_sin_true)):
        mask = ((time_grid >= t_hss_sin_true[i] - dt) &
                (time_grid <= t_hss_sin_true[i] + dt)) 
        vmax_sin_true[i] = np.nanmax(v_sin_dphi_true[mask,n])
    vmax_lin_back = np.ones(len(t_hss_lin_back))*np.nan
    for i in range(0,len(t_hss_lin_back)):
        mask = ((time_grid >= t_hss_lin_back[i] - dt) &
                (time_grid <= t_hss_lin_back[i] + dt)) 
        vmax_lin_back[i] = np.nanmax(v_lin_dphi_back[mask,n])
    vmax_sin_back = np.ones(len(t_hss_sin_back))*np.nan
    for i in range(0,len(t_hss_sin_back)):
        mask = ((time_grid >= t_hss_sin_back[i] - dt) &
                (time_grid <= t_hss_sin_back[i] + dt)) 
        vmax_sin_back[i] = np.nanmax(v_sin_dphi_back[mask,n])
    vmax_lin_both = np.ones(len(t_hss_lin_both))*np.nan
    for i in range(0,len(t_hss_lin_both)):
        mask = ((time_grid >= t_hss_lin_both[i] - dt) &
                (time_grid <= t_hss_lin_both[i] + dt)) 
        vmax_lin_both[i] = np.nanmax(v_lin_dphi_both[mask,n])
    vmax_sin_both = np.ones(len(t_hss_sin_both))*np.nan
    for i in range(0,len(t_hss_sin_both)):
        mask = ((time_grid >= t_hss_sin_both[i] - dt) &
                (time_grid <= t_hss_sin_both[i] + dt)) 
        vmax_sin_both[i] = np.nanmax(v_sin_dphi_both[mask,n])
    vmax_lin_dtw = np.ones(len(t_hss_lin_dtw))*np.nan
    for i in range(0,len(t_hss_lin_dtw)):
        mask = ((time_grid >= t_hss_lin_dtw[i] - dt) &
                (time_grid <= t_hss_lin_dtw[i] + dt)) 
        vmax_lin_dtw[i] = np.nanmax(v_lin_dphi_dtw[mask,n])
    vmax_sin_dtw = np.ones(len(t_hss_sin_dtw))*np.nan
    for i in range(0,len(t_hss_sin_dtw)):
        mask = ((time_grid >= t_hss_sin_dtw[i] - dt) &
                (time_grid <= t_hss_sin_dtw[i] + dt)) 
        vmax_sin_dtw[i] = np.nanmax(v_sin_dphi_dtw[mask,n]) 
        
    #find the HSS timing errors
    dt_back_lin = np.ones(len(t_hss_lin_true))*np.nan
    dt_both_lin = np.ones(len(t_hss_lin_true))*np.nan
    dt_dtw_lin = np.ones(len(t_hss_lin_true))*np.nan
    dt_back_sin = np.ones(len(t_hss_sin_true))*np.nan
    dt_both_sin = np.ones(len(t_hss_sin_true))*np.nan
    dt_dtw_sin = np.ones(len(t_hss_sin_true))*np.nan
    dv_back_lin = np.ones(len(t_hss_lin_true))*np.nan
    dv_both_lin = np.ones(len(t_hss_lin_true))*np.nan
    dv_dtw_lin = np.ones(len(t_hss_lin_true))*np.nan
    dv_back_sin = np.ones(len(t_hss_sin_true))*np.nan
    dv_both_sin = np.ones(len(t_hss_sin_true))*np.nan
    dv_dtw_sin = np.ones(len(t_hss_sin_true))*np.nan
    for nstream in range(0,len(t_hss_lin_true)):
        #find the closest HSS in time
        nclosest = np.argmin(np.abs(t_hss_lin_true[nstream] - t_hss_lin_back))
        dt_back_lin[nstream] = t_hss_lin_true[nstream] - t_hss_lin_back[nclosest]
        dv_back_lin[nstream] = vmax_lin_true[nstream] - vmax_lin_back[nclosest]
        
        nclosest = np.argmin(np.abs(t_hss_lin_true[nstream] - t_hss_lin_both))
        dt_both_lin[nstream] = t_hss_lin_true[nstream] - t_hss_lin_both[nclosest]
        dv_both_lin[nstream] = vmax_lin_true[nstream] - vmax_lin_both[nclosest]
        
        nclosest = np.argmin(np.abs(t_hss_lin_true[nstream] - t_hss_lin_dtw))
        dt_dtw_lin[nstream] = t_hss_lin_true[nstream] - t_hss_lin_dtw[nclosest]
        dv_dtw_lin[nstream] = vmax_lin_true[nstream] - vmax_lin_dtw[nclosest]
        
    for nstream in range(0,len(t_hss_sin_true)):    
        nclosest = np.argmin(np.abs(t_hss_sin_true[nstream] - t_hss_sin_back))
        dt_back_sin[nstream] = t_hss_sin_true[nstream] - t_hss_sin_back[nclosest]
        dv_back_sin[nstream] = vmax_sin_true[nstream] - vmax_sin_back[nclosest]
        
        nclosest = np.argmin(np.abs(t_hss_sin_true[nstream] - t_hss_sin_both))
        dt_both_sin[nstream] = t_hss_sin_true[nstream] - t_hss_sin_both[nclosest]
        dv_both_sin[nstream] = vmax_sin_true[nstream] - vmax_sin_both[nclosest]
        
        nclosest = np.argmin(np.abs(t_hss_sin_true[nstream] - t_hss_sin_dtw))
        dt_dtw_sin[nstream] = t_hss_sin_true[nstream] - t_hss_sin_dtw[nclosest]
        dv_dtw_sin[nstream] = vmax_sin_true[nstream] - vmax_sin_dtw[nclosest]
    
    #find the mean dt
    v_lin_back_dt[n] = np.nanmean(dt_back_lin)
    v_lin_back_absdt[n] = np.nanmean(np.abs(dt_back_lin))
    v_lin_both_dt[n] = np.nanmean(dt_both_lin)
    v_lin_both_absdt[n] = np.nanmean(np.abs(dt_both_lin))
    v_lin_dtw_dt[n] = np.nanmean(dt_dtw_lin)
    v_lin_dtw_absdt[n] = np.nanmean(np.abs(dt_dtw_lin))
    
    v_sin_back_dt[n] = np.nanmean(dt_back_sin)
    v_sin_back_absdt[n] = np.nanmean(np.abs(dt_back_sin))
    v_sin_both_dt[n] = np.nanmean(dt_both_sin)
    v_sin_both_absdt[n] = np.nanmean(np.abs(dt_both_sin))
    v_sin_dtw_dt[n] = np.nanmean(dt_dtw_sin)
    v_sin_dtw_absdt[n] = np.nanmean(np.abs(dt_dtw_sin))
    
    v_lin_back_dv[n] = np.nanmean(dv_back_lin)
    v_lin_back_absdv[n] = np.nanmean(np.abs(dv_back_lin))
    v_lin_both_dv[n] = np.nanmean(dv_both_lin)
    v_lin_both_absdv[n] = np.nanmean(np.abs(dv_both_lin))
    v_lin_dtw_dv[n] = np.nanmean(dv_dtw_lin)
    v_lin_dtw_absdv[n] = np.nanmean(np.abs(dv_dtw_lin))
    v_sin_back_dv[n] = np.nanmean(dv_back_sin)
    v_sin_back_absdv[n] = np.nanmean(np.abs(dv_back_sin))
    v_sin_both_dv[n] = np.nanmean(dv_both_sin)
    v_sin_both_absdv[n] = np.nanmean(np.abs(dv_both_sin))
    v_sin_dtw_dv[n] = np.nanmean(dv_dtw_sin)
    v_sin_dtw_absdv[n] = np.nanmean(np.abs(dv_dtw_sin))
        
        
    

    
#sanity check the HSS identification            
#plt.figure()
#plt.plot(time_grid,dv_sin_true)
#for i in range(0,len(t_hss_sin_true)):
#    plt.plot([t_hss_sin_true[i],t_hss_sin_true[i]], [-50, 0],'k')
#
#plt.figure()
#plt.plot(time_grid,dv_sin_dtw)
#for i in range(0,len(t_hss_sin_dtw)):
#    plt.plot([t_hss_sin_dtw[i],t_hss_sin_dtw[i]], [-50, 0],'k')    
    
    
    
    
plt.figure()
loc='upper right'


ax1=plt.subplot(321)
plt.plot(dphis*180/np.pi, v_lin_back_mae, 'g', label='Corotation (back in time)')
plt.plot(dphis*180/np.pi, v_lin_both_mae, 'b', label='Corotation (smooth in time)')
plt.plot(dphis*180/np.pi, v_lin_dtw_mae, 'r', label='DTW')
ax1.set_ylim((0,70))
ax1.set_xlim((0,360))
ax1.set_xticks((0,90,180,270,360))
plt.ylabel('MAE [km/s]')
anchored_text = AnchoredText("(a)", loc=loc)
ax1.add_artist(anchored_text)
plt.title('Linearly-evolving V')
plt.legend(loc='upper left')
    
ax2=plt.subplot(323)
#plt.plot(dphis*180/np.pi, v_lin_back_dt, 'g--', label='Corotation (back in time)')
#plt.plot(dphis*180/np.pi, v_lin_both_dt, 'b--', label='Corotation (smooth in time)')
#plt.plot(dphis*180/np.pi, v_lin_dtw_dt, 'r--', label='DTW')

plt.plot(dphis*180/np.pi, v_lin_back_absdt, 'g', label='Corotation (back in time)')
plt.plot(dphis*180/np.pi, v_lin_both_absdt, 'b', label='Corotation (smooth in time)')
plt.plot(dphis*180/np.pi, v_lin_dtw_absdt, 'r', label='DTW')
ax2.set_ylim((0,2.6))
ax2.set_xlim((0,360))
ax2.set_xticks((0,90,180,270,360))
plt.ylabel(r'HSE $|\Delta$ T$|$ [days]')
anchored_text = AnchoredText("(b)", loc=loc)
ax2.add_artist(anchored_text)

ax3=plt.subplot(325)
#plt.plot(dphis*180/np.pi, v_lin_back_dt, 'g--', label='Corotation (back in time)')
#plt.plot(dphis*180/np.pi, v_lin_both_dt, 'b--', label='Corotation (smooth in time)')
#plt.plot(dphis*180/np.pi, v_lin_dtw_dt, 'r--', label='DTW')
plt.plot(dphis*180/np.pi, v_lin_back_absdv, 'g', label='Corotation (back in time)')
plt.plot(dphis*180/np.pi, v_lin_both_absdv, 'b', label='Corotation (smooth in time)')
plt.plot(dphis*180/np.pi, v_lin_dtw_absdv, 'r', label='DTW')
plt.ylabel(r'HSE $|\Delta$ V$|$ [km/s]')
ax3.set_ylim((0,80))
ax3.set_xlim((0,360))
ax3.set_xticks((0,90,180,270,360))
plt.xlabel('$\Delta \phi$ [deg]')
anchored_text = AnchoredText("(c)", loc=loc)
ax3.add_artist(anchored_text)

ax4=plt.subplot(322)
plt.plot(dphis*180/np.pi, v_sin_back_mae, 'g', label='Corotation (back in time)')
plt.plot(dphis*180/np.pi, v_sin_both_mae, 'b', label='Corotation (smooth in time)')
plt.plot(dphis*180/np.pi, v_sin_dtw_mae, 'r', label='DTW')
ax4.set_ylim((0,70))
ax4.set_xlim((0,360))
ax4.set_xticks((0,90,180,270,360))
anchored_text = AnchoredText("(d)", loc=loc)
ax4.add_artist(anchored_text)
plt.title('Nonlinearly-evolving V')
  
ax5=plt.subplot(324)
#plt.plot(dphis*180/np.pi, v_lin_back_dt, 'g--', label='Corotation (back in time)')
#plt.plot(dphis*180/np.pi, v_lin_both_dt, 'b--', label='Corotation (smooth in time)')
#plt.plot(dphis*180/np.pi, v_lin_dtw_dt, 'r--', label='DTW')
plt.plot(dphis*180/np.pi, v_sin_back_absdt, 'g', label='Corotation (back in time)')
plt.plot(dphis*180/np.pi, v_sin_both_absdt, 'b', label='Corotation (smooth in time)')
plt.plot(dphis*180/np.pi, v_sin_dtw_absdt, 'r', label='DTW')
ax5.set_xlim((0,360))
ax5.set_ylim((0,2.6))
ax5.set_xticks((0,90,180,270,360))
anchored_text = AnchoredText("(e)", loc=loc)
ax5.add_artist(anchored_text)

ax6=plt.subplot(326)
#plt.plot(dphis*180/np.pi, v_lin_back_dt, 'g--', label='Corotation (back in time)')
#plt.plot(dphis*180/np.pi, v_lin_both_dt, 'b--', label='Corotation (smooth in time)')
#plt.plot(dphis*180/np.pi, v_lin_dtw_dt, 'r--', label='DTW')
plt.plot(dphis*180/np.pi, v_sin_back_absdv, 'g', label='Corotation (back in time)')
plt.plot(dphis*180/np.pi, v_sin_both_absdv, 'b', label='Corotation (smooth in time)')
plt.plot(dphis*180/np.pi, v_sin_dtw_absdv, 'r', label='DTW')
ax6.set_ylim((0,80))
ax6.set_xlim((0,360))
ax6.set_xticks((0,90,180,270,360))
plt.xlabel('$\Delta \phi$ [deg]')
anchored_text = AnchoredText("(f)", loc=loc)
ax6.add_artist(anchored_text)