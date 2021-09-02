# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 15:14:09 2021

@author: mathewjowens
"""


import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import copy
import h5py

os.chdir(os.path.abspath(os.environ['DBOX'] + '\\python'))
import helio_time as htime

os.chdir(os.path.abspath(os.environ['DBOX'] + 'python_repos\\HUXt\\code'))
import huxt as H
import huxt_analysis as HA


#sim_duration = (time_grid[-1] - time_grid[0]) *u.day



datadir = os.environ['DBOX'] + '\\Apps\\Overleaf\\Solar Wind Inputs\\'


#load the vinput reconstructions
filepath = datadir + 'vinputs_TrueStateHGI.h5'
h5f = h5py.File(filepath,'r')
vgrid_HGI_tlin = np.array(h5f['vgrid_HGI_tlin'])
vgrid_HGI_tsin = np.array(h5f['vgrid_HGI_tsin'])
time_edges = np.array(h5f['time_edges'])
lon_edges = np.array(h5f['lon_edges'])
time_grid = np.array(h5f['time_grid'])
lon_grid = np.array(h5f['lon_grid'])
h5f.close()

filepath = datadir + 'vinputs_corot_HGI.h5'
h5f = h5py.File(filepath,'r')
vgrid_HGI_recon_back_tlin = np.array(h5f['vgrid_HGI_recon_back_tlin'])
vgrid_HGI_recon_back_tsin = np.array(h5f['vgrid_HGI_recon_back_tsin'])
vgrid_HGI_recon_both_tlin = np.array(h5f['vgrid_HGI_recon_both_tlin'])
vgrid_HGI_recon_both_tsin = np.array(h5f['vgrid_HGI_recon_both_tsin'])
h5f.close()

filepath = datadir + 'vinputs_dtw_HGI.h5'
h5f = h5py.File(filepath,'r')
vgrid_HGI_recon_dtw_tlin = np.array(h5f['vgrid_HGI_recon_dtw_tlin'])
vgrid_HGI_recon_dtw_tsin = np.array(h5f['vgrid_HGI_recon_dtw_tsin'])
h5f.close()

smjd = time_grid[0] + 29
sim_duration = time_grid[-1] - 29 - smjd
#sim_duration = 50

#plt.figure()
#plt.pcolor(vgrid_HGI_tlin)

#get the length of the run

cr = htime.mjd2crnum(smjd)

#create a dummy model istance, get the model time step
model = H.HUXt(v_boundary = np.ones(128) * 400*(u.km/u.s), cr_num = np.floor(cr),
               frame='sidereal', r_min = 215*u.solRad, r_max = 2165*u.solRad,
               simtime=sim_duration * u.day, dt_scale=50)
dt = model.dt

#set up the model time step
model_time=np.arange(0,model.simtime.value,dt.value) * u.s 
model_mjd = smjd + model_time.to(u.day).value

#loop through the longitude bins and interpolate the time series
input_ts_lin_true = np.ones((len(model_mjd),len(lon_grid)))*np.nan
input_ts_lin_back = np.ones((len(model_mjd),len(lon_grid)))*np.nan
input_ts_lin_both = np.ones((len(model_mjd),len(lon_grid)))*np.nan
input_ts_lin_dtw = np.ones((len(model_mjd),len(lon_grid)))*np.nan
input_ts_sin_true = np.ones((len(model_mjd),len(lon_grid)))*np.nan
input_ts_sin_back = np.ones((len(model_mjd),len(lon_grid)))*np.nan
input_ts_sin_both = np.ones((len(model_mjd),len(lon_grid)))*np.nan
input_ts_sin_dtw = np.ones((len(model_mjd),len(lon_grid)))*np.nan

for lon in range(0,len(lon_grid)):
    input_ts_lin_true[:,lon] = np.interp(model_mjd,time_grid,vgrid_HGI_tlin[lon,:])
    input_ts_lin_back[:,lon] = np.interp(model_mjd,time_grid,vgrid_HGI_recon_back_tlin[lon,:])
    input_ts_lin_both[:,lon] = np.interp(model_mjd,time_grid,vgrid_HGI_recon_both_tlin[lon,:])
    input_ts_lin_dtw[:,lon] = np.interp(model_mjd,time_grid,vgrid_HGI_recon_dtw_tlin[lon,:])
    
    input_ts_sin_true[:,lon] = np.interp(model_mjd,time_grid,vgrid_HGI_tsin[lon,:])
    input_ts_sin_back[:,lon] = np.interp(model_mjd,time_grid,vgrid_HGI_recon_back_tsin[lon,:])
    input_ts_sin_both[:,lon] = np.interp(model_mjd,time_grid,vgrid_HGI_recon_both_tsin[lon,:])
    input_ts_sin_dtw[:,lon] = np.interp(model_mjd,time_grid,vgrid_HGI_recon_dtw_tsin[lon,:])
    
#plt.figure()
#plt.pcolor(input_ts_lin_dtw)

# <codecell> run the models

    
#add the time sereis to the model and run
model_true =  copy.deepcopy(model) 
model_true.model_time = model_time
model_true.input_v_ts = input_ts_lin_true
model_true.solve([])

model_back = copy.deepcopy(model)  
model_back.model_time = model_time
model_back.input_v_ts = input_ts_lin_back
model_back.solve([])

model_both = copy.deepcopy(model)  
model_both.model_time = model_time
model_both.input_v_ts = input_ts_lin_both
model_both.solve([])

model_dtw = copy.deepcopy(model)  
model_dtw.model_time = model_time
model_dtw.input_v_ts = input_ts_lin_dtw
model_dtw.solve([])


# <codecell> Loop through each radial distance and compute the MAE
t_start=35 *u.day
t_end = model.time_out[-1]
id_tstart = np.argmin(np.abs(model.time_out - t_start))
id_tend = np.argmin(np.abs(model.time_out - t_end))

for r in range(0, len(model.v_grid)):
    
    
    
    
    
    
    

# <codecell> Plots

t_interest = 85*u.day
fig, axs = plt.subplots(2, 2, subplot_kw={"projection": "polar"})
gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
#(axs[0,0], axs[0,1]), (axs[1,0], axs[1,1]) = gs.subplots(sharex='col', sharey='row')

fig, ax, cnt = HA.plot(model_true, t_interest, fighandle = fig, axhandle = axs[0,0], minimalplot = True)
axs[0,0].set_title('(a) True state', fontsize=16)
HA.plot(model_back, t_interest, fighandle = fig, axhandle = axs[0,1], minimalplot = True)
axs[0,1].set_title('(b) Corotation (back in time)', fontsize=16)
HA.plot(model_both, t_interest, fighandle = fig, axhandle = axs[1,0], minimalplot = True)
axs[1,0].set_title('(c) Corotation (smooth in time)', fontsize=16)
HA.plot(model_dtw, t_interest, fighandle = fig, axhandle = axs[1,1], minimalplot = True)
axs[1,1].set_title('(d) Dynamic time warping', fontsize=16)
#HA.animate(model, 'vt_test')
#fig.tight_layout() 
 # Add color bar

dw = 0.005
dh = 0.045
left = axs[0,0].get_position().x0 + dw
bottom = axs[1,1].get_position().y0 - dh
wid = 2*axs[0,0].get_position().width +4*dw 
cbaxes = fig.add_axes([left, bottom, wid, 0.03])
cbar1 = fig.colorbar(cnt, cax=cbaxes, orientation='horizontal')
cbar1.set_ticks(np.arange(200, 810, 100))
cbar1.set_label('V [km/s]')


# <codecell> plot dV from truth

id_t = np.argmin(np.abs(model.time_out - t_interest))
v_sub = np.abs(model_true.v_grid.value[id_t, :, :].copy() - model_both.v_grid.value[id_t, :, :].copy())
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})

# Get plotting data
lon_arr, dlon, nlon = H.longitude_grid()
lon, rad = np.meshgrid(lon_arr.value, model.r.value)
mymap = mpl.cm.gist_heat

plotvmin = 0
plotvmax = 150.1
dv = 5
ylab = "Solar Wind Speed (km/s)"

# Insert into full array
if lon_arr.size != model.lon.size:
    v = np.zeros((model.nr, nlon)) * np.NaN
    if model.lon.size != 1:
        for i, lo in enumerate(model.lon):
            id_match = np.argwhere(lon_arr == lo)[0][0]
            v[:, id_match] = v_sub[:, i]
    else:
        print('Warning: Trying to contour single radial solution will fail.')
else:
    v = v_sub

# Pad out to fill the full 2pi of contouring
pad = lon[:, 0].reshape((lon.shape[0], 1)) + model.twopi
lon = np.concatenate((lon, pad), axis=1)
pad = rad[:, 0].reshape((rad.shape[0], 1))
rad = np.concatenate((rad, pad), axis=1)
pad = v[:, 0].reshape((v.shape[0], 1))
v = np.concatenate((v, pad), axis=1)

mymap.set_over('lightgrey')
mymap.set_under([0, 0, 0])
levels = np.arange(plotvmin, plotvmax + dv, dv)

    
cnt = ax.contourf(lon, rad, v, levels=levels, cmap=mymap, extend='both')

# Add on CME boundaries
cme_colors = ['r', 'c', 'm', 'y', 'deeppink', 'darkorange']
for j, cme in enumerate(model.cmes):
    cid = np.mod(j, len(cme_colors))
    cme_lons = cme.coords[id_t]['lon']
    cme_r = cme.coords[id_t]['r'].to(u.solRad)
    if np.any(np.isfinite(cme_r)):
        # Pad out to close the profile.
        cme_lons = np.append(cme_lons, cme_lons[0])
        cme_r = np.append(cme_r, cme_r[0])
        ax.plot(cme_lons, cme_r, '-', color=cme_colors[cid], linewidth=3)
   
    
ax.set_ylim(0, model.r.value.max())
ax.set_yticklabels([])
ax.set_xticklabels([]) 

 # Add color bar
pos = ax.get_position()
dw = 0.005
dh = 0.045
left = pos.x0 + dw
bottom = pos.y0 - dh
wid = pos.width - 2 * dw
cbaxes = fig.add_axes([left, bottom, wid, 0.03])
cbar1 = fig.colorbar(cnt, cax=cbaxes, orientation='horizontal')
cbar1.set_label(ylab)
cbar1.set_ticks(np.arange(plotvmin, plotvmax, dv*20))
    