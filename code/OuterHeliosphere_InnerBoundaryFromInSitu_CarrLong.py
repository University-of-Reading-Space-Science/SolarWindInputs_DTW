# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 11:01:26 2021

@author: mathewjowens

a script to test different methods of processing 1-AU in situ observations to 
generate inner boundary conditions for solar wind models of the outer 
heliosphere
"""

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.offsetbox import AnchoredText
import pandas as pd
import datetime 
import pickle as pl
import h5py

from dtaidistance import dtw

import helio_time as htime
import helio_coords as hcoord

dt = 1
nlon_grid = 128
starttime = datetime.datetime(2018, 4, 1, 0, 0, 0)
endtime =  datetime.datetime(2018, 10, 1, 0, 0, 0)
#endtime =  datetime.datetime(2018, 4, 15, 0, 0, 0)

savedir =  '..\\output\\'
datadir =  '..\\data\\'
savenow = False #whether to save the reconstructed data

# <codecell> Functions
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

#interpolate to the data time step - use nearest value, to avoid filling in across 0/2pi
def nearest_interp(xi, x, y):
    idx = np.abs(x - xi[:,None])
    return y[idx.argmin(axis=1)]

daysec = 24 * 60 * 60 * u.s
kms = u.km / u.s
synodic_period = 27.2753 * daysec  # Solar Synodic rotation period from Earth.
sidereal_period = 25.38 *daysec 
omega_sidereal = 2*np.pi *u.rad / sidereal_period
omega_synodic = 2*np.pi *u.rad / synodic_period
domega = omega_synodic - omega_sidereal

# <codecell> Load in the HGI coords for the period

#load the planetary ephemeris data from  https://omniweb.gsfc.nasa.gov/coho/helios/heli.html

#Earth
filepath = datadir + 'Earth_HGI.lst'
pos_Earth = pd.read_csv(filepath,
                     skiprows = 1, delim_whitespace=True,
                     names=['year','doy',
                            'rad_au','HGI_lat','HGI_lon'])
#convert to mjd
pos_Earth['mjd'] = htime.doyyr2mjd(pos_Earth['doy'],pos_Earth['year'])


#create OMNI position files
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


# <codecell> generate some synthetic data with steady-state conditions



nlon = 360
dphi = 360/nlon
lon_Carr = np.arange(dphi/2,360.1-dphi/2,dphi) *u.deg
lon_Carr = lon_Carr.to(u.rad)


#linear evolution

def vlong_timeevolve_lin(t_days):
    v0 = 400 * (u.km/u.s)
    m = 5 * (u.km/u.s)/u.deg
    
    v1max = 700 * (u.km/u.s)
    v1width = 50 * u.deg
    v1start = 100 * u.deg
    v1start_wrap = (100 + 360) * u.deg #for when the stream moves past 360 Carr lon
    
    v2max = 500 * (u.km/u.s)
    v2width = 40 * u.deg
    v2start = 250 * u.deg

    #put together the longitudinal series
    #base speed
    vlong = np.ones((nlon)) * v0
    
    v1start = v1start - t_days*u.deg
    v1start_wrap = v1start_wrap - t_days*u.deg
    
    v2start = v2start - 0.5*t_days*u.deg
    v1max = v1max - 1*t_days * (u.km/u.s)
    v2max = v2max + 2*t_days * (u.km/u.s)
    
    #declines from high speed intervals
    declinewidth = (v1max - v0)/m
    mask = (lon_Carr >= v1start+v1width) & (lon_Carr <= v1start+v1width+declinewidth)
    vlong[mask] = v1max - m * (lon_Carr[mask] - (v1start+v1width))
    
    declinewidth = (v1max - v0)/m
    mask = (lon_Carr >= v1start_wrap+v1width) & (lon_Carr <= v1start_wrap+v1width+declinewidth)
    vlong[mask] = v1max - m * (lon_Carr[mask] - (v1start_wrap+v1width))
    
    declinewidth = (v2max - v0)/m
    mask = (lon_Carr >= v2start+v2width) & (lon_Carr <= v2start+v2width+declinewidth)
    vlong[mask] = v2max - m * (lon_Carr[mask] - (v2start+v2width))
    
    #put the streams in
    mask = (lon_Carr >= v1start) & (lon_Carr <= v1start+v1width)
    vlong[mask] = v1max 
    mask = (lon_Carr >= v1start_wrap) & (lon_Carr <= v1start_wrap+v1width)
    vlong[mask] = v1max 
    
    mask = (lon_Carr >= v2start) & (lon_Carr <= v2start+v2width)
    vlong[mask] = v2max 
    
    mask = vlong < v0
    vlong[mask] = v0
    
    mask = vlong > 800 * (u.km/u.s)
    vlong[mask] = 800 * (u.km/u.s)
    
    
    #oops, this looks like a time series. flip it to make a longitudinal series
    vlong = np.flip(vlong)
    
    return vlong


#non-linear eovlution

def vlong_timeevolve_sin(t_days):
    v0 = 400 * (u.km/u.s)
    m = 5 * (u.km/u.s)/u.deg
    
    v1max = 700 * (u.km/u.s)
    v1width = 50 * u.deg
    v1start = 200 * u.deg
    v1start_wrap = (200 + 360) * u.deg #for when the stream moves past 360 Carr lon
    
    v1max = v1max - 100*np.sin(2*np.pi * t_days/200) * (u.km/u.s)

    #put together the longitudinal series
    #base speed
    vlong = np.ones((nlon)) * v0
    
    v1start = v1start - 100*np.sin(2*np.pi * t_days/200)*u.deg - t_days*u.deg
    v1start_wrap = v1start_wrap - 100*np.sin(2*np.pi * t_days/200)*u.deg - t_days*u.deg
    
    
    #declines from high speed intervals
    declinewidth = (v1max - v0)/m
    mask = (lon_Carr >= v1start+v1width) & (lon_Carr <= v1start+v1width+declinewidth)
    vlong[mask] = v1max - m * (lon_Carr[mask] - (v1start+v1width))
    
    declinewidth = (v1max - v0)/m
    mask = (lon_Carr >= v1start_wrap+v1width) & (lon_Carr <= v1start_wrap+v1width+declinewidth)
    vlong[mask] = v1max - m * (lon_Carr[mask] - (v1start_wrap+v1width))
   
    #put the streams in
    mask = (lon_Carr >= v1start) & (lon_Carr <= v1start+v1width)
    vlong[mask] = v1max 
    mask = (lon_Carr >= v1start_wrap) & (lon_Carr <= v1start_wrap+v1width)
    vlong[mask] = v1max 
    
  
    mask = vlong < v0
    vlong[mask] = v0
    
    mask = vlong > 800 * (u.km/u.s)
    vlong[mask] = 800 * (u.km/u.s)
    
    
    #oops, this looks like a time series. flip it to make a longitudinal series
    vlong = np.flip(vlong)
    
    return vlong

#change the line colours to use the inferno colormap
import cycler
n = 6
color = plt.cm.inferno(np.linspace(0, 1,n))
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

fig_model = plt.figure()

ax=plt.subplot(211)
plt.plot(lon_Carr*180/np.pi, vlong_timeevolve_lin(0),label = 't = 0 days')
plt.plot(lon_Carr*180/np.pi, vlong_timeevolve_lin(27),label = 't = 27 days')
plt.plot(lon_Carr*180/np.pi, vlong_timeevolve_lin(27*2),label = 't = 54 days')
plt.plot(lon_Carr*180/np.pi, vlong_timeevolve_lin(27*3),label = 't = 81 days')
plt.plot(lon_Carr*180/np.pi, vlong_timeevolve_lin(27*4),label = 't = 108 days')
plt.legend(loc='upper left')
ax.get_xaxis().set_ticks([0,90,180,270,360])
plt.ylabel('V, linearly evolving [km/s]')

anchored_text = AnchoredText("(a)", loc='upper right')
ax.add_artist(anchored_text)

#ax.text(0.0, 1.0, '(a)', transform=ax.transAxes + trans,
#            fontsize='medium', verticalalignment='top', fontfamily='serif',
#            bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))


ax=plt.subplot(212)
plt.plot(lon_Carr*180/np.pi, vlong_timeevolve_sin(0),label = 't = 0 days')
plt.plot(lon_Carr*180/np.pi, vlong_timeevolve_sin(27),label = 't = 27 days')
plt.plot(lon_Carr*180/np.pi, vlong_timeevolve_sin(27*2),label = 't = 54 days')
plt.plot(lon_Carr*180/np.pi, vlong_timeevolve_sin(27*3),label = 't = 81 days')
plt.plot(lon_Carr*180/np.pi, vlong_timeevolve_sin(27*4),label = 't = 108 days')
plt.legend(loc='upper left')
plt.xlabel('Carr long [deg]')
plt.ylabel('V, nonlinearly evolving [km/s]')
ax.get_xaxis().set_ticks([0,90,180,270,360])

anchored_text = AnchoredText("(b)", loc='upper right')
ax.add_artist(anchored_text)

#ax.text(0.0, 1.0, '(b)', transform=ax.transAxes + trans,
#            fontsize='medium', verticalalignment='top', fontfamily='serif',
#            bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

pl.dump(fig_model, open(savedir + 'V_model_CarrLon_linePlots.pickle','wb'))
#figx = pickle.load(open('V_model_CarrLon_linePlots.pickle', 'rb'))
#figx.show()
# <codecell> generate time series at all Carr longitudes, assuming corotation



dphi_grid = 360/nlon_grid
lon_grid = np.arange(dphi_grid/2, 360.1-dphi_grid/2, dphi_grid) * np.pi/180 * u.rad
lon_edges = np.arange(0, 360.1, dphi_grid) * np.pi/180 * u.rad

smjd=OMNI['mjd'][0] 
fmjd=OMNI['mjd'][-1] 
time_grid = np.arange(smjd,fmjd+dt/2,dt)
time_edges = np.arange(smjd-dt/2,fmjd+dt/2,dt)

#the initial phase difference between Carr and HGI longitudes
dphi0 = (OMNI['HGI_lon'][0]*u.rad - OMNI['Carr_lon'][0]*u.rad)

vgrid_Carr_tlin = np.ones((nlon_grid,len(time_grid)))*np.nan
vgrid_Carr_tsin = np.ones((nlon_grid,len(time_grid)))*np.nan
for t in range(0,len(time_grid)):
    delta_phi = 0#omega_sidereal.value * (time_grid[t] - smjd)*daysec.value *u.rad
     
      
    #time evolving solar wind speed
    t_day = time_grid[t] - smjd
    
    vlong_t = vlong_timeevolve_lin(t_day)  
    vgrid_Carr_tlin[:,t] = np.interp(lon_grid - delta_phi,
                           lon_Carr, vlong_t, period = 2*np.pi)
    vlong_t = vlong_timeevolve_sin(t_day)  
    vgrid_Carr_tsin[:,t] = np.interp(lon_grid - delta_phi,
                           lon_Carr, vlong_t, period = 2*np.pi)

    

vgrid_HGI_tlin = np.ones((nlon_grid,len(time_grid)))*np.nan
vgrid_HGI_tsin = np.ones((nlon_grid,len(time_grid)))*np.nan
for t in range(0,len(time_grid)):
    delta_phi = omega_sidereal.value * (time_grid[t] - smjd)*daysec.value *u.rad
    
    #add the initial phase offset
    delta_phi = delta_phi + dphi0
          
    #time evolving solar wind speed
    t_day = time_grid[t] - smjd
    
    vlong_t = vlong_timeevolve_lin(t_day)  
    vgrid_HGI_tlin[:,t] = np.interp(zerototwopi(lon_grid.value - delta_phi.value),
                           lon_Carr.value, vlong_t, period = 2*np.pi)
    vlong_t = vlong_timeevolve_sin(t_day)  
    vgrid_HGI_tsin[:,t] = np.interp(zerototwopi(lon_grid.value - delta_phi.value),
                           lon_Carr.value, vlong_t, period = 2*np.pi)
 
#save the HGI data for use with HUXt
if savenow:
    h5f = h5py.File(savedir + 'vinputs_TrueStateHGI.h5', 'w')
    h5f.create_dataset('vgrid_HGI_tlin', data=vgrid_HGI_tlin)
    h5f.create_dataset('vgrid_HGI_tsin', data=vgrid_HGI_tsin)
    h5f.create_dataset('time_edges', data=time_edges)
    h5f.create_dataset('lon_edges', data=lon_edges)
    h5f.create_dataset('lon_grid', data=lon_grid)
    h5f.create_dataset('time_grid', data=time_grid)
    h5f.close()       

#generate the Earth time series
OMNI['Vsynth_Carr_tlin'] = np.nan
OMNI['Vsynth_Carr_tsin'] = np.nan
for t in range(0,len(OMNI)):
    delta_phi = 0#omega_sidereal.value * (OMNI['mjd'][t] - smjd)*daysec.value *u.rad
    #vslice = np.interp(lon_Carr - delta_phi,
    #                       lon_Carr, vlong, period = 2*np.pi)
    
    t_day = OMNI['mjd'][t] - smjd
    vlong_t = vlong_timeevolve_lin(t_day)
    vslice_t = np.interp(lon_Carr - delta_phi,
                           lon_Carr, vlong_t, period = 2*np.pi)
    
    OMNI['Vsynth_Carr_tlin'][t] =  np.interp(OMNI['Carr_lon'][t]*u.rad, 
                                       lon_Carr,vslice_t, period = 2*np.pi).value
        
    vlong_t = vlong_timeevolve_sin(t_day)
    vslice_t = np.interp(lon_Carr - delta_phi,
                           lon_Carr, vlong_t, period = 2*np.pi)
    OMNI['Vsynth_Carr_tsin'][t] =  np.interp(OMNI['Carr_lon'][t]*u.rad, 
                                       lon_Carr,vslice_t, period = 2*np.pi).value
        
        
#generate the Earth time series
OMNI['Vsynth_HGI_tlin'] = np.nan
OMNI['Vsynth_HGI_tsin'] = np.nan
for t in range(0,len(OMNI)):
    
    t_day = OMNI['mjd'][t] - smjd
    delta_phi = omega_sidereal.value * (OMNI['mjd'][t] - smjd)*daysec.value *u.rad
    
    #add the initial phase offset
    delta_phi = delta_phi + dphi0
    
    vlong_t = vlong_timeevolve_lin(t_day)
    vslice_t = np.interp(lon_Carr - delta_phi,
                           lon_Carr, vlong_t, period = 2*np.pi)
        
    OMNI['Vsynth_HGI_tlin'][t] =  np.interp(OMNI['HGI_lon'][t]*u.rad, 
                                       lon_Carr,vslice_t, period = 2*np.pi).value
        
    vlong_t = vlong_timeevolve_sin(t_day)
    vslice_t = np.interp(lon_Carr - delta_phi,
                           lon_Carr, vlong_t, period = 2*np.pi)
    OMNI['Vsynth_HGI_tsin'][t] =  np.interp(OMNI['HGI_lon'][t]*u.rad, 
                                       lon_Carr,vslice_t, period = 2*np.pi).value
    
##produce the STA series
#STA['Vsynth_Carr_steady'] = np.nan
#STA['Vsynth_Carr_t'] = np.nan
#for t in range(0,len(STA)):
#    delta_phi = 0#omega_sidereal.value * (STA['mjd'][t] - smjd)*daysec.value *u.rad
#    vslice = np.interp(lon_Carr - delta_phi,
#                           lon_Carr, vlong, period = 2*np.pi)
#    
#    t_day = STA['mjd'][t] - smjd
#    vlong_t = vlong_timeevolve(t_day)
#    vslice_t = np.interp(lon_Carr - delta_phi,
#                           lon_Carr, vlong_t, period = 2*np.pi)
#    
#    STA['Vsynth_Carr_steady'][t] =  np.interp(STA['Carr_lon'][t]*u.rad, 
#                                       lon_Carr, vslice, period = 2*np.pi).value
#    STA['Vsynth_Carr_t'][t] =  np.interp(STA['Carr_lon'][t]*u.rad, 
#                                       lon_Carr, vslice_t, period = 2*np.pi).value

# <codecell> Plot V in Carr frame
    
xmax = np.floor(time_edges[-1]-smjd)
fig_Carrlon_t = plt.figure()
loc='upper left'

ax=plt.subplot(221)
pc = ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, vgrid_Carr_tlin, 
              shading='flat', edgecolor = 'face',norm=plt.Normalize(300,800))
#plt.xlabel('Time [days]')
plt.ylabel('Carr longitude [deg]')
plt.title('Linearly-evolving solar wind')
#plot Earth position for this time
plt.plot(OMNI['mjd'] - smjd, OMNI['Carr_lon']*180/np.pi,'k.',label='Earth',
         markersize = 0.5)
#plt.plot(STA['mjd'] - smjd, STA['Carr_lon']*180/np.pi,'r',label='STA')
ax.get_xaxis().set_ticklabels([])
ax.set_xlim((0,xmax))

anchored_text = AnchoredText("(a)", loc=loc)
ax.add_artist(anchored_text)


ax=plt.subplot(223)
plt.plot(OMNI['mjd'] - smjd, OMNI['Vsynth_Carr_tlin'],'k',label='Earth')
#plt.plot(STA['mjd'] - smjd, STA['Vsynth_Carr_steady'],'r',label='STA')
plt.ylabel('Solar wind speed [km/s]')
plt.xlabel('Time [days]')
#plt.legend()
ax.set_xlim((0,xmax))

anchored_text = AnchoredText("(c)", loc=loc)
ax.add_artist(anchored_text)

ax=plt.subplot(222)
ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, vgrid_Carr_tsin, 
              shading='flat', edgecolor = 'face',norm=plt.Normalize(300,800))
#plt.xlabel('Time [days]')
#plt.ylabel('Carr longitude [deg]')
ax.get_yaxis().set_ticklabels([])
plt.title('Nonlinearly-evolving solar wind')
#plot Earth position for this time
plt.plot(OMNI['mjd'] - smjd, OMNI['Carr_lon']*180/np.pi,'k.',label='Earth',
          markersize = 0.5)
#plt.plot(STA['mjd'] - smjd, STA['Carr_lon']*180/np.pi,'r',label='STA')
ax.get_xaxis().set_ticklabels([])
ax.set_xlim((0,xmax))

anchored_text = AnchoredText("(b)", loc=loc)
ax.add_artist(anchored_text)

ax=plt.subplot(224)
plt.plot(OMNI['mjd'] - smjd, OMNI['Vsynth_Carr_tsin'],'k',label='Earth')
#plt.plot(STA['mjd'] - smjd, STA['Vsynth_Carr_t'],'r',label='STA')
#plt.ylabel('Solar wind speed [km/s]')
ax.get_yaxis().set_ticklabels([])
ax.set_xlim((0,xmax))
plt.xlabel('Time [days]')
#plt.legend()

anchored_text = AnchoredText("(d)", loc=loc)
ax.add_artist(anchored_text)

plt.tight_layout()


plt.subplots_adjust(bottom=0.1, right=0.85, top=0.9)

cax = plt.axes([0.89, 0.1, 0.03, 0.8])
cbar = plt.colorbar(pc, cax=cax)
cbar.set_label('V [km/s]')

#pl.dump(fig_Carrlon_t, open(savedir + 'V_model_CarrLon_t_ColourMaps.pickle','wb'))
#figx = pickle.load(open('V_model_CarrLon_linePlots.pickle', 'rb'))
#figx.show()

# <codecell> Plot V in HGI frame
fig_HGI_t = plt.figure()
loc='upper right'

ax=plt.subplot(221)
pc = ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, vgrid_HGI_tlin, 
              shading='flat', edgecolor = 'face',norm=plt.Normalize(300,800))
#plt.xlabel('Time [days]')
plt.ylabel('HGI longitude [deg]')
plt.title('Linearly-evolving solar wind')
#plot Earth position for this time
plt.plot(OMNI['mjd'] - smjd, OMNI['HGI_lon']*180/np.pi,'k.',label='Earth',
         markersize = 0.5)
#plt.plot(STA['mjd'] - smjd, STA['Carr_lon']*180/np.pi,'r',label='STA')
ax.get_xaxis().set_ticklabels([])
ax.set_xlim((0,xmax))

anchored_text = AnchoredText("(a)", loc=loc)
ax.add_artist(anchored_text)



ax=plt.subplot(223)
plt.plot(OMNI['mjd'] - smjd, OMNI['Vsynth_Carr_tlin'],'k',label='Earth')
plt.plot(OMNI['mjd'] - smjd, OMNI['Vsynth_HGI_tlin'],'r--',label='Earth (HGI)')
plt.ylabel('Solar wind speed [km/s]')
plt.xlabel('Time [days]')
plt.legend()
ax.set_xlim((0,xmax))

anchored_text = AnchoredText("(c)", loc=loc)
ax.add_artist(anchored_text)


ax=plt.subplot(222)
ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, vgrid_HGI_tsin, 
              shading='flat', edgecolor = 'face',norm=plt.Normalize(300,800))
#plt.xlabel('Time [days]')
#plt.ylabel('Carr longitude [deg]')
ax.get_yaxis().set_ticklabels([])
plt.title('Nonlinearly-evolving solar wind')
#plot Earth position for this time
plt.plot(OMNI['mjd'] - smjd, OMNI['HGI_lon']*180/np.pi,'k.',label='Earth',
          markersize = 0.5)
#plt.plot(STA['mjd'] - smjd, STA['Carr_lon']*180/np.pi,'r',label='STA')
ax.get_xaxis().set_ticklabels([])
ax.set_xlim((0,xmax))

anchored_text = AnchoredText("(b)", loc=loc)
ax.add_artist(anchored_text)


ax=plt.subplot(224)
plt.plot(OMNI['mjd'] - smjd, OMNI['Vsynth_Carr_tsin'],'k',label='Earth')
plt.plot(OMNI['mjd'] - smjd, OMNI['Vsynth_HGI_tsin'],'r--',label='Earth (HGI)')
#plt.plot(STA['mjd'] - smjd, STA['Vsynth_Carr_t'],'r',label='STA')
#plt.ylabel('Solar wind speed [km/s]')
ax.get_yaxis().set_ticklabels([])
plt.xlabel('Time [days]')
plt.legend()
ax.set_xlim((0,xmax))

anchored_text = AnchoredText("(d)", loc=loc)
ax.add_artist(anchored_text)


plt.tight_layout()


plt.subplots_adjust(bottom=0.1, right=0.85, top=0.9)

cax = plt.axes([0.89, 0.1, 0.03, 0.8])
cbar = plt.colorbar(pc, cax=cax)
cbar.set_label('V [km/s]')

#pl.dump(fig_Carrlon_t, open(savedir + 'V_model_HGILon_t_ColourMaps.pickle','wb'))
#figx = pickle.load(open('V_model_CarrLon_linePlots.pickle', 'rb'))
#figx.show()

# <codecell> Reconstruct  V at all lons using synthetic OMNI obs


vgrid_Carr_recon_back_tlin = np.ones((nlon_grid,len(time_grid)))*np.nan
vgrid_Carr_recon_forward_tlin = np.ones((nlon_grid,len(time_grid)))*np.nan
vgrid_Carr_recon_both_tlin = np.ones((nlon_grid,len(time_grid)))*np.nan
for t in range(0,len(time_grid)):
    #find nearest time and current Carrington longitude
    t_id = np.argmin(np.abs(OMNI['mjd'] - time_grid[t]))
    Elong = OMNI['Carr_lon'][t_id]*u.rad
    
    #get the Carrington longitude difference from current Earth pos
    dlong_back = zerototwopi(lon_grid.value - Elong.value) *u.rad
    dlong_forward = zerototwopi(Elong.value- lon_grid.value) *u.rad
    
    dt_back = (dlong_back / (omega_omni)).to(u.day)
    dt_forward = (dlong_forward / (omega_omni)).to(u.day)
    
    vgrid_Carr_recon_back_tlin[:,t] = np.interp(time_grid[t] - dt_back.value, 
                                 OMNI['mjd'], 
                                 OMNI['Vsynth_Carr_tlin'], left = np.nan, right = np.nan)
    vgrid_Carr_recon_forward_tlin[:,t] = np.interp(time_grid[t] + dt_forward.value, 
                                 OMNI['mjd'], 
                                 OMNI['Vsynth_Carr_tlin'], left = np.nan, right = np.nan)
    vgrid_Carr_recon_both_tlin[:,t] = (dt_forward * vgrid_Carr_recon_back_tlin[:,t] +
                             dt_back * vgrid_Carr_recon_forward_tlin[:,t])/(dt_forward + dt_back)
    
vgrid_HGI_recon_back_tlin = np.ones((nlon_grid,len(time_grid)))*np.nan
vgrid_HGI_recon_forward_tlin = np.ones((nlon_grid,len(time_grid)))*np.nan
vgrid_HGI_recon_both_tlin = np.ones((nlon_grid,len(time_grid)))*np.nan
for t in range(0,len(time_grid)):
    #find nearest time and current Carrington longitude
    t_id = np.argmin(np.abs(OMNI['mjd'] - time_grid[t]))
    Elong = OMNI['HGI_lon'][t_id]*u.rad
    
    #get the HGI longitude difference from current Earth pos
    dlong_back = zerototwopi(lon_grid.value - Elong.value) *u.rad
    dlong_forward = zerototwopi(Elong.value- lon_grid.value) *u.rad
    
    dt_back = (dlong_back / (omega_omni)).to(u.day)
    dt_forward = (dlong_forward / (omega_omni)).to(u.day)
    
    vgrid_HGI_recon_back_tlin[:,t] = np.interp(time_grid[t] - dt_back.value, 
                                 OMNI['mjd'], 
                                 OMNI['Vsynth_HGI_tlin'], left = np.nan, right = np.nan)
    vgrid_HGI_recon_forward_tlin[:,t] = np.interp(time_grid[t] + dt_forward.value, 
                                 OMNI['mjd'], 
                                 OMNI['Vsynth_HGI_tlin'], left = np.nan, right = np.nan)
    vgrid_HGI_recon_both_tlin[:,t] = (dt_forward * vgrid_HGI_recon_back_tlin[:,t] +
                             dt_back * vgrid_HGI_recon_forward_tlin[:,t])/(dt_forward + dt_back)

vgrid_Carr_recon_back_tsin = np.ones((nlon_grid,len(time_grid)))*np.nan
vgrid_Carr_recon_forward_tsin = np.ones((nlon_grid,len(time_grid)))*np.nan
vgrid_Carr_recon_both_tsin = np.ones((nlon_grid,len(time_grid)))*np.nan
for t in range(0,len(time_grid)):
    #find nearest time
    t_id = np.argmin(np.abs(OMNI['mjd'] - time_grid[t]))
    Elong = OMNI['Carr_lon'][t_id]*u.rad
    
    #get the longitude difference from Earth pos
    dlong_back = zerototwopi(lon_grid.value - Elong.value) *u.rad
    dlong_forward = zerototwopi(Elong.value- lon_grid.value) *u.rad
    
    dt_back = (dlong_back / (omega_omni)).to(u.day)
    dt_forward = (dlong_forward / (omega_omni)).to(u.day)
    
    #get the vlong value for thiws time
    t_back = (time_grid[t] - dt_back.value) - smjd
    
    vgrid_Carr_recon_back_tsin[:,t] = np.interp(time_grid[t] - dt_back.value, 
                                 OMNI['mjd'], 
                                 OMNI['Vsynth_Carr_tsin'], left = np.nan, right = np.nan)
    vgrid_Carr_recon_forward_tsin[:,t] = np.interp(time_grid[t] + dt_forward.value, 
                                 OMNI['mjd'], 
                                 OMNI['Vsynth_Carr_tsin'], left = np.nan, right = np.nan)
    vgrid_Carr_recon_both_tsin[:,t] = (dt_forward * vgrid_Carr_recon_back_tsin[:,t] +
                             dt_back * vgrid_Carr_recon_forward_tsin[:,t])/(dt_forward + dt_back)
    
vgrid_HGI_recon_back_tsin = np.ones((nlon_grid,len(time_grid)))*np.nan
vgrid_HGI_recon_forward_tsin = np.ones((nlon_grid,len(time_grid)))*np.nan
vgrid_HGI_recon_both_tsin = np.ones((nlon_grid,len(time_grid)))*np.nan
for t in range(0,len(time_grid)):
    #find nearest time and current Carrington longitude
    t_id = np.argmin(np.abs(OMNI['mjd'] - time_grid[t]))
    Elong = OMNI['HGI_lon'][t_id]*u.rad
    
    #get the HGI longitude difference from current Earth pos
    dlong_back = zerototwopi(lon_grid.value - Elong.value) *u.rad
    dlong_forward = zerototwopi(Elong.value- lon_grid.value) *u.rad
    
    dt_back = (dlong_back / (omega_omni)).to(u.day)
    dt_forward = (dlong_forward / (omega_omni)).to(u.day)
    
    vgrid_HGI_recon_back_tsin[:,t] = np.interp(time_grid[t] - dt_back.value, 
                                 OMNI['mjd'], 
                                 OMNI['Vsynth_HGI_tsin'], left = np.nan, right = np.nan)
    vgrid_HGI_recon_forward_tsin[:,t] = np.interp(time_grid[t] + dt_forward.value, 
                                 OMNI['mjd'], 
                                 OMNI['Vsynth_HGI_tsin'], left = np.nan, right = np.nan)
    vgrid_HGI_recon_both_tsin[:,t] = (dt_forward * vgrid_HGI_recon_back_tsin[:,t] +
                             dt_back * vgrid_HGI_recon_forward_tsin[:,t])/(dt_forward + dt_back)
    
#save the HGI data for use with HUXt
if savenow:
    h5f = h5py.File(savedir + 'vinputs_corot_HGI.h5', 'w')
    h5f.create_dataset('vgrid_HGI_recon_back_tlin', data=vgrid_HGI_recon_back_tlin)
    h5f.create_dataset('vgrid_HGI_recon_both_tlin', data=vgrid_HGI_recon_both_tlin)
    h5f.create_dataset('vgrid_HGI_recon_back_tsin', data=vgrid_HGI_recon_back_tsin)
    h5f.create_dataset('vgrid_HGI_recon_both_tsin', data=vgrid_HGI_recon_both_tsin)
    h5f.create_dataset('time_edges', data=time_edges)
    h5f.create_dataset('lon_edges', data=lon_edges)
    h5f.create_dataset('lon_grid', data=lon_grid)
    h5f.create_dataset('time_grid', data=time_grid)
    h5f.close()       


# <codecell> Test reconstruction using Dynamic Time Warping
    
    
    
def single_spacecraft_DTW(obs_mjd, obs_V, obs_Carrlon, ref_mjd, ref_Carrlon, npoints = 100):
    """
    A function to use dynamic time warping to determine the solar wind speed at 
    a given refererence time (ref_mjd) and Carrington longitude (Carr_lon)
    from a single-spacecraft observation (e.g., OMNI).
    """
    from dtaidistance import dtw
    
    #find nearest timestep
    t_id = np.argmin(abs(obs_mjd - ref_mjd))

    #take 2 pi centred on the reference long as the periods to match to
    #=============================================================
    #difference in Carr lon of OMNI and ref point at the reference time
    d_Carrlon_ref_OMNI = ref_Carrlon - obs_Carrlon[t_id]
    
    #unwrap the Carr lon
    carrlonunwrap = np.unwrap(obs_Carrlon)
    #fidn teh unwrapped reference longitude at the reference time 
    ref_lon_unwrap = carrlonunwrap[t_id] + d_Carrlon_ref_OMNI
    
    #check whether OMNI is ahead or behind (in Carr long) the reference point at this time
    EarthIsAhead = True
    if obs_Carrlon[t_id] < ref_Carrlon:
        EarthIsAhead = False
    
    if EarthIsAhead:
        #carr lon decreases with time, therefore the start/end of the behind period is at larger Carr lon
        behind_start_id = np.argmin(abs(carrlonunwrap - (ref_lon_unwrap + 3*np.pi)))
        behind_end_id = np.argmin(abs(carrlonunwrap - (ref_lon_unwrap + 1*np.pi)))
        
        ahead_start_id = np.argmin(abs(carrlonunwrap - (ref_lon_unwrap + 1*np.pi)))
        ahead_end_id = np.argmin(abs(carrlonunwrap - (ref_lon_unwrap - 1*np.pi)))
    else:
        behind_start_id = np.argmin(abs(carrlonunwrap - (ref_lon_unwrap + 1*np.pi)))
        behind_end_id = np.argmin(abs(carrlonunwrap - (ref_lon_unwrap - 1*np.pi)))
        
        ahead_start_id = np.argmin(abs(carrlonunwrap - (ref_lon_unwrap - 1*np.pi)))
        ahead_end_id = np.argmin(abs(carrlonunwrap - (ref_lon_unwrap - 3*np.pi)))
    
    
    vlong_behind = obs_V[behind_start_id:behind_end_id]
    Carrlon_behind = obs_Carrlon[behind_start_id:behind_end_id]
    mjd_behind = obs_mjd[behind_start_id:behind_end_id]
    
    vlong_ahead = obs_V[ahead_start_id:ahead_end_id]
    Carrlon_ahead = obs_Carrlon[ahead_start_id:ahead_end_id]
    mjd_ahead = obs_mjd[ahead_start_id:ahead_end_id]
    
    #centre the Carr longitude so ref longitude is at pi
    lon_centred_behind =  zerototwopi(Carrlon_behind + np.pi - ref_Carrlon)
    lon_centred_ahead = zerototwopi(Carrlon_ahead + np.pi - ref_Carrlon)
        
    #interpolate the data to the required resolution
    
    
    
    #make sure behind and ahead series are the same length
    #assert(len(vlong_behind) == len(vlong_ahead))
    
    #compute the DTW betweeen the behind and ahead V series
    path = dtw.warping_path(vlong_behind, vlong_ahead,
                            window = int(len(vlong_behind)/5), 
                            psi = int(len(vlong_behind)/5))
 
    #loop through each warp line and determine the distance to the required point
    dy = np.zeros((len(path),1))
    i = 0
    for [map_x, map_y] in path:
        #find the equation of the line
        y1 = lon_centred_behind[map_x]
        y2 = lon_centred_ahead[map_y]
        x1 = mjd_behind[map_x]
        x2 = mjd_ahead[map_y]
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m*x1
        
        dy[i] = m * ref_mjd + c - np.pi
        i = i +1
    
    #find the closest line in Carrington longitude    
    dymin_id = np.argmin(abs(dy))
    path_dymin = path[dymin_id]
    
    #weight the speed by the time to the ahead/behind points
    dt_behind = ref_mjd - mjd_behind[path_dymin[0]]
    dt_ahead = mjd_ahead[path_dymin[1]] - ref_mjd
    
    v_ref = ( vlong_behind[path_dymin[0]] * dt_ahead/(dt_behind+dt_ahead) + 
              vlong_ahead[path_dymin[1]] * dt_behind/(dt_behind+dt_ahead))
    
    return v_ref

#select a particular time and longitude
t_ref = 90*u.day
lon_ref = 345*(np.pi/180) *u.rad

t_ref_mjd = t_ref.value + smjd

#find nearest timestep
t_id = np.argmin(abs(OMNI['mjd'] - t_ref_mjd))




#take 2 pi centred on the reference long as the periods to match to
#=============================================================
#difference in Carr lon of OMNI and ref point at the reference time
d_Carrlon_ref_OMNI = lon_ref.value - OMNI['Carr_lon'][t_id]


#unwrap the Carr lon
carrlonunwrap = np.unwrap(OMNI['Carr_lon'].values)
#the unwrapped Carr lon of OMNI at the reference time 
ref_lon_unwrap = carrlonunwrap[t_id] + d_Carrlon_ref_OMNI

#check whether OMNI is ahead or behind (in Carr long) the reference point at this time
EarthIsAhead = True
if OMNI['Carr_lon'][t_id] < lon_ref.value:
    EarthIsAhead = False

if EarthIsAhead:
    #carr lon decreases with time, therefore the start/end of the behind period is at larger Carr lon
    behind_start_id = np.argmin(abs(carrlonunwrap - (ref_lon_unwrap + 3*np.pi)))
    behind_end_id = np.argmin(abs(carrlonunwrap - (ref_lon_unwrap + 1*np.pi)))
    
    ahead_start_id = np.argmin(abs(carrlonunwrap - (ref_lon_unwrap + 1*np.pi)))
    ahead_end_id = np.argmin(abs(carrlonunwrap - (ref_lon_unwrap - 1*np.pi)))
else:
    behind_start_id = np.argmin(abs(carrlonunwrap - (ref_lon_unwrap + 1*np.pi)))
    behind_end_id = np.argmin(abs(carrlonunwrap - (ref_lon_unwrap - 1*np.pi)))
    
    ahead_start_id = np.argmin(abs(carrlonunwrap - (ref_lon_unwrap - 1*np.pi)))
    ahead_end_id = np.argmin(abs(carrlonunwrap - (ref_lon_unwrap - 3*np.pi)))

#check
plt.figure()
plt.plot(OMNI['mjd'],OMNI['Carr_lon'])
plt.plot(t_ref_mjd,lon_ref.value,'ro')
plt.plot(OMNI['mjd'][behind_start_id],OMNI['Carr_lon'][behind_start_id],'k+',label='behind start')
plt.plot(OMNI['mjd'][behind_end_id],OMNI['Carr_lon'][behind_end_id],'ko',label='behind end')
plt.plot(OMNI['mjd'][ahead_start_id],OMNI['Carr_lon'][ahead_start_id],'b+',label='ahead start')
plt.plot(OMNI['mjd'][ahead_end_id],OMNI['Carr_lon'][ahead_end_id],'bo',label='behind end')
plt.legend()

vlong_behind_t = OMNI['Vsynth_Carr_tlin'][behind_start_id:behind_end_id].values
Carrlon_behind_t = OMNI['Carr_lon'][behind_start_id:behind_end_id].values
mjd_behind_t = OMNI['mjd'][behind_start_id:behind_end_id].values
Carrlon_unwrap_behind_t = carrlonunwrap[behind_start_id:behind_end_id]

vlong_ahead_t = OMNI['Vsynth_Carr_tlin'][ahead_start_id:ahead_end_id].values
Carrlon_ahead_t = OMNI['Carr_lon'][ahead_start_id:ahead_end_id].values
mjd_ahead_t = OMNI['mjd'][ahead_start_id:ahead_end_id].values
Carrlon_unwrap_ahead_t = carrlonunwrap[ahead_start_id:ahead_end_id]


#reduce the resolution
res=10
x = vlong_behind_t[0:len(vlong_behind_t):res]
y = vlong_ahead_t[0:len(vlong_ahead_t):res]
x_t = mjd_behind_t[0:len(vlong_behind_t):res] - mjd_behind_t[0]
y_t = mjd_ahead_t[0:len(vlong_ahead_t):res] - mjd_ahead_t[0]
x_mjd = mjd_behind_t[0:len(vlong_behind_t):res] 
y_mjd = mjd_ahead_t[0:len(vlong_ahead_t):res] 
x_lon = Carrlon_behind_t[0:len(vlong_behind_t):res] 
y_lon = Carrlon_ahead_t[0:len(vlong_ahead_t):res] 

#centre the longitude maps on the ref longitude
ref_lon_centred = np.pi 

x_lon_centred =  zerototwopi(Carrlon_behind_t[0:len(vlong_behind_t):res] + np.pi - lon_ref.value)
y_lon_centred = zerototwopi(Carrlon_ahead_t[0:len(vlong_ahead_t):res] + np.pi - lon_ref.value)


path = dtw.warping_path(x, y , window = int(len(x)/5), psi = int(len(x)/5))


#plot the explanation

fig, ax = plt.subplots()
for [map_x, map_y] in path:
    #ax.plot([map_x, map_y], [x[map_x], y[map_y]], 'k', linewidth=4)
    ax.plot([x_mjd[map_x] - smjd, y_mjd[map_y] - smjd], 
            [x_lon_centred[map_x]*180/np.pi, y_lon_centred[map_y]*180/np.pi], 'k', linewidth=1)
for [map_x, map_y] in path:
    p=ax.scatter(x_mjd[map_x] - smjd,x_lon_centred[map_x]*180/np.pi,
               c=x[map_x],norm=plt.Normalize(300,700))
    ax.scatter(y_mjd[map_y] - smjd,y_lon_centred[map_y]*180/np.pi,
               c=y[map_y],norm=plt.Normalize(300,700))
plt.xlabel('Time [days]')
plt.ylabel('Carr lon [deg]')
ax.set_ylim(0,2*np.pi*180/np.pi)
plt.plot(t_ref_mjd - smjd,np.pi*180/np.pi,'ro')    
plt.colorbar(p,label="Vsw (km/s)")

#loop through each warp line and determine the distance to the required point
dy = np.zeros((len(path),1))
i = 0
for [map_x, map_y] in path:
    #find the equation of the line
    ya =  x_lon_centred[map_x]
    yb = y_lon_centred[map_y]
    xa = x_mjd[map_x]
    xb = y_mjd[map_y]
    m = (yb - ya) / (xb - xa)
    c = ya - m*xa
    
    dy[i] = m * t_ref_mjd + c - np.pi
    i = i +1

#find the closest line in longitude    
dymin_id = np.argmin(abs(dy))
path_dymin = path[dymin_id]

ax.plot([x_mjd[path_dymin[0]] - smjd, y_mjd[path_dymin[1]]- smjd], 
        [x_lon_centred[path_dymin[0]]*180/np.pi, y_lon_centred[path_dymin[1]]*180/np.pi],
        'r', linewidth=2)

ax.plot([x_mjd[path_dymin[0]] - smjd -1, y_mjd[path_dymin[1]]- smjd + 1.7], 
        [np.pi*180/np.pi, np.pi*180/np.pi], 'r--', linewidth=2)


#weight the speed by the time to the ahead/behind points
dt_behind = t_ref_mjd - x_mjd[path_dymin[0]]
dt_ahead = y_mjd[path_dymin[1]] - t_ref_mjd

v_ref = ( x[path_dymin[0]] * dt_ahead/(dt_behind+dt_ahead) + 
          y[path_dymin[1]] * dt_behind/(dt_behind+dt_ahead))


print(v_ref)

#test the function
v_func = single_spacecraft_DTW(OMNI['mjd'].values, OMNI['Vsynth_Carr_tlin'].values, 
                               OMNI['Carr_lon'].values, t_ref_mjd, lon_ref.value)

print(v_func)

# <codecell> plot the path matrix

#axs = dtwvis.plot_warpingpaths(x, y, paths, best_path)

s1 = vlong_behind_t
t1 = mjd_behind_t - mjd_behind_t[0]
s2 = vlong_ahead_t
t2 = mjd_ahead_t - mjd_ahead_t[0]
shownumbers = False
showlegend = True

#reduce the resolution
res = 5
s1 = s1[0:len(vlong_behind_t):res]
t1 = t1[0:len(vlong_behind_t):res]
s2 = s2[0:len(vlong_behind_t):res]
t2 = t2[0:len(vlong_behind_t):res]


d, paths = dtw.warping_paths(s1, s2 , window = int(len(s1)/2), psi = int(len(s1)/5))
path = dtw.best_path(paths)


"""Plot the warping paths matrix.
:param s1: Series 1
:param s2: Series 2
:param paths: Warping paths matrix
:param path: Path to draw (typically this is the best path)
:param filename: Filename for the image (optional)
:param shownumbers: Show distances also as numbers
:param showlegend: Show colormap legend
"""
try:
    from matplotlib import pyplot as plt
    from matplotlib import gridspec
    from matplotlib.ticker import FuncFormatter
except ImportError:
    logger.error("The plot_warpingpaths function requires the matplotlib package to be installed.")
ratio = max(len(s1), len(s2))
min_y = min(np.min(s1), np.min(s2)) - 20
max_y = max(np.max(s1), np.max(s2)) + 20

fig = plt.figure(figsize=(7, 7), frameon=True)
if showlegend:
    grows = 3
    gcols = 3
    height_ratios = [2, 6, 1]
    width_ratios = [2, 6, 1]
else:
    grows = 2
    gcols = 2
    height_ratios = [1, 6]
    width_ratios = [1, 6]
gs = gridspec.GridSpec(grows, gcols, wspace=0, hspace=0,
                       left=1.0, right=10.0, bottom=0, top=1.0,
                       height_ratios=height_ratios,
                       width_ratios=width_ratios)
max_s2_x = np.max(s2)
max_s2_y = np.max(t2)
max_s1_x = np.max(s1)
min_s1_x = np.min(s1) 
max_s1_y = np.max(t1)

if path is None:
    p = dtw.best_path(paths)
else:
    p = path

def format_fn2_x(tick_val, tick_pos):
    return max_s2_x - tick_val

def format_fn2_y(tick_val, tick_pos):
    return int(max_s2_y - tick_val)

ax0 = fig.add_subplot(gs[0, 0])
ax0.set_axis_off()
#ax0.text(0, 0, "Dist = {:.4f}".format(paths[p[-1][0], p[-1][1]]))
ax0.xaxis.set_major_locator(plt.NullLocator())
ax0.yaxis.set_major_locator(plt.NullLocator())

ax1 = fig.add_subplot(gs[0, 1])
ax1.set_ylim([min_y, max_y])
#ax1.set_axis_off()
#ax1.xaxis.tick_top()
# ax1.set_aspect(0.454)
ax1.plot(t1, s1, "-")
ax1.scatter(t1, s1, c=s1, cmap='viridis', norm=plt.Normalize(300,750))
ax1.set_ylabel('V [km/s]')
ax1.set_title('(a) CR N')
#ax1.get_xaxis().set_ticklabels([])
#ax1.xaxis.set_major_locator(plt.NullLocator())
#ax1.yaxis.set_major_locator(plt.NullLocator())

ax2 = fig.add_subplot(gs[1, 0])
ax2.set_xlim([max_y, min_y])
#ax2.set_axis_off()
# ax2.set_aspect(0.8)
# ax2.xaxis.set_major_formatter(FuncFormatter(format_fn2_x))
# ax2.yaxis.set_major_formatter(FuncFormatter(format_fn2_y))
#ax2.xaxis.set_major_locator(plt.NullLocator())
#ax2.yaxis.set_major_locator(plt.NullLocator())
ax2.plot(s2, t2, "-")
ax2.scatter(s2, t2, c=s2, cmap='viridis', norm=plt.Normalize(300,750))
ax2.set_xlabel('V [km/s]')
ax2.set_ylabel('Time [days]')
ax2.set_title('(b) CR N+1')
#ax2.invert_xaxis()
ax2.invert_yaxis()
#ax2.get_yaxis().set_ticklabels([])
#ax2.yaxis.tick_right()

ax3 = fig.add_subplot(gs[1, 1])
# ax3.set_aspect(1)
#img = ax3.matshow(paths[1:, 1:])
pc = ax3.pcolormesh(t1, t2, np.transpose((paths[1:, 1:])), shading='flat',
                    cmap = 'cool', edgecolor = 'face')
ax3.set_xlabel('Time [days]')
#ax3.get_yaxis().set_ticklabels([])
ax3.plot([0,t1[-1]],[0,t2[-1]],'w')
ax3.set_title('(c) Cumulative distance matrix')
ax3.invert_yaxis()

# ax3.grid(which='major', color='w', linestyle='-', linewidth=0)
# ax3.set_axis_off()
py, px = zip(*p)
for i in range(0,len(px)):
    ax3.plot(t1[py[i]], t2[px[i]], ".-", color="k")
# ax3.xaxis.set_major_locator(plt.NullLocator())
# ax3.yaxis.set_major_locator(plt.NullLocator())
if shownumbers:
    for r in range(1, paths.shape[0]):
        for c in range(1, paths.shape[1]):
            ax3.text(c - 1, r - 1, "{:.2f}".format(paths[r, c]))

gs.tight_layout(fig, pad=1.0, h_pad=1.0, w_pad=1.0)
# fig.subplots_adjust(hspace=0, wspace=0)

if showlegend:
    # ax4 = fig.add_subplot(gs[0:, 2])
    ax4 = fig.add_axes([0.86, 0.19, 0.03, 0.5])
    cbar = fig.colorbar(pc, cax=ax4)
    cbar.set_label(r'Cumulative distance')

ax = fig.axes


# <codecell> Reconstruct V from sunthetic OMNI using DTW
vgrid_Carr_recon_dtw_tlin = np.ones((nlon_grid,len(time_grid)))*np.nan
vgrid_Carr_recon_dtw_tsin = np.ones((nlon_grid,len(time_grid)))*np.nan


buffertime = 28
for t in range(0,len(time_grid)):
    tfromstart =  time_grid - time_grid[0]
    if tfromstart[t] <28:
        vgrid_Carr_recon_dtw_tlin[:,t] = np.nan
    elif tfromstart[-1] - tfromstart[t] <28:
        vgrid_Carr_recon_dtw_tlin[:,t] = np.nan
    else:
        for nlon in range(0,len(lon_grid)):
            vgrid_Carr_recon_dtw_tlin[nlon,t] = single_spacecraft_DTW(OMNI['mjd'].values, 
                                OMNI['Vsynth_Carr_tlin'].values, 
                                OMNI['Carr_lon'].values,
                                time_grid[t], lon_grid[nlon].value)
            vgrid_Carr_recon_dtw_tsin[nlon,t] = single_spacecraft_DTW(OMNI['mjd'].values, 
                                OMNI['Vsynth_Carr_tsin'].values, 
                                OMNI['Carr_lon'].values,
                                time_grid[t], lon_grid[nlon].value)
            
# <codecell> Convert DTW from Carr to HGI
       
vgrid_HGI_recon_dtw_tlin = np.ones((nlon_grid,len(time_grid)))*np.nan
vgrid_HGI_recon_dtw_tsin = np.ones((nlon_grid,len(time_grid)))*np.nan  

for t in range(0,len(time_grid)):
    delta_phi = omega_sidereal.value * (time_grid[t] - smjd)*daysec.value *u.rad
    
    #add the initial phase offset
    delta_phi = delta_phi + dphi0
          
   
    vlong_t = vgrid_Carr_recon_dtw_tlin[:,t]  
    vgrid_HGI_recon_dtw_tlin[:,t] = np.interp(zerototwopi(lon_grid.value - delta_phi.value),
                           lon_grid.value, vlong_t, period = 2*np.pi)
    vlong_t = vgrid_Carr_recon_dtw_tsin[:,t]  
    vgrid_HGI_recon_dtw_tsin[:,t] = np.interp(zerototwopi(lon_grid.value - delta_phi.value),
                           lon_grid.value, vlong_t, period = 2*np.pi)    

if savenow:
    h5f = h5py.File(savedir + 'vinputs_dtw_HGI.h5', 'w')
    h5f.create_dataset('vgrid_Carr_recon_dtw_tlin', data=vgrid_Carr_recon_dtw_tlin)
    h5f.create_dataset('vgrid_Carr_recon_dtw_tsin', data=vgrid_Carr_recon_dtw_tsin)
    h5f.create_dataset('vgrid_HGI_recon_dtw_tlin', data=vgrid_HGI_recon_dtw_tlin)
    h5f.create_dataset('vgrid_HGI_recon_dtw_tsin', data=vgrid_HGI_recon_dtw_tsin)
    h5f.create_dataset('time_edges', data=time_edges)
    h5f.create_dataset('lon_edges', data=lon_edges)
    h5f.create_dataset('lon_grid', data=lon_grid)
    h5f.create_dataset('time_grid', data=time_grid)
    h5f.close()   

# <codecell> show some example time series
dlong = np.pi # offset from Earth
tstart = 85  #time from start of simulation
tend = tstart + 27

t_start_mjd = tstart + smjd
t_end_mjd = tend + smjd
#find nearest timestep
t_id_start = np.argmin(abs(time_grid- t_start_mjd))
t_id_end = np.argmin(abs(time_grid - t_end_mjd))

tseries_mjd =  np.ones((t_id_end - t_id_start +1))*np.nan
tseries_true_lin =  np.ones((t_id_end - t_id_start +1))*np.nan
tseries_back_lin =  np.ones((t_id_end - t_id_start +1))*np.nan
tseries_both_lin =  np.ones((t_id_end - t_id_start +1))*np.nan
tseries_dtw_lin =  np.ones((t_id_end - t_id_start +1))*np.nan
tseries_true_sin =  np.ones((t_id_end - t_id_start +1))*np.nan
tseries_back_sin =  np.ones((t_id_end - t_id_start +1))*np.nan
tseries_both_sin =  np.ones((t_id_end - t_id_start +1))*np.nan
tseries_dtw_sin =  np.ones((t_id_end - t_id_start +1))*np.nan
for t in range(0,t_id_end - t_id_start +1):
    tseries_mjd[t] = time_grid[t_id_start +t]
    t_id = np.argmin(abs(OMNI['mjd'] -  tseries_mjd[t]))
    
    #get the required Carr long value
    phi = zerototwopi(OMNI['Carr_lon'][t_id] +dlong)
    
    #interpolate the Carr lon to the required value  - LINEAR       
    tseries_true_lin[t] = np.interp(phi, lon_grid.value, 
                vgrid_Carr_tlin[:,t_id_start +t], period=2*np.pi)
    
    tseries_back_lin[t] = np.interp(phi, lon_grid.value, 
                vgrid_Carr_recon_back_tlin[:,t_id_start +t], period=2*np.pi)
    
    tseries_both_lin[t] = np.interp(phi, lon_grid.value, 
                vgrid_Carr_recon_both_tlin[:,t_id_start +t], period=2*np.pi)
    
    tseries_dtw_lin[t] = np.interp(phi, lon_grid.value, 
                vgrid_Carr_recon_dtw_tlin[:,t_id_start +t], period=2*np.pi)
    
    #NONLINEAR
    tseries_true_sin[t] = np.interp(phi, lon_grid.value, 
                vgrid_Carr_tsin[:,t_id_start +t], period=2*np.pi)
    
    tseries_back_sin[t] = np.interp(phi, lon_grid.value, 
                vgrid_Carr_recon_back_tsin[:,t_id_start +t], period=2*np.pi)
    
    tseries_both_sin[t] = np.interp(phi, lon_grid.value, 
                vgrid_Carr_recon_both_tsin[:,t_id_start +t], period=2*np.pi)
    
    tseries_dtw_sin[t] = np.interp(phi, lon_grid.value, 
                vgrid_Carr_recon_dtw_tsin[:,t_id_start +t], period=2*np.pi)

plt.figure()
ax=plt.subplot(211) 
plt.plot(tseries_mjd - smjd, tseries_true_lin, 'k', label = 'True state')
plt.plot(tseries_mjd - smjd, tseries_back_lin, 'g', label = 'Corotation: back in t')
plt.plot(tseries_mjd - smjd, tseries_both_lin, 'b', label = 'Corotation: smooth in t')
plt.plot(tseries_mjd - smjd, tseries_dtw_lin, 'r--', label = 'DTW')
plt.ylabel('V [km/s]')

anchored_text = AnchoredText("(a)", loc=loc)
ax.add_artist(anchored_text)

plt.title('Linearly evolving solar wind model')
ax.get_xaxis().set_ticklabels([])

ax=plt.subplot(212) 
plt.plot(tseries_mjd - smjd, tseries_true_sin, 'k', label = 'True state')
plt.plot(tseries_mjd - smjd, tseries_back_sin, 'g', label = 'Corotation: back in t')
plt.plot(tseries_mjd - smjd, tseries_both_sin, 'b', label = 'Corotation: smooth in t')
plt.plot(tseries_mjd - smjd, tseries_dtw_sin, 'r--', label = 'DTW')
plt.xlabel('Time [days]')

anchored_text = AnchoredText("(b)", loc=loc)
ax.add_artist(anchored_text)

plt.legend(loc='upper right')
plt.ylabel('V [km/s]')
plt.title('Nonlinearly evolving solar wind model')
# <codecell> Plot OMNI-based reconstructions - Carr long

fig_solutions_linear = plt.figure() 

ax=plt.subplot(411)  
pc = ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, vgrid_Carr_tlin, 
              shading='flat',  edgecolor = 'face', norm=plt.Normalize(300,800))
plt.plot(OMNI['mjd'] - smjd, OMNI['Carr_lon']*180/np.pi,'k.',label='Earth',markersize = 0.5)
plt.title('True (linearly evolving)')
plt.ylabel('Carr long [deg]')
ax.get_xaxis().set_ticklabels([])

anchored_text = AnchoredText("(a)", loc=loc)
ax.add_artist(anchored_text)

 
ax=plt.subplot(412)  
ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, vgrid_Carr_recon_back_tlin, 
              shading='flat', edgecolor = 'face', norm=plt.Normalize(300,800))
plt.title('Corotation: Backward in t')
plt.ylabel('Carr long [deg]')
ax.get_xaxis().set_ticklabels([])

anchored_text = AnchoredText("(b)", loc=loc)
ax.add_artist(anchored_text)

 
ax=plt.subplot(413)  
ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, vgrid_Carr_recon_both_tlin, 
              shading='flat', edgecolor = 'face', norm=plt.Normalize(300,800))
plt.title('Corotation: Smoothing in t')
plt.ylabel('Carr long [deg]')
ax.get_xaxis().set_ticklabels([])

anchored_text = AnchoredText("(c)", loc=loc)
ax.add_artist(anchored_text)

 
ax=plt.subplot(414)  
ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, vgrid_Carr_recon_dtw_tlin, 
              shading='flat', edgecolor = 'face', norm=plt.Normalize(300,800))
plt.title('DTW')
plt.ylabel('Carr long [deg]')
plt.xlabel('Time [days]')

anchored_text = AnchoredText("(d)", loc=loc)
ax.add_artist(anchored_text)

#plt.tight_layout()

plt.subplots_adjust(bottom=0.1, right=0.80, top=0.9)
cax = plt.axes([0.85, 0.1, 0.03, 0.8])
cbar = plt.colorbar(pc, cax=cax)
cbar.set_label(r'V [km/s]')



#pl.dump(fig_solutions_linear, open(savedir + 'V_solutions_linear.pickle','wb'))
#figx = pickle.load(open('V_solutions_linear.pickle', 'rb'))
#figx.show()

#plt.tight_layout()


fig_solutions_nonlinear = plt.figure()
ax=plt.subplot(411)  
pc = ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, vgrid_Carr_tsin, 
                   shading='flat', edgecolor = 'face', norm=plt.Normalize(300,800))

plt.plot(OMNI['mjd'] - smjd, OMNI['Carr_lon']*180/np.pi,'k.',label='Earth',markersize = 0.5)
plt.title('True (nonlinearly evolving)')
ax.get_xaxis().set_ticklabels([])
#ax.get_yaxis().set_ticks([])
plt.ylabel('Carr long [deg]')

anchored_text = AnchoredText("(a)", loc=loc)
ax.add_artist(anchored_text)
 
ax=plt.subplot(412)  
ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, vgrid_Carr_recon_back_tsin, 
              shading='flat', edgecolor = 'face', norm=plt.Normalize(300,800))
plt.title('Corotation: Backward in t')
ax.get_xaxis().set_ticklabels([])
#ax.get_yaxis().set_ticks([])
plt.ylabel('Carr long [deg]')

anchored_text = AnchoredText("(b)", loc=loc)
ax.add_artist(anchored_text)

ax=plt.subplot(413)  
ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, vgrid_Carr_recon_both_tsin, 
              shading='flat', edgecolor = 'face', norm=plt.Normalize(300,800))
plt.title('Corotation: Smoothing in t')
ax.get_xaxis().set_ticklabels([])
#ax.get_yaxis().set_ticks([])
plt.ylabel('Carr long [deg]')

anchored_text = AnchoredText("(c)", loc=loc)
ax.add_artist(anchored_text)

ax=plt.subplot(414)  
ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, vgrid_Carr_recon_dtw_tsin, 
              shading='flat', edgecolor = 'face', norm=plt.Normalize(300,800))
plt.title('DTW')
plt.xlabel('Time [days]')
#ax.get_yaxis().set_ticks([])
plt.ylabel('Carr long [deg]')

anchored_text = AnchoredText("(d)", loc=loc)
ax.add_artist(anchored_text)

plt.subplots_adjust(bottom=0.1, right=0.80, top=0.9)
cax = plt.axes([0.85, 0.1, 0.03, 0.8])
cbar = plt.colorbar(pc, cax=cax)
cbar.set_label(r'V [km/s]')

#pl.dump(fig_solutions_linear, open(savedir + 'V_solutions_nonlinear.pickle','wb'))
#figx = pickle.load(open('V_solutions_nonlinear.pickle', 'rb'))
#figx.show()

#plt.tight_layout()

# <codecell> Plot OMNI-based reconstructions - HGI long

fig_solutions_linear = plt.figure() 

ax=plt.subplot(411)  
pc = ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, vgrid_HGI_tlin, 
              shading='flat',  edgecolor = 'face', norm=plt.Normalize(300,800))
plt.plot(OMNI['mjd'] - smjd, OMNI['Carr_lon']*180/np.pi,'k.',label='Earth',markersize = 0.5)
plt.title('True (linearly evolving)')
plt.ylabel('HGI long [deg]')
ax.get_xaxis().set_ticklabels([])
 
ax=plt.subplot(412)  
ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, vgrid_HGI_recon_back_tlin, 
              shading='flat', edgecolor = 'face', norm=plt.Normalize(300,800))
plt.title('Corotation: Backward in t')
plt.ylabel('HGI long [deg]')
ax.get_xaxis().set_ticklabels([])
 
ax=plt.subplot(413)  
ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, vgrid_HGI_recon_both_tlin, 
              shading='flat', edgecolor = 'face', norm=plt.Normalize(300,800))
plt.title('Corotation: Smoothing in t')
plt.ylabel('HGI long [deg]')
ax.get_xaxis().set_ticklabels([])
 
ax=plt.subplot(414)  
ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, vgrid_HGI_recon_dtw_tlin, 
              shading='flat', edgecolor = 'face', norm=plt.Normalize(300,800))
plt.title('DTW')
plt.ylabel('HGI long [deg]')
plt.xlabel('Time [days]')

plt.subplots_adjust(bottom=0.1, right=0.80, top=0.9)
cax = plt.axes([0.85, 0.1, 0.03, 0.8])
cbar = plt.colorbar(pc, cax=cax)
cbar.set_label(r'V [km/s]')

fig_solutions_nonlinear = plt.figure() 

ax=plt.subplot(411)  
pc = ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, vgrid_HGI_tsin, 
              shading='flat',  edgecolor = 'face', norm=plt.Normalize(300,800))
plt.plot(OMNI['mjd'] - smjd, OMNI['Carr_lon']*180/np.pi,'k.',label='Earth',markersize = 0.5)
plt.title('True (nonlinearly evolving)')
plt.ylabel('HGI long [deg]')
ax.get_xaxis().set_ticklabels([])
 
ax=plt.subplot(412)  
ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, vgrid_HGI_recon_back_tsin, 
              shading='flat', edgecolor = 'face', norm=plt.Normalize(300,800))
plt.title('Corotation: Backward in t')
plt.ylabel('HGI long [deg]')
ax.get_xaxis().set_ticklabels([])
 
ax=plt.subplot(413)  
ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, vgrid_HGI_recon_both_tsin, 
              shading='flat', edgecolor = 'face', norm=plt.Normalize(300,800))
plt.title('Corotation: Smoothing in t')
plt.ylabel('HGI long [deg]')
ax.get_xaxis().set_ticklabels([])
 
ax=plt.subplot(414)  
ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, vgrid_HGI_recon_dtw_tsin, 
              shading='flat', edgecolor = 'face', norm=plt.Normalize(300,800))
plt.title('DTW')
plt.ylabel('HGI long [deg]')
plt.xlabel('Time [days]')

plt.subplots_adjust(bottom=0.1, right=0.80, top=0.9)
cax = plt.axes([0.85, 0.1, 0.03, 0.8])
cbar = plt.colorbar(pc, cax=cax)
cbar.set_label(r'V [km/s]')

# <codecell> Plot OMNI-based reconstruction errors

mpl.rcParams['text.usetex'] = False 

cmax=300

fig_dV_linear = plt.figure() 

ax=plt.subplot(311)  
pc = ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, 
                   vgrid_Carr_recon_back_tlin-vgrid_Carr_tlin, 
              shading='flat', edgecolor = 'face', cmap='RdBu', norm=plt.Normalize(-cmax,cmax))
plt.title(r'$\Delta$V, Corotation: Backward in t - True (linearly evolving)')
plt.ylabel('Carr long [deg]')
ax.get_xaxis().set_ticklabels([])

anchored_text = AnchoredText("(a)", loc=loc)
ax.add_artist(anchored_text)
 
ax=plt.subplot(312)  
ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, 
              vgrid_Carr_recon_both_tlin-vgrid_Carr_tlin, 
              shading='flat', edgecolor = 'face', cmap='RdBu', norm=plt.Normalize(-cmax,cmax))
plt.title(r'$\Delta$V, Corotation: Smoothing in t - True (linearly evolving)')
plt.ylabel('Carr long [deg]')
ax.get_xaxis().set_ticklabels([])

anchored_text = AnchoredText("(b)", loc=loc)
ax.add_artist(anchored_text)
 
ax=plt.subplot(313)  
ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, 
              vgrid_Carr_recon_dtw_tlin-vgrid_Carr_tlin, 
              shading='flat', edgecolor = 'face', cmap='RdBu', norm=plt.Normalize(-cmax,cmax))
plt.title(r'$\Delta$V, DTW - True (linearly evolving)')
plt.ylabel('Carr long [deg]')
plt.xlabel('Time [days]')

anchored_text = AnchoredText("(c)", loc=loc)
ax.add_artist(anchored_text)

plt.subplots_adjust(bottom=0.1, right=0.80, top=0.9)
cax = plt.axes([0.85, 0.1, 0.03, 0.8])
cbar = plt.colorbar(pc, cax=cax)
cbar.set_label(r'$\Delta$V [km/s]')


#pl.dump(fig_dV_linear, open(savedir + 'dV_linear.pickle','wb'))
#figx = pickle.load(open('V_solutions_linear.pickle', 'rb'))
#figx.show()

#plt.tight_layout()

cmax=300

fig_dV_nonlinear = plt.figure() 
ax=plt.subplot(311)  
pc = ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, 
                   vgrid_Carr_recon_back_tsin-vgrid_Carr_tsin, 
              shading='flat', edgecolor = 'face', cmap='RdBu', norm=plt.Normalize(-cmax,cmax))
plt.title(r'$\Delta$V, Corotation: Backward in t - True (nonlinearly evolving)')
plt.ylabel('Carr long [deg]')
ax.get_xaxis().set_ticklabels([])

anchored_text = AnchoredText("(a)", loc=loc)
ax.add_artist(anchored_text)
 
ax=plt.subplot(312)  
ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, 
              vgrid_Carr_recon_both_tsin-vgrid_Carr_tsin, 
              shading='flat', edgecolor = 'face', cmap='RdBu', norm=plt.Normalize(-cmax,cmax))
plt.title(r'$\Delta$V, Corotation: Smoothing in t - True (nonlinearly evolving)')
plt.ylabel('Carr long [deg]')
ax.get_xaxis().set_ticklabels([])

anchored_text = AnchoredText("(b)", loc=loc)
ax.add_artist(anchored_text)
 
ax=plt.subplot(313)  
ax.pcolormesh(time_edges - smjd, lon_edges.value*180/np.pi, 
              vgrid_Carr_recon_dtw_tsin-vgrid_Carr_tsin, 
              shading='flat', edgecolor = 'face', cmap='RdBu', norm=plt.Normalize(-cmax,cmax))
plt.title(r'$\Delta$V, DTW - True (nonlinearly evolving)')
plt.ylabel('Carr long [deg]')
plt.xlabel('Time [days]')

anchored_text = AnchoredText("(c)", loc=loc)
ax.add_artist(anchored_text)

plt.subplots_adjust(bottom=0.1, right=0.80, top=0.9)
cax = plt.axes([0.85, 0.1, 0.03, 0.8])
cbar = plt.colorbar(pc, cax=cax)
cbar.set_label(r'$\Delta$V [km/s]')

#pl.dump(fig_dV_nonlinear, open(savedir + 'dV_nonlinear.pickle','wb'))
#figx = pickle.load(open('V_solutions_linear.pickle', 'rb'))
#figx.show()



# <codecell> Compute some simple metrics

dV_back_lin = (vgrid_Carr_recon_back_tlin[:,27:len(time_grid)-27] 
                - vgrid_Carr_tlin[:,27:len(time_grid)-27])
dV_both_lin = (vgrid_Carr_recon_both_tlin[:,27:len(time_grid)-27] 
                - vgrid_Carr_tlin[:,27:len(time_grid)-27])
dV_dtw_lin = (vgrid_Carr_recon_dtw_tlin[:,27:len(time_grid)-27] 
                - vgrid_Carr_tlin[:,27:len(time_grid)-27])

print('<|dV|>, corotation back in t, linear: ' + str(np.nanmean(abs(dV_back_lin))))
print('<|dV|>, corotation smooth in t, linear: ' + str(np.nanmean(abs(dV_both_lin))))
print('<|dV|>, DTW in t, linear: ' + str(np.nanmean(abs(dV_dtw_lin))))

dV_back_sin = (vgrid_Carr_recon_back_tsin[:,27:len(time_grid)-27] 
                - vgrid_Carr_tsin[:,27:len(time_grid)-27])
dV_both_sin = (vgrid_Carr_recon_both_tsin[:,27:len(time_grid)-27] 
                - vgrid_Carr_tsin[:,27:len(time_grid)-27])
dV_dtw_sin = (vgrid_Carr_recon_dtw_tsin[:,27:len(time_grid)-27] 
                - vgrid_Carr_tsin[:,27:len(time_grid)-27])

print('<|dV|>, corotation back in t, nonlinear: ' + str(np.nanmean(abs(dV_back_sin))))
print('<|dV|>, corotation smooth in t, nonlinear: ' + str(np.nanmean(abs(dV_both_sin))))
print('<|dV|>, DTW in t, nonlinear: ' + str(np.nanmean(abs(dV_dtw_sin))))

