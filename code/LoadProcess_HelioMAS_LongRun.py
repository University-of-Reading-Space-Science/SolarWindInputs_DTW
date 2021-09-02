# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 15:15:46 2021

@author: mathewjowens
"""

#a script to read in and process the output of Linker et al., 2016 for use with DTW
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import pandas as pd
import urllib
from pyhdf.SD import SD, SDC 
import h5py
from dtaidistance import dtw

import helio_time as htime
import helio_coords as hcoord

savedir =  '..\\output\\'
hmasdir =  'D:\\data\\HelioMAS_longrun\\'
datadir =  '..\\data\\'

#download a file
downloadnow = False
urlbase = 'http://predsci.com/~pete/research/matt/matt-1-year-eq-slices/'
varlist = ['bp','bt','br','vp', 'vt','vr','rho', 't']
filetype = '.hdf'
nmax = 1457
nlon = 360
smjd = 52852

savenow = False
# <codecell> Download and compile the data into a single file

if downloadnow:
    for var in varlist:
        for n in range(1,nmax+1):

            if n<10:
                nstr = '00000' + str(n)
            elif n<100:
                nstr = '0000' + str(n)
            elif n<1000:
                nstr = '000' + str(n)    
            else:
                nstr = '00' + str(n)
            
            filename =    var + '_1D_' + nstr + filetype
     
            urllib.request.urlretrieve(urlbase + filename, hmasdir + filename)

#compile slices into single file
h5f = h5py.File(savedir + 'vinputs_hmas_HGI.h5', 'w')
for var in varlist:
    exec(var + ' = np.ones((nmax,nlon-1))*np.nan')    
    for n in range(1,nmax+1): #loop through each time step
 
        if n<10:
            nstr = '00000' + str(n)
        elif n<100:
            nstr = '0000' + str(n)
        elif n<1000:
            nstr = '000' + str(n)    
        else:
            nstr = '00' + str(n)
        
        filename =    var + '_1D_' + nstr + filetype
        
        #read the file
        file = SD(hmasdir + filename, SDC.READ)
        sds_obj = file.select('fakeDim0')  # select sds
        phi = sds_obj.get()  # get sds data
        
        sds_obj = file.select('Data-Set-2')  # select sds
        data = sds_obj.get()  # get sds data
        
        #apply unit conversions
        if ((var == 'vr') or (var =='vp') or (var == 'vt')):
            data = data *481
        #store the data in a variable. HelioMAS 0 and 360 is duplicated
        exec(var + '[n-1,:] = data[0:nlon-1]')
    #nlon_var = len(phi)
    #print(var + '; ' +str(len(data)))
    #print(var + '(phi); ' +str(len(phi)))
    


        
       
    #create the time and lon grid
    lon_edges = np.arange(0,360.00001,360/359) * np.pi/180
    lon_grid = phi[0:nlon-1]
    
    dt = 365/nmax
    time_grid = smjd + np.arange(dt/2,365.0001 - dt/2,dt)
    time_edges = smjd + np.arange(0,365.0001,dt)
    
    #add the data to a file
    exec( 'h5f.create_dataset("vgrid_HGI_hmas_' + var + '", data=' +var +'.T)')

h5f.create_dataset('time_edges', data=time_edges)
h5f.create_dataset('lon_edges', data=lon_edges)
h5f.create_dataset('lon_grid', data=lon_grid)
h5f.create_dataset('time_grid', data=time_grid)
h5f.close()   
                
#plt.pcolor(vr)

# <codecell> 
#load Earth ephemeris

#load the H5 file back in
filepath = savedir + 'vinputs_hmas_HGI.h5'
h5f = h5py.File(filepath,'r')
for var in varlist:
    exec( 'vgrid_HGI_hmas_' + var + '=np.array(h5f["vgrid_HGI_hmas_' + var +'"])')

time_edges = np.array(h5f['time_edges'])
lon_edges = np.array(h5f['lon_edges'])
time_grid = np.array(h5f['time_grid'])
lon_grid = np.array(h5f['lon_grid'])
h5f.close()


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



starttime = htime.mjd2datetime(smjd).item()
endtime = htime.mjd2datetime(smjd + 365).item()
filepath = datadir + 'Earth_HGI.lst'
pos_Earth = pd.read_csv(filepath,
                     skiprows = 1, delim_whitespace=True,
                     names=['year','doy',
                            'rad_au','HGI_lat','HGI_lon'])
#convert to mjd
pos_Earth['mjd'] = htime.doyyr2mjd(pos_Earth['doy'],pos_Earth['year'])


#create OMNI position files
d = {'mjd': time_grid}
OMNI = pd.DataFrame(data=d)

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
fmjd=OMNI['mjd'][nmax-1] 
omega_omni = omega_sidereal - (OMNI_lon_unwrap[-1] - OMNI_lon_unwrap[0])*u.rad/(fmjd-smjd)/daysec

#create the Earth time series
#generate the Earth time series
for var in varlist:
    exec('OMNI["synth_' + var +'"] = np.nan')
#for t in range(0,len(OMNI['mjd'])):
#    vlong_t = vgrid_HGI_hmas_vr[t,:]   
#    OMNI['synth_vr'][t] =  np.interp(OMNI['HGI_lon'][t], 
#                                       lon_grid,vlong_t, period = 2*np.pi)
for t in range(0,len(OMNI['mjd'])):
    for var in varlist:
        #extract the longitudinal slice at the correct time
        #    vlong_t = vgrid_HGI_hmas_vr[t,:]  
        exec('long_t = vgrid_HGI_hmas_' + var + '[:,t]')   
        
        #interpolate the value at the Earth longitude
        #    OMNI['synth_vr'][t] =  np.interp(OMNI['HGI_lon'][t], 
        #                            lon_grid,vlong_t, period = 2*np.pi)
        exec('OMNI["synth_' + var + '"][t] =  np.interp(OMNI["HGI_lon"][t],'
             +'lon_grid,long_t, period = 2*np.pi)')
        
# <codecell> reconstruct parameters from time series using corotation       
nlon_grid = len(lon_grid)

#vgrid_Carr_recon_back_hmas = np.ones((nlon_grid,len(time_grid)))*np.nan
#vgrid_Carr_recon_forward_hmas = np.ones((nlon_grid,len(time_grid)))*np.nan
#vgrid_Carr_recon_both_hmas = np.ones((nlon_grid,len(time_grid)))*np.nan
#for t in range(0,len(time_grid)):
#    #find nearest time and current Carrington longitude
#    t_id = np.argmin(np.abs(OMNI['mjd'] - time_grid[t]))
#    Elong = OMNI['Carr_lon'][t_id]*u.rad
#    
#    #get the Carrington longitude difference from current Earth pos
#    dlong_back = zerototwopi(lon_grid.value - Elong.value) *u.rad
#    dlong_forward = zerototwopi(Elong.value- lon_grid.value) *u.rad
#    
#    dt_back = (dlong_back / (omega_omni)).to(u.day)
#    dt_forward = (dlong_forward / (omega_omni)).to(u.day)
#    
#    vgrid_Carr_recon_back_hmas[:,t] = np.interp(time_grid[t] - dt_back.value, 
#                                 OMNI['mjd'], 
#                                 OMNI['synth_vr'], left = np.nan, right = np.nan)
#    vgrid_Carr_recon_forward_hmas[:,t] = np.interp(time_grid[t] + dt_forward.value, 
#                                 OMNI['mjd'], 
#                                 OMNI['Vsynth_vr'], left = np.nan, right = np.nan)
#    vgrid_Carr_recon_both_hmas[:,t] = (dt_forward * vgrid_Carr_recon_back_hmas[:,t] +
#                             dt_back * vgrid_Carr_recon_forward_hmas[:,t])/(dt_forward + dt_back)

for var in varlist:  
    #create the variables
    #vgrid_HGI_recon_back_hmas = np.ones((nlon_grid,len(time_grid)))*np.nan
    exec('vgrid_HGI_recon_back_hmas_' + var + '= np.ones((nlon_grid,len(time_grid)))*np.nan')
    exec('vgrid_HGI_recon_forward_hmas_' + var +' = np.ones((nlon_grid,len(time_grid)))*np.nan')
    exec('vgrid_HGI_recon_both_hmas_' +var +' = np.ones((nlon_grid,len(time_grid)))*np.nan')
    
    for t in range(0,len(time_grid)):
        #find nearest time and current Carrington longitude
        t_id = np.argmin(np.abs(OMNI['mjd'] - time_grid[t]))
        Elong = OMNI['HGI_lon'][t_id]*u.rad
        
        #get the HGI longitude difference from current Earth pos
        dlong_back = zerototwopi(lon_grid - Elong.value) *u.rad
        dlong_forward = zerototwopi(Elong.value- lon_grid) *u.rad
        
        dt_back = (dlong_back / (omega_omni)).to(u.day)
        dt_forward = (dlong_forward / (omega_omni)).to(u.day)
        
        
#        vgrid_HGI_recon_back_hmas[:,t] = np.interp(time_grid[t] - dt_back.value, 
#                                     OMNI['mjd'], 
#                                     OMNI['synth_vr'], left = np.nan, right = np.nan)
        exec('vgrid_HGI_recon_back_hmas_' +var +'[:,t] = np.interp(time_grid[t] - '
             +'dt_back.value, OMNI["mjd"], OMNI["synth_' +var+'"], left = np.nan, right = np.nan)')
        exec('vgrid_HGI_recon_forward_hmas_' + var + '[:,t] = np.interp(time_grid[t] '
             +'+ dt_forward.value, OMNI["mjd"], OMNI["synth_'+var+'"], left = np.nan, right = np.nan)')
        exec('vgrid_HGI_recon_both_hmas_' +var + '[:,t] = (dt_forward * vgrid_HGI_recon_back_hmas[:,t]'
             +' +dt_back * vgrid_HGI_recon_forward_hmas[:,t])/(dt_forward + dt_back)')


    
#save the HGI data for use with HUXt
if savenow:
    h5f = h5py.File(savedir + 'vinputs_corot_HGI_hmas.h5', 'w')
    h5f.create_dataset('vgrid_HGI_recon_back_hmas', data=vgrid_HGI_recon_back_hmas)
    h5f.create_dataset('vgrid_HGI_recon_both_hmas', data=vgrid_HGI_recon_both_hmas)
    h5f.create_dataset('time_edges', data=time_edges)
    h5f.create_dataset('lon_edges', data=lon_edges)
    h5f.create_dataset('lon_grid', data=lon_grid)
    h5f.create_dataset('time_grid', data=time_grid)
    h5f.close()       

# <codecell> Plot it

fig_solutions_linear = plt.figure() 

ax=plt.subplot(411)  
pc = ax.pcolormesh(time_edges - smjd, lon_edges*180/np.pi, vgrid_HGI_hmas_vr, 
              shading='flat',  edgecolor = 'face', norm=plt.Normalize(300,800))
plt.plot(OMNI['mjd'] - smjd, OMNI['HGI_lon']*180/np.pi,'k.',label='Earth',markersize = 0.5)
plt.title('True (helioMAS)')
plt.ylabel('HGI long [deg]')
ax.get_xaxis().set_ticklabels([])
 
ax=plt.subplot(412)  
ax.pcolormesh(time_edges - smjd, lon_edges*180/np.pi, vgrid_HGI_recon_back_hmas, 
              shading='flat', edgecolor = 'face', norm=plt.Normalize(300,800))
plt.title('Corotation: Backward in t')
plt.ylabel('HGI long [deg]')
ax.get_xaxis().set_ticklabels([])
 
ax=plt.subplot(413)  
ax.pcolormesh(time_edges - smjd, lon_edges*180/np.pi, vgrid_HGI_recon_both_hmas, 
              shading='flat', edgecolor = 'face', norm=plt.Normalize(300,800))
plt.title('Corotation: Smoothing in t')
plt.ylabel('HGI long [deg]')
ax.get_xaxis().set_ticklabels([])
 
ax=plt.subplot(414)  
ax.pcolormesh(time_edges - smjd, lon_edges*180/np.pi, vgrid_HGI_recon_back_hmas, 
              shading='flat', edgecolor = 'face', norm=plt.Normalize(300,800))
plt.title('Back')
plt.ylabel('HGI long [deg]')
plt.xlabel('Time [days]')

plt.subplots_adjust(bottom=0.1, right=0.80, top=0.9)
cax = plt.axes([0.85, 0.1, 0.03, 0.8])
cbar = plt.colorbar(pc, cax=cax)
cbar.set_label(r'V [km/s]')


# <codecell> convert to Carrington longitude, mainly for plotting

#the initial phase difference between Carr and HGI longitudes
dphi0 = (OMNI['HGI_lon'][0]*u.rad - OMNI['Carr_lon'][0]*u.rad)


nlon = 128
dphi = 360/nlon
lon_Carr = np.arange(dphi/2,360.1-dphi/2,dphi) *u.deg
lon_Carr = lon_Carr.to(u.rad)
for var in varlist:
    #create the variables
    #vgrid_Carr_hmas_vr = np.ones((nlon,len(time_grid)))*np.nan
    exec('vgrid_Carr_hmas_' +var +' = np.ones((nlon,len(time_grid)))*np.nan')
    exec('vgrid_Carr_recon_back_hmas_' +var +' = np.ones((nlon,len(time_grid)))*np.nan')
    exec('vgrid_Carr_recon_both_hmas_' +var +' = np.ones((nlon,len(time_grid)))*np.nan')
    
    for t in range(0,len(time_grid)):
        delta_phi = omega_sidereal.value * (time_grid[t] - smjd)*daysec.value *u.rad
        
        #add the initial phase offset
        delta_phi = delta_phi - dphi0
              
        #time evolving solar wind speed
        t_day = time_grid[t] - smjd
        
        #vlong = vgrid_HGI_hmas_vr[:,t] 
        exec('vlong = vgrid_HGI_hmas_' + var + '[:,t]')  
        exec('vlong_back = vgrid_HGI_recon_back_hmas_' + var + '[:,t]')
        exec('vlong_both = vgrid_HGI_recon_both_hmas_' + var + '[:,t]') 
        
#        vgrid_Carr_hmas_vr[:,t] = np.interp(zerototwopi(lon_Carr.value + delta_phi.value),
#                               lon_grid, vlong, period = 2*np.pi)
        exec('vgrid_Carr_hmas_' +var +'[:,t] = np.interp(zerototwopi(lon_Carr.value'
             '+ delta_phi.value),  lon_grid, vlong, period = 2*np.pi)')
        exec('vgrid_Carr_recon_back_hmas_' +var +'[:,t] = np.interp(zerototwopi(lon_Carr.value'
             '+ delta_phi.value),  lon_grid, vlong_back, period = 2*np.pi)')
        exec('vgrid_Carr_recon_both_hmas_' +var +'[:,t] = np.interp(zerototwopi(lon_Carr.value'
             '+ delta_phi.value),  lon_grid, vlong_both, period = 2*np.pi)')
  


# <codecell> Reconstruct V from sunthetic OMNI using DTW
def single_spacecraft_DTW(obs_mjd, obs_V, obs_Carrlon, ref_mjd, ref_Carrlon, npoints = 50):
    """
    A function to use dynamic time warping to determine the solar wind speed at 
    a given refererence time (ref_mjd) and Carrington longitude (Carr_lon)
    from a single-spacecraft observation (e.g., OMNI).
    """
    
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
    path = dtw.warping_path(vlong_behind, vlong_ahead)#,
                            #window = int(len(vlong_behind)/5), 
                            #psi = int(len(vlong_behind)/5))
 
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


vgrid_Carr_recon_dtw_hmas_vr = np.ones((nlon,len(time_grid)))*np.nan
buffertime = 28
for t in range(0,len(time_grid)):
    tfromstart =  time_grid - time_grid[0]
    if tfromstart[t] <buffertime:
        vgrid_Carr_recon_hmas_dtw_vr[:,t] = np.nan
    elif tfromstart[-1] - tfromstart[t] <buffertime:
        vgrid_Carr_recon_hmas_dtw_vr[:,t] = np.nan
    else:
        for ilon in range(0,len(lon_Carr)):
            vgrid_Carr_recon_dtw_hmas_vr[ilon,t] = single_spacecraft_DTW(OMNI['mjd'].values, 
                                OMNI['synth_vr'].values, 
                                OMNI['Carr_lon'].values,
                                time_grid[t], lon_Carr[ilon].value)

# <codecell> Plot it
fig_solutions_Carr = plt.figure() 

ax=plt.subplot(411)  
pc = ax.pcolormesh(time_edges - smjd, lon_Carr.value*180/np.pi, vgrid_Carr_hmas_vr, 
              shading='flat',  edgecolor = 'face', norm=plt.Normalize(300,800))
plt.plot(OMNI['mjd'] - smjd, OMNI['Carr_lon']*180/np.pi,'w.',label='Earth',markersize = 0.5)
plt.title('True (helioMAS)')
plt.ylabel('Carr long [deg]')
ax.get_xaxis().set_ticklabels([])
 
ax=plt.subplot(412)  
ax.pcolormesh(time_edges - smjd, lon_Carr.value*180/np.pi, vgrid_Carr_recon_back_hmas_vr, 
              shading='flat', edgecolor = 'face', norm=plt.Normalize(300,800))
plt.title('Corotation: Backward in t')
plt.ylabel('Carr long [deg]')
ax.get_xaxis().set_ticklabels([])
 
ax=plt.subplot(413)  
ax.pcolormesh(time_edges - smjd, lon_Carr.value*180/np.pi, vgrid_Carr_recon_both_hmas_vr, 
              shading='flat', edgecolor = 'face', norm=plt.Normalize(300,800))
plt.title('Corotation: Smoothing in t')
plt.ylabel('Carr long [deg]')
ax.get_xaxis().set_ticklabels([])
 
ax=plt.subplot(414)  
ax.pcolormesh(time_edges - smjd, lon_Carr.value*180/np.pi, vgrid_Carr_recon_dtw_hmas_vr, 
              shading='flat', edgecolor = 'face', norm=plt.Normalize(300,800))
plt.title('DTW')
plt.ylabel('Carr long [deg]')
plt.xlabel('Time [days]')

plt.subplots_adjust(bottom=0.1, right=0.80, top=0.9)
cax = plt.axes([0.85, 0.1, 0.03, 0.8])
cbar = plt.colorbar(pc, cax=cax)
cbar.set_label(r'V [km/s]')