# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 21:07:23 2017

a script to convert GSE to RTN coords
Based upon Matlab legacy code. The original code was scapred from various old 
IDL and fortran routines and hence is very badly commented. But it's tested and 
works.
(Mathew Owens, 23/12/17)

@author: vy902033
"""

import numpy as np


# #converts angles into values periodic between 0 and 2*pi
# def zerototwopi(inputangles):
#     #convert to numpy array, to get length, be able to index, etc.
#     if np.isscalar(inputangles):
#         angles=np.array([inputangles])
#     else:
#         angles=np.array(inputangles)
    

#     #first do the negative values...
#     for i in range(0,angles.size):
#         if (angles[i]<0):
#             npi=np.floor(np.abs(angles[i])/(2*np.pi))
#             angles[i]=angles[i]+(npi+1)*2*np.pi
           
#         if (angles[i]>2*np.pi):
#             npi=np.floor(angles[i]/(2*np.pi))
#             angles[i]=angles[i]-npi*2*np.pi
           
#     return angles

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

#returns the ECLIPTIC longitude (in degrees) of Earth for a given MJD
def earthecliplong(inputMJD):
    
    #convert to numpy array, to get length, be able to index, etc.
    if np.isscalar(inputMJD):
        MJD=np.array([inputMJD])
    else:
        MJD=np.array(inputMJD)
    L=0.0*MJD    
        
    for i in range(0,MJD.size):   
        
        julianday=MJD[i] + 2400000.5
        
        dd=julianday-2451909.5
        ep=0.01670790-0.00000000120*dd
        MM=356.779027+0.98560028*dd
        M=MM*np.pi/180
        
        EEi=0.0
        EEione=0.0
        N=0.0
        delEE=1.0
        
        while ( delEE>1E-10 ):
            EEi=EEione;   
            EEione=EEi-(M-EEi+ep*np.sin(EEi))/(ep*np.cos(EEi)-1)
            N=N+1
            delEE=np.abs(EEione-EEi)
        
        tanv=np.sqrt((1+ep)/(1-ep))*np.tan(EEione/2)
        v=2*np.arctan(tanv);
        vdeg=v*180/np.pi
        
        if vdeg<0:
            vdeg=vdeg+360
        elif vdeg>360:
            interg=np.floor(vdeg/360)
            vdeg=vdeg-360*interg
        
        L[i]=vdeg+282.955505 +0.00004708*dd
        
        if L[i]>360:
            interL=np.floor(L[i]/360)
            L[i]=L[i]-360*interL

    return L


#a functiont o convert from ecliptic to heliogrpahic coordiates. Input and answer in degrees(?)
def eclip2heliograph(inputMJD,ecliplong,ecliplat):
    #convert to numpy array, to get length, be able to index, etc.
    if np.isscalar(inputMJD):
        MJD=np.array([inputMJD])
    else:
        MJD=np.array(inputMJD)
    #create the array to store the answer
    y=np.empty([MJD.size, 2])
      
    for i in range(0,MJD.size):   
        #convert to julian day
        julianday=MJD[i] + 2400000.5
        #ecliptic inclination to heliographic equator
        incl=7.25
        
        #the Earth's ecliptic latitude is zero
        ecliplat=0.0
        ecliplong=earthecliplong(MJD[i])
        
        omega=73 +40/60 +50.25*((julianday-2396760)/365)/3600
        
        ww=(360/25.38)*(julianday-2398220)
        ww=(ww/360 - np.floor(ww/360))*360
        
        sinB=np.sin(ecliplat*np.pi/180)*np.cos(incl*np.pi/180) - \
            np.cos(ecliplat*np.pi/180)*np.sin(incl*np.pi/180)*np.sin((ecliplong-omega)*np.pi/180)
            
        B=np.arcsin(sinB)*180/np.pi
        
        sinWL=np.sin(ecliplat*np.pi/180)*np.sin(incl*np.pi/180) + \
            np.cos(incl*np.pi/180)*np.cos(ecliplat*np.pi/180)*np.sin((ecliplong-omega)*np.pi/180)/np.cos(B*np.pi/180)
        cosWL=np.cos((ecliplong-omega)*np.pi/180)*np.cos(ecliplat*np.pi/180)/np.cos(B*np.pi/180)
        WL=np.arctan2(sinWL,cosWL)*180/np.pi
        
        if WL<0:
            WL=WL+360
        L=WL-ww
        
        if L<0:
            L=L+360
        
        y[i][0]=B
        y[i][1]=L
        
    return y
        

def carringtonlatlong_earth(inputMJD):
     #convert to numpy array, to get length, be able to index, etc.
    if np.isscalar(inputMJD):
        MJD=np.array([inputMJD])
    else:
        MJD=np.array(inputMJD)
    #create the array to store the answer
    y=np.empty([MJD.size, 2])
    
      
    for i in range(0,MJD.size):  
        incl=7.25*np.pi/180
               
        #use the kepler's law function to calculate the longitude
        temp= (eclip2heliograph(MJD[i],earthecliplong(MJD[i]),0)-180)*np.pi/180
        carrlong=zerototwopi(temp[:,1])
        
        #the longitude of the ascending node (from Dusan's coord.pdf document)
        omega=73.666666667 +(0.01395833)*(MJD[i]+3243.72)/365.25
        omega=omega*np.pi/180
        
        #ecliptic longitude....
        n=MJD[i]-51544.5
        g=357.528+0.9856003*n
        g=g*np.pi/180
        L=280.460+0.9856474*n
        phi=L+1.915*np.sin(g)+0.020*np.sin(2*g)
        phi=phi*np.pi/180
        
        #ecliptic longitude of the Sun's central meridian
        theta=np.arctan(np.cos(incl)*np.tan(phi-omega))
        
        if (zerototwopi(phi-omega) < np.pi) and (zerototwopi(theta)<np.pi):
            theta=theta+np.pi
        elif (zerototwopi(phi-omega) > np.pi) and (zerototwopi(theta)>np.pi):
            theta=theta+np.pi
        
        B0=np.arcsin(np.sin(theta)*np.sin(incl))
        
        carrlat=(np.pi/2)+B0
        y[i][0]=carrlat
        y[i][1]=carrlong
    return y

def earth_R(mjd):
    #returns the heliocentric distance of Earth (in km). Based on Franz+Harper2002
    AU = 149597870.691
    
    #first up, switch to JD.
    JD=mjd+2400000.5
    d0=JD-2451545
    T0=d0/36525


    L2=100.4664568 + 35999.3728565*T0
    g2=L2-(102.9373481+0.3225654*T0)
    g2=g2*np.pi/180 #mean anomaly

    rAU=1.00014 - 0.01671*np.cos(g2)-0.00014*np.cos(2*g2);
    R=rAU*AU
    
    return R
  
    
#a function to convert GSE data into heliographic RTN
def gse2heliograph(inputMJD,xgse,ygse,zgse):
    
    #convert to numpy array, to get length, be able to index, etc.
    if np.isscalar(inputMJD):
        mjd=np.array([inputMJD])
        x=np.array([xgse])
        y=np.array([ygse])
        z=np.array([zgse])
    else:
        mjd=np.array(inputMJD)
        x=np.array(xgse,dtype=float)
        y=np.array(ygse)
        z=np.array(zgse)
    #create the array to store the answer
    rtn=np.empty([mjd.size, 3])
    
    deltamjd=1; #the incrememnt to use to calculate the gradient in the Earth's latitude.
    
    for i in range(0,mjd.size):  
        #First get the heliographic latitude of the Earth
        temp=carringtonlatlong_earth(mjd[i]); 
        elat=temp[:,0] 
        
        temp=carringtonlatlong_earth(mjd[i]+deltamjd)
        dElat=temp[:,0]
        
        dlong=2*np.pi*deltamjd/365.25;
        
        #use the gradient of the Earth's latitiude to convert the data
        #probs a bit crude, but will do for now (good to within ~degree)....
        gradlat=np.arctan2((elat-dElat),dlong)
        
        htheta=-z[i]*np.cos(gradlat)+y[i]*np.sin(gradlat)
        hphi=-z[i]*np.sin(gradlat)-y[i]*np.cos(gradlat)
        hr=-x[i] #the radial/x-components
        
        rtn[i][0]=hr
        rtn[i][1]=-htheta
        rtn[i][2]=hphi
    
    return rtn

#a function to add rtn V and B to a dataframe
def df_gse2rtn(data):
    df=data.copy()
    
    mjd=df.index.to_julian_date()-2400000.5
    bx=df['Bx_gse']
    by=df['By_gse']
    bz=df['Bz_gse']
    
    vx=df['Vx_gse']
    vy=df['Vx_gse']
    vz=df['Vx_gse']
    
    brtndata=gse2heliograph(mjd,bx,by,bz)
    
    df['Br']=brtndata[:,0]
    df['Bn']=brtndata[:,1]
    df['Bt']=brtndata[:,2]
    
    #df.rename(columns={'Bx_gse': 'Br', 'By_gse': 'Bt','Bz_gse': 'Bn'}, inplace=True)
    
    vrtndata=gse2heliograph(mjd,vx,vy,vz)
    
    df['Vr']=vrtndata[:,0]
    df['Vn']=vrtndata[:,1]
    df['Vt']=vrtndata[:,2]
    
    #df.rename(columns={'vp_x': 'vp_r', 'vp_y': 'vp_t','vp_z': 'vp_n'}, inplace=True)
    
    return df