# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 18:30:50 2020

@author: mathewjowens

A collect of time conversion routes. Mostly ported from Matlab

"""

import numpy as np
import datetime as datetime
import pandas as pd



def date2jd(*args):
    
    """
    date2mjd(year,month,day, *hour, *miunute, *second)
    *optional
    
    Based on Matlab file central code: date2jd
    Mathew Owens, 15/10/20
    
    %   JD = DATE2JD(YEAR, MONTH, DAY, HOUR, MINUTE, SECOND, MICROSECOND) returns the Julian
    %   day number of the given date (Gregorian calendar) plus a fractional part
    %   depending on the time of day.
    %
    %   Start of the JD (Julian day) count is from 0 at 12 noon 1 January -4712
    %   (4713 BC), Julian proleptic calendar.  Note that this day count conforms
    %   with the astronomical convention starting the day at noon, in contrast
    %   with the civil practice where the day starts with midnight.
    %
    %   Astronomers have used the Julian period to assign a unique number to
    %   every day since 1 January 4713 BC.  This is the so-called Julian Day
    %   (JD).  JD 0 designates the 24 hours from noon UTC on 1 January 4713 BC
    %   (Julian proleptic calendar) to noon UTC on 2 January 4713 BC.
    
    %   Sources:  - http://tycho.usno.navy.mil/mjd.html
    %             - The Calendar FAQ (http://www.faqs.org)
    
    %   Author:      Peter J. Acklam
    %   Time-stamp:  2002-05-24 13:30:06 +0200
    %   E-mail:      pjacklam@online.no
    %   URL:         http://home.online.no/~pjacklam
    
    
    """
    
    assert(len(args)>=3)
    year=args[0]
    month=args[1]
    day=args[2]
    
    if isinstance(year,int):
        L=1
    else:
        L=len(year)
    
    #use 00:00:00.0 as default time
    if len(args)>3:
        hour=args[3]
    else:
        hour=np.zeros(L,dtype=int)
    
    if len(args)>4:
        minute=args[4]
    else:
        minute=np.zeros(L,dtype=int)
        
    if len(args)>5:
        second=args[5]
    else:
        second=np.zeros(L,dtype=int)
        
    if len(args)>6:
        microsecond=args[6]
    else:
        microsecond=np.zeros(L,dtype=int)
        
           
    
    #check inputs are integers (except seconds)
    # assert(isinstance(year,int) or isinstance(year[0],np.int32))
    # assert(isinstance(month,int) or isinstance(month[0],np.int32))
    # assert(isinstance(day,int) or isinstance(day[0],np.int32))
    # assert(isinstance(hour,int) or isinstance(hour[0],np.int32))
    # assert(isinstance(minute,int) or isinstance(minute[0],np.int32))
    # assert(isinstance(second,int) or isinstance(second[0],np.int32))
    # assert(isinstance(microsecond,int) or isinstance(microsecond[0],np.int32))
    
    #check the input value ranges
    assert(np.all(month>=0)) 
    assert(np.all(month<=12))
    assert(np.all(day>=1)) 
    assert(np.all(day<=31))
    assert(np.all(hour>=0)) 
    assert(np.all(hour<=23))
    assert(np.all(minute>=0)) 
    assert(np.all(minute<=59))
    assert(np.all(second>=0)) 
    assert(np.all(second<60))
        
    
    a = np.floor((14-month)/12)
    y = year + 4800 - a
    m = month + 12*a -3

    jd = day + np.floor((153*m + 2)/5) + y*365 + np.floor(y/4) - np.floor(y/100) + np.floor(y/400) - 32045 
        
    #add the fractional day part
    jd = jd + (microsecond/1000000  + second + 60*minute + 3600*(hour - 12) )/86400
        
    return jd
 

def date2mjd(*args):
    """
        date2mjd(year,month,day, *hour, *miunute, *second)
        *optional
        
        Convert a date to MJD. Just a wrapper for date2jd
        Mathew Owens, 15/10/20

    """
        
    mjd = date2jd(*args) - 2400000.5
    
    return mjd

def datetime2jd(dt):
    """
    Convert a datetime to JD. Just a wrapper for date2jd
    datetime2mjd(datetime)
    Mathew Owens, 15/10/20

    """ 
    
    #check whether the input is an array of daetimes or a single instance
    if isinstance(dt,np.ndarray):
        year=np.vectorize(lambda x: x.year)(dt)
        month=np.vectorize(lambda x: x.month)(dt)
        day=np.vectorize(lambda x: x.day)(dt)
        hour=np.vectorize(lambda x: x.hour)(dt)
        minute=np.vectorize(lambda x: x.minute)(dt)
        second=np.vectorize(lambda x: x.second)(dt)
        microsecond=np.vectorize(lambda x: x.microsecond)(dt)
        jd=date2jd(year,month,day,hour,minute,second,microsecond)
    elif isinstance(dt,datetime.datetime) or isinstance(dt,pd.core.indexes.datetimes.DatetimeIndex):
        jd=date2jd(dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second,
                   dt.microsecond)
        
    return jd
    

def datetime2mjd(dt):
    """
    Convert a datetime to MJD. Just a wrapper for date2jd
    
    datetime2mjd(datetime)
    
    Mathew Owens, 15/10/20

    """ 
    jd = datetime2jd(dt)
            
    return jd - 2400000.5
    
 
    
def jd2datetime(jd):
    """

    Convert from Julian Day to a datetime object or array of datetimes
    
    Adapted from Matlab code, presumably the companion to date2jd, but can't 
    find original source.
    
    BUG? Seems to gain 16 microseconds, possibly due to numerical roundoff.
    
    (Mathew Owens, 16/10/2020)

    """
    
    #get the integer part of jd
    #Adding 0.5 to JD and taking FLOOR ensures that the date is correct.
    ijd = np.floor(jd + 0.5)
    #get the fractional part
    fjd = jd - ijd + 0.5
    
    a = ijd + 32044
    b = np.floor((4 * a + 3)/146097)
    c = a - np.floor((b*146097) / 4)
    
    
    d = np.floor((4* c + 3)/1461)
    e = c - np.floor((1461*d) /4)
    m = np.floor((5 * e + 2) / 153)
    
    day = e - np.floor((153 * m + 2) / 5) + 1
    month = m + 3 - 12 * np.floor(m/10)
    year = b * 100 + d - 4800 + np.floor(m/10)
    
    hour = np.floor(fjd*24)
    fjd = fjd - hour/24
    minute = np.floor(fjd*60*24)
    fjd = fjd - minute/24/60
    second = np.floor(fjd*24*60*60)
    fjd = fjd - second/24/60/60
    microsecond = np.floor(fjd*24*60*60*1000000)    
    
    #datetime requires integer input
    #check whether the input is an array of daetimes or a single instance
    if isinstance(hour,np.ndarray):
        year=year.astype(int)
        month=month.astype(int)
        day=day.astype(int)
        hour=hour.astype(int)
        minute=minute.astype(int)
        second=second.astype(int)
        microsecond=microsecond.astype(int)
    else:
        year=int(year)
        month=int(month)
        day=int(day)
        hour=int(hour)
        minute=int(minute)
        second=int(second)
        microsecond=int(microsecond)
            
    
    #can't create a datetime array from an array of year, month, etc values
    #make a function and vectorize it
    def date_to_datetime(year,month,day,hour,minute,second,microsecond):
        return datetime.datetime(year,month,day,hour,minute,second,microsecond)
    converttime = np.vectorize(date_to_datetime)
    
    return converttime(year,month,day,hour,minute,second,microsecond)
    
def mjd2datetime(mjd):
    """
    Convert MJD to datetime object. Just a wrapper for jd2datetime
    
    mjd2dateim(mjd)
    
    Mathew Owens, 15/10/20

    """ 
    jd = mjd + 2400000.5
            
    return jd2datetime(jd)

def crnum2mjd(crnum):
    """   
    Converts a Carrington Rotation number to MJD
    Mathew Owens, 16/10/20

    """
    return (crnum - 1750)*27.2753 + 45871.41

def mjd2crnum(mjd):
    """
    Converts MJD to Carrington Rotation number
    Mathew Owens, 16/10/20
    """ 
    return 1750 + ((mjd-45871.41)/27.2753)

def isleapyear(year):
    """
    Tests if "year" is a leap year, returns boolean
    Mathew Owens, 16/10/20
    """
    yearf = np.floor(year/4) - year/4
    return yearf == 0

def mjd2doyyr(mjd):
    """
    Convert mjd to (decimal) day-of-year and year
    Mathew Owens, 16/10/20
    """
   
    #convert to datetime and extract the necessary info
    dt=mjd2datetime(mjd)
    year=np.vectorize(lambda x: x.year)(dt)
    year=year.astype(int)
    doy=np.vectorize(lambda x: x.timetuple().tm_yday)(dt)
    
    #include the fractional part of the day
    doyfrac=mjd-np.floor(mjd)
    doy = doy + doyfrac
    
    return doy, year
       

def doyyr2mjd(doy,yr):
    """
    Converts (decimal) day-of-year and (integer) year to MJD
    Mathew Owens, 16/10/20
    """ 
    
    #create a datetime object at the start of the year, add doy as a delta
    def create_dt(doy,yr):
        dt = datetime.datetime(yr,yr*0+1,yr*0+1) + datetime.timedelta(days = np.floor(doy)-yr*0+1)
        return dt
    vec_create_dt = np.vectorize(create_dt)
    dt = vec_create_dt(doy,yr) 
    
    #convert to mjd
    mjd = datetime2mjd(dt)
    
    #add the fractional part of doy
    mjd = mjd + (doy-np.floor(doy))
    
    return mjd

       
    
        
# <codecell> Testing    
 
# dt = datetime.datetime(2020,5,17,2,40,23,999999)
# jd=datetime2jd(dt)
# print(dt,jd,jd2datetime(jd))

# dt = datetime.datetime(2020,5,17,23,40,23)
# jd=datetime2jd(dt)
# print(dt,jd,jd2datetime(jd))

# dt = datetime.datetime(2020,5,17,0,0,0)
# jd=datetime2jd(dt)
# print(dt,jd,jd2datetime(jd))


# dt = datetime.datetime(2020,5,17,2,40,23,999999)
# mjd=datetime2mjd(dt)
# print(dt,mjd,mjd2datetime(mjd))



# t = np.arange(datetime.datetime(1985,7,1,4,5,6), datetime.datetime(2085,7,4,4,5,6), 
#               datetime.timedelta(days=1)).astype(datetime.datetime)
# jd=datetime2jd(t)
# mjd=datetime2mjd(t)
# print(t,mjd,mjd2datetime(mjd))

    
# jd=date2jd(year,month,day,hour,minute,second)
# print(jd)

# jd=date2jd(year,month,day)
# print(jd)

# mjd=date2mjd(year,month,day,hour,minute,second)
# print(mjd)
