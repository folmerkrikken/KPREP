# -*- coding: utf-8 -*-
#!/usr/bin/env python

from __future__ import unicode_literals
import os, sys, glob, re, pickle, time
import numpy as np
import numpy.ma as ma
import scipy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
from netCDF4 import Dataset
from scipy import stats
from scipy.stats import linregress,pearsonr
from sklearn import  linear_model
from sklearn.preprocessing import Imputer
import urllib2
import zipfile
from SPECS_forecast_v2_tools import *
from cdo import *
cdo = Cdo()
from pyresample import geometry,image, kd_tree
import datetime
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# TODO
# Check climatology period for different datasets.. jonathan had 1981-2010
# Check if there are higher resolution datafiles when choosing higher resolution..

dt = datetime.date.today()
date_list = [dt.year, dt.month, dt.day]
start0 = time.time()

predictands = ["GCEcom","20CRslp","GPCCcom"]
predictands = ["20CRslp","GPCCcom"]
#predictands = ["GISTEMP"]
#predictands = ['GCEcom']

bd = '/nobackup/users/krikken/SPESv2/'
bdid = bd+'inputdata/'
bdp = bd+'plots/'
bdnc = bd+'ncfiles/'

# Load these predictors, this does not mean that these are neceserally used.. see predictorz for those
predictors_1d = ['CO2EQ','NINO34','PDO','AMO','IOD']
predictors_3d = ['PERS','CPREC']
predictors = ['CO2EQ','NINO34','PDO','AMO','IOD','CPREC','PERS','PERS_TREND']

# Select method how to run

# NAMELIST

## Resolution, currently only 25 or 50 is supported..
resolution = 25             # 10, 25 or 50

## Redo full hindcast period and remove original nc output file?
overwrite = True

## Redo a specific month / year?
overwrite_m = False         # Overwrite only the month specified with overw_m and overw_y
overw_m = 5                 # Jan = 1, Feb = 2.. etc
overw_y = 2017

UPDATE = False

## Save a figure with the correlation between predictors and predictand
PLOT_PREDCOR =  True

##
VALIDATION =    True        # Validates and makes figures of predicted values 

DYN_MONAVG = False          # Include the dynamical monthly averaging in the predictors
MLR_PRED = True             # Include the trend of the last 3 month as predictor

FORECAST = True            # Do forecast for given date?
HINDCAST = True            # Validate forecast using leave n-out cross validation?
CROSVAL = True
CAUSAL = False
cv_years = 3                       # Leave n out cross validation

## Validation period is 1961 - current

ens_size =      51
styear  =       1901    # Use data from this year until current
stvalyear =     1961    # Start validation from this year until previous year
endyear =       dt.year
endmonth =      dt.month-1  # -1 as numpy arrays start with 0
tot_time = (dt.year - styear) * 12 + endmonth

# Defining some arrays used for writing labels and loading data
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
monthzz = 'JFMAMJJASONDJFMAMJJASOND'

print 'Data from Jan '+str(styear)+' up to '+str(months[dt.month-2])+' '+str(dt.year)

print 'Predictands = ',predictands
print 'Predictors = ',predictors
print 'Horizontal resolution is ',str(resolution/10.),' degrees'

## Predefine arrays and make lats,lons according to specified resolution

## All url for data downloads



if resolution == 10:
    targetgrid = 'griddes10.txt'        # grid description file used for regridding
    predadata = np.zeros((len(predictands),tot_time,180,360))
    latx = 180; lonx = 360
    latz = np.arange(89.5,-90.,-1.)
    lonz = np.arange(-179.5,180.,1.)
elif resolution == 25:
    targetgrid = 'griddes25.txt'        # grid description file used for regridding
    predadata = np.zeros((len(predictands),tot_time,72,144))
    latx = 72; lonx = 144
    latz = np.arange(88.75,-90.,-2.5)
    lonz = np.arange(-178.75,180.,2.5)
elif resolution == 50:
    targetgrid = 'griddes50.txt'        # grid description file used for regridding
    predadata = np.zeros((len(predictands),tot_time,36,82))
    latx = 36; lonx = 82
    latz = np.arange(87.5,-90.,-5)
    lonz = np.arange(-177.5,180.,5)
    
# ************************************************************************
# Read in predictand data for fitting 
# ************************************************************************
start1 = time.time()
print '-- Read in predictand data for fitting --'

predictorz = []     # Predefine empty array, fill with specified predictors for predictand
#predictorz_1d = []
#predictorz_3d = []


for p,predictand in enumerate(predictands):
    
        
    if predictand == 'GISTEMP':
        predictorz.append(['CO2EQ','NINO34','PDO','AMO','IOD','CPREC','PERS','PERS_TREND'])
        url_gistemp = "https://data.giss.nasa.gov/pub/gistemp/gistemp1200_ERSSTv5.nc.gz"
        #if check_timestap(url_gisstemp):
        #    loaddata(url_gisstemp,bd='inputdata/')
        try: gistemp = cdo.selyear(str(styear)+'/2100',input = '-remapbil,targetgrid/griddes'+str(resolution)+'.txt inputdata/gistemp1200_ERSSTv5.nc',returnMaArray = 'tempanomaly')
        except IOError:
            if check_timestamp(url_gistemp) and UPDATE:
                loaddata(url_gistemp,bd='inputdata/')
            gistemp = cdo.selyear(str(styear)+'/2100',input = '-remapbil,targetgrid/griddes'+str(resolution)+'.txt inputdata/gistemp1200_ERSSTv5.nc',returnMaArray = 'temperature_anomaly')    
        mask = np.sum(gistemp.mask,axis=0) > 500
        gistemp_nm = np.array(gistemp)
        gistemp_nm[np.tile(mask,(gistemp.shape[0],1,1))]=np.nan # Put nans where too little data...
        gistemp_nm[gistemp_nm==32767] = np.nan
        gistemp_anom = anom(gistemp_nm,1980,2010,1901)
        gistemp_anom
        predadata[p,:] = gistemp_anom
        
    
    elif predictand == 'GCEcom':
        ## These predictors are selelected for GCEcom in the first predictor selection step
        predictorz.append(['CO2EQ','NINO34','PDO','AMO','IOD','CPREC','PERS','PERS_TREND'])
        #predictorz_1d.append(['CO2EQ','NINO34','PDO','AMO','IOD'])
        #predictorz_3d.append(['PERS','CPREC'])
        
        url_ersstv5 = "http://climexp.knmi.nl/NCDCData/ersstv5.nc"
        if check_timestamp(url_ersstv5) and UPDATE:
            # Load ERSST, data starts from 1854
            #ersst,ersst_lat,ersst_lon,ersst_time = loaddata(url_ersstv5,var="sst",bd='inputdata/')
            loaddata(url_ersstv5,bd='inputdata/')

        #ersst_rg = cdo.remapbil('targetgrid/griddes'+str(resolution)+'.txt',input = '-selyear,1948/2100 inputdata/ersstv5.nc',returnMaArray = 'sst').squeeze()
        # TODO > if smaller resolution, load GHCN_CAMS 0.5 resolution iso regridding 2.5 resolution
        #ersst_rg[ersst_rg==-1.8] = np.nan 
        #
        url_ghcn_cams = "ftp://ftp.cpc.ncep.noaa.gov/wd51yf/GHCN_CAMS/ghcn_cams_1948_cur_2.5.grb"
        if check_timestamp(url_ghcn_cams) and UPDATE:
            loaddata(url_ghcn_cams,var="var11",bd='inputdata/')
        
        # merge data, first regrid (bilinear) to targetgrid, then use fillmiss to fill in the coastal data with bilinear interpolation using 4 nearest neighbours.. this also fills the complete antarctic which has to be corrected manually.
        gcecom = cdo.fillmiss(input = '-mergegrid -addc,273.15 -selyear,1948/2100 -remapbil,targetgrid/griddes'+str(resolution)+'.txt -setmissval,-999 inputdata/ersstv5.nc -remapbil,targetgrid/griddes'+str(resolution)+'.txt -setmissval,-999 inputdata/ghcn_cams_1948_cur_2.5.nc',returnMaArray = 'sst').squeeze()
        
        #gcecom[:,60:,:][ersst_rg.mask[:gcecom.shape[0],60:,:]] = np.nan
        mask = np.sum((gcecom == 271.35),axis=0)>100.
        mask[61:,:] = True  # Mask antarctic and sea ice of southern ocean
        #gcecom[np.tile(mask,(gcecom.shape[0],1,1))] = np.nan
        #gcecom[gcecom==271.35] = np.nan   # Set all values below ice to nan
        
        #loaddata("http://www-users.york.ac.uk/~kdc3/papers/coverage2013/had4_krig_v2_0_0.nc.gz",ret=False)
        try: hadcrucw = cdo.selyear(str(styear)+'/2100',input = '-remapbil,targetgrid/griddes'+str(resolution)+'.txt inputdata/had4_krig_v2_0_0.nc',returnMaArray = 'temperature_anomaly')
        except IOError:
            if check_timestamp("http://www-users.york.ac.uk/~kdc3/papers/coverage2013/had4_krig_v2_0_0.nc.gz") and UPDATE:
                loaddata("http://www-users.york.ac.uk/~kdc3/papers/coverage2013/had4_krig_v2_0_0.nc.gz",bd='inputdata/')
            hadcrucw = cdo.selyear(str(styear)+'/2100',input = '-remapbil,targetgrid/griddes'+str(resolution)+'.txt inputdata/had4_krig_v2_0_0.nc',returnMaArray = 'temperature_anomaly')
        hadcrucw_anom = anom(hadcrucw,1980,2010,1901)
        gcecom_anom = anom(gcecom,1980,2010,1948)
        #np.save('pickle/clim_gcecom.npy',clim_gcecom)
        com = np.concatenate((hadcrucw_anom[:564+480,:],gcecom_anom[480:]),axis=0)
        gcecom[:,mask] = np.nan
        clim_gcecom = clim(gcecom,1980,2010,1948,keepdims=True)
        # Where no or limited data fill with nans
        #com[np.tile(mask,(com.shape[0],1,1))] = np.nan 
        com[:,mask] = np.nan
        #clim_gcecom[:,mask] = np.nan
        predadata[p,:] = com
        


    elif predictand == 'HadCRU4CW':
        # These predictors are selelected for HadCRU4CW in the first predictor selection step
        predictorz.append(['CO2EQ','NINO34','PDO','AMO','IOD','PERS','CPREC'])
        #predictorz_1d.append(['CO2EQ','NINO34','PDO','AMO','IOD'])
        #predictorz_3d.append(['PERS','CPREC'])
        # TAS anomalies relative to 1961-1990 climatology
        if check_timestamp("http://www-users.york.ac.uk/~kdc3/papers/coverage2013/had4_krig_v2_0_0.nc.gz") and UPDATE:
            loaddata("http://www-users.york.ac.uk/~kdc3/papers/coverage2013/had4_krig_v2_0_0.nc.gz",bd='inputdata/')
        hadcrucw = cdo.selyear(str(styear)+'/2100',input = '-remapbil,targetgrid/griddes'+str(resolution)+'.txt inputdata/had4_krig_v2_0_0.nc',returnMaArray = 'temperature_anomaly')
        #HadSAT,latHadSAT,lonHad SAT = loaddata("predsys/tas_Amon_HadCRUT4_1850-2013_CW_v2_anomaly.nc","temperature_anomaly")
        #predadata[p,:] = hadcrucw[(styear-1850)*12:(endyear-1850)*12+endmonth,:,:]
        predadata[p,:hadcrucw.shape[0],:] = hadcrucw

    elif predictand == 'GPCCcom':
        # These predictors are selelected for GPCCcom in the first predictor selection step
        predictorz.append(['CO2EQ','NINO34','AMO','IOD','PERS'])
        #predictorz_1d.append(['CO2EQ','NINO34',,'AMO','IOD'])
        #predictorz_3d.append(['PERS'])
        # Load GPCC precip data, starts in 1901 to current
        url_gpcccom = "http://climexp.knmi.nl/GPCCData/gpcc_10_combined.nc"
        if check_timestamp(url_gpcccom) and UPDATE:
            loaddata(url_gpcccom,bd='inputdata/') 
        gpccprec = cdo.remapbil('targetgrid/griddes'+str(resolution)+'.txt',input = 'inputdata/gpcc_10_combined.nc',returnMaArray = 'prcp')
        gpccprec = gpccprec[(styear-1901)*12:(endyear-1901)*12+endmonth,:,:]
        gpccprec[gpccprec<=-1000.] = np.nan
        clim_gpcc = clim(gpccprec,1980,2010,styear)
        predadata[p,:] = anom(gpccprec,1980,2010,styear)

    elif predictand == '20CRslp':
        # These predictors are selelected for 20CRslp in the first predictor selection step
        predictorz.append(['CO2EQ','NINO34','PDO','AMO','IOD','CPREC','PERS','PERS_TREND'])
        #predictorz_1d.append(['CO2EQ','NINO34','PDO','AMO','IOD'])
        #predictorz_3d.append(['PERS','CPREC'])
        # Load 20CR (1851-2011)
        if check_timestamp('http://climexp.knmi.nl/20C/prmsl.mon.mean.nc') and UPDATE:
            loaddata('http://climexp.knmi.nl/20C/prmsl.mon.mean.nc',bd='inputdata/')
        slp20cr = cdo.remapbil('targetgrid/griddes'+str(resolution)+'.txt',input = '-selvar,prmsl inputdata/prmsl.mon.mean.nc',returnMaArray = 'prmsl')[(styear-1851)*12:(1948-1851)*12,:,:]
        
        
        ## Load NCEP-NCAR reanalysis SLP (1948-current)
        if check_timestamp('http://climexp.knmi.nl/NCEPNCAR40/slp.mon.mean.nc') and UPDATE:
            loaddata('http://climexp.knmi.nl/NCEPNCAR40/slp.mon.mean.nc',bd='inputdata/')
        slp21cr = cdo.remapbil('targetgrid/griddes'+str(resolution)+'.txt',input = 'inputdata/slp.mon.mean.nc',returnMaArray = 'slp')
        ## Combine both datasets, slp20cr 1901-1947, ncepncar 1948-current and remove climatology
        predadata[p,:] = anom(np.concatenate((slp20cr/100.,slp21cr),axis=0),1980,2010,styear)

    else:
        print 'predictand not yet known.. exiting!'
        sys.exit()
 
 
print '-- Done reading in predictand data for fitting, time = ',str(np.int(time.time()-start1)),' seconds --' 


# ************************************************************************
# Read in predictor data for fitting 
# ************************************************************************
start1 = time.time()
print '-- Read in predictor data for fitting --'
#predodata_1d = np.zeros((len(predictors_1d),predadata.shape[1]))
#predodata_3d = np.zeros((len(predictors_3d),predadata.shape[1],latx,lonx))
#predodata = []
predodata = np.zeros((len(predictors),predadata.shape[1],latx,lonx))

# Load 3d predictors (time,lat,lon)
for i,pred in enumerate(predictors):

    if pred == 'CO2EQ':       # CO2EQ RCP45  [Years,data] - 1765-2500]
        data = urllib2.urlopen("http://climexp.knmi.nl/CDIACData/RCP45_CO2EQ.dat")
        co2eq = np.repeat(np.genfromtxt(data),12,axis=0)  # [Years (1765:2500) : CO2 concentration]
        co2eqs = co2eq[(styear-int(co2eq[0,0]))*12:(endyear-int(co2eq[0,0]))*12+endmonth,:]
        #predodata[i,:] = np.rollaxis(np.tile(co2eqs[:,1],(latx,lonx,1)),2,0)
        predodata[i,:] = co2eqs[:,1][:,np.newaxis,np.newaxis]
        #predodata_1d[i,:] = co2eqs[:,1]
        


    elif pred == 'NINO34':    # NINO34 [years,data] - [1854 - currently]
        data = urllib2.urlopen("http://climexp.knmi.nl/NCDCData/ersst_nino3.4a.dat")
        nino34 = np.genfromtxt(data)  # Maanden (jaar.maand) / nino34
        nino34s = nino34[(styear-1854)*12:(endyear-1854)*12+endmonth,:]
        #predodata[i,:] = np.rollaxis(np.tile(nino34s[:,1],(latx,lonx,1)),2,0)
        predodata[i,:] = nino34s[:,1][:,np.newaxis,np.newaxis]
        #predodata_1d[i,:] = nino34s[:,1]   
     
    elif pred == 'QBO':       # QBO   [years,months,data] - [1501-2300]
        data = urllib2.urlopen("http://climexp.knmi.nl/data/iqbo_30.dat")
        qbo = np.genfromtxt(data)  # Maanden (jaar.maand) / qbo
        qbos = qbo[(styear-int(qbo[0,0]))*12:(endyear-int(qbo[0,0]))*12+endmonth,:]
        #predodata[i,:] = np.rollaxis(np.tile(qbos[:,2],(latx,lonx,1)),2,0)
        #predodata_1d[i,:] = qbos[:,1]
        predodata[i,:] = qbos[:,2][:,np.newaxis,np.newaxis]

    elif pred == 'IOD':       # IOD   [years,jan,feb,...,dec] - [1854-2017]
        data = urllib2.urlopen("http://climexp.knmi.nl/NCDCData/dmi_ersst.dat")
        iod = np.genfromtxt(data)  # Maanden (jaar.maand) / IOD
        iodys = iod[:,0]
        iodda = iod[:,1:].ravel()
        iods = iodda[(styear-int(iodys[0]))*12:(endyear-int(iodys[0]))*12+endmonth]
        #predodata[i,:] = np.rollaxis(np.tile(iods,(latx,lonx,1)),2,0)
        #predodata_1d[i,:] = iods[:,1] 
        predodata[i,:] = iods[:,np.newaxis,np.newaxis]

    elif pred == 'PDO':       # PDO   [years,jan,feb,...,dec]
        data = urllib2.urlopen("http://climexp.knmi.nl/UWData/pdo_ersst.dat")
        pdo = np.genfromtxt(data)  # Maanden (jaar.maand) / PDO 
        pdoys = pdo[:,0]
        pdoda = pdo[:,1:].ravel()
        pdos = pdoda[(styear-int(pdoys[0]))*12:(endyear-int(pdoys[0]))*12+endmonth]
        #predodata[i,:] = np.rollaxis(np.tile(pdos,(latx,lonx,1)),2,0)
        #predodata_1d[i,:] = pdos[:,1] 
        predodata[i,:] = pdos[:,np.newaxis,np.newaxis]

    elif pred == 'AMO':       # AMO   [years,jan,feb,...,dec] - [1854-2017]
        #data = urllib2.urlopen("http://climexp.knmi.nl/data/iamo_ersst_ts.dat")
        data = urllib2.urlopen("http://climexp.knmi.nl/NCDCData/amo_ersst_ts.dat")
        amo = np.genfromtxt(data)  # Maanden (jaar.maand) / AMO
        amoys = amo[:,0]
        amoda = amo[:,1:].ravel()       
        amos = amoda[(styear-int(amoys[0]))*12:(endyear-int(amoys[0]))*12+endmonth]
        #predodata[i,:] = np.rollaxis(np.tile(amos,(latx,lonx,1)),2,0)
        #predodata_1d[i,:] = amos[:,1] 
        predodata[i,:] = amos[:,np.newaxis,np.newaxis]
        
    elif pred == 'SIE_EA': # Sea ice extent Eurasia
        sic1 = cdo.selyear(str(styear)+'/2100',input = 'inputdata/sic_1850-2013.nc',returnMaArray = 'seaice_conc')
        lonz = np.arange(-44.5,315.,1.) # Quicker than loading data from netcdf :-)
        #lon = cdo.selvar('longitude',input = 'inputdata/G10010_SIBT1850_v1.1.nc',returnMaArray = 'longitude')
        area = cdo.selvar('cell_area',input = 'inputdata/area_sic_1x1.nc',returnMaArray = 'cell_area')
        sic2 = cdo.selyear('2014/2100',input='inputdata/conc_n.nc',returnMaArray = 'ice')[1:,:].squeeze()
        # Calculat extent, so assume more than 15% as fully ice covered..
        sic2[sic2.mask] = 0.
        sic2[sic2<0.15] = 0.
        sic2[sic2>0.15] = 1.
        
        sic1[sic1<15.] = 0.   
        sic1[sic1>=15.] = 1.
        
        sic = np.concatenate((sic1,sic2),axis=0)
        
        sia = sic * area[None,:]
        sie_ea = np.nansum(np.nansum(np.roll(sia,-1)[:,:,:180],axis=1),axis=1) / 1.e12
        sie_am = np.nansum(np.nansum(np.roll(sia,-1)[:,:,180:],axis=1),axis=1) / 1.e12
        sie_ea_anom = anom(sie_ea,1980,2010,1901)
        sie_am_anom = anom(sie_am,1980,2010,1901)
        predodata[i,:] = sie_ea_anom[:,np.newaxis,np.newaxis]
        
        
            
        
    #else:
        #print 'predictor: '+pred+' not known.. exiting!'
        #sys.exit()        

# Load 3d predictors (time,lat,lon)
#for i,pred in enumerate(predictors_3d):
    
    elif pred == 'LSST':      # Local SST
        print 'LSST not operational'
        sys.exit()

    elif pred == 'PERS':
        # Fill array later
        predodata[i,:] = np.nan
        
    elif pred == 'PERS_TREND':
        # Fill array later
        predodata[i,:] = np.nan
        
        
    elif pred == 'CPREC':    # Cum precip [time,lat,lon] - 1901 -current
        if 'GPCCcom' in predictands:
            predodata[i,:] = predadata[predictands.index('GPCCcom'),:]
        else:
            gpccprec,gpccprec_lats,gpccprec_lons,gpccprec_time = loaddata("http://climexp.knmi.nl/GPCCData/gpcc_10_combined.nc",var="prcp",bd='inputdata/') 
            if resolution != 10:
                gpccprec = cdo.remapbil('targetgrid/griddes'+str(resolution)+'.txt',input = 'inputdata/gpcc_10_combined.nc',returnMaArray = 'prcp')
            gpccprec = gpccprec[(styear-1901)*12:(endyear-1901)*12+endmonth,:,:]
            gpccprec[gpccprec<-1000.] = np.nan
            predodata[i,:] = anom(gpccprec,1980,2010,styear)
            
    else:
        print 'predictor: '+pred+' not known.. exiting!'
        sys.exit()
        
print '-- Done reading in predictor data for fitting, time = ',str(np.int(time.time()-start1)),' seconds --' 


# Normalize predodata..
#predodata = (predodata - np.nanmean(predodata,axis=1)[:,np.newaxis,:,:]) / np.nanstd(predodata,axis=1)[:,np.newaxis,:,:]

#sys.exit()
# *************************************************************************   
# Now start the predictor selection and the forecasting / hindcasting loop
# *************************************************************************

for p,predictand in enumerate(predictands):
    # Fill persistence predictor with predictand data
    predodata[predictors.index('PERS'),:] = predadata[p,:]
    if predictand != 'GPCCcom':
        predodata[predictors.index('PERS_TREND'),:] = predadata[p,:]
     
    # Only use predictors that are selected using 1st step in selection process
    ps = []
    for pr in predictorz[p]:
        ps.append(predictors.index(pr))
    predodata2 = predodata[ps,:]  
    #ps = []
    #for pr in predictorz_1d[p]:
    #    ps.append(predictors_1d.index(pr))
    #predodata2_1d = predodata_1d[ps,:]  
    
    print 'Predictand: ';predictand
    print 'Predictors: ',predictorz[p]
    #print '3d predictors: ',predictorz_3d[p]
    

    # Try to collect last saved forecast month

    try:    
        datanc = Dataset(bdnc+'pred_v2_'+predictand+'.nc')
        timenc = datanc.variables['time']
        year_nc = num2date(timenc[:][-1],timenc.units).year
        month_nc = num2date(timenc[:][-1],timenc.units).month
        #dr = date_range(num2date(timenc[:][-1],timenc.units),datetime.datetime.today())
        datanc.close()
        print 'last forecast month is: '+str(month_nc)
        mon_range = range(month_nc,endmonth+1)
        print mon_range
    except IOError:  # If file does not exist do full hindcast
        month_nc = 0
        mon_range = range(month_nc,12)
        print 'no previous output, do full hindcast!'

    if FORECAST and not HINDCAST: # Only redo forecast loop
        mon_range = range(12)
        
    if overwrite:
        month_nc = 0
        mon_range = range(month_nc,12)
        if os.path.isfile(bdnc+'pred_v2_'+predictand+'.nc') and HINDCAST:
            os.rename(bdnc+'pred_v2_'+predictand+'.nc',bdnc+'pred_v2_'+predictand+'_moved.nc')
        if os.path.isfile(bdnc+'fit_data/rg_v2_'+predictand+'.nc'):    
            os.rename(bdnc+'fit_data/rg_v2_'+predictand+'.nc',bdnc+'rg_v2_'+predictand+'_moved.nc')
    
    if overwrite_m:
        mon_range = [overw_m]
        
        
    #else:
    #    mon_range = range(month_nc,12)

    # Exit loop if already up to date..
    #if len(dr)==0 and not overwrite and not overwrite_m:
    #    print 'already up to date for ',predictand
    #    continue

    #for m,mon in enumerate(months):
    for m in mon_range:
        
        if m < endmonth+1:
            years = np.arange(1902,dt.year+1)
        else:
            years = np.arange(1902,dt.year)
        
        mo = np.array([m-3,m-2,m-1]) # Select months to calculate seasonal average predictors
        ma = np.array([m+1,m+2,m+3]) # Select months to calculate seasonal average predictors
        print 'prediction month = ',months[m]
        print 'predictor season = ',np.asarray(months+months)[mo]
        print 'predictor season = ',np.asarray(months+months)[ma]
        #print years
        
        # Seasonalize data.. m is the month the forecast is made, predo is 3 months prior to m and preda 3 months after m
        #predo_seas,preda_seas = seazon(predodata2,predadata[p,:],m)
        predo_seas3,preda_seas = seazon(predodata2,predadata[p,:],m)
        if DYN_MONAVG:
            predo_seas1 = seazon(predodata2,predadata[p,:],m,month_avg=1)[0]
            predo_seas5 = seazon(predodata2,predadata[p,:],m,month_avg=5)[0]
        if MLR_PRED: 
            predo_trend = seazon_trend(predodata2,m)    # Predictor trend over last 3 months
            predo_prad = seazon_prad(predodata2,m)      # Predictor data at future timestep (i.e. same time step as predictand)
            # Put PERS_TREND at its correct location in the predictor data
            if predictand != 'GPCCcom':
                predo_seas3[predictorz[p].index('PERS_TREND'),:] = predo_trend[predictorz[p].index('PERS_TREND'),:]
            
        #sys.exit()

        if FORECAST:
            print 'Forecasting mode'
            train = np.ones((predo_seas3.shape[1]),dtype='bool')
            n=1
            train[-1] = False
            test = ~train
            if predictand == 'GPCCcom':
                train[:49] = False
            predo_tr = predo_seas3[:,train,:,:]
            predo_te = predo_seas3[:,test,:,:]
            if MLR_PRED:
                predo_trend_tr = predo_trend[:,train,:,:]
                predo_trend_te = predo_trend[:,test,:,:]
            else: predo_trend_tr = []; predo_trend_te = []
            year = years[test]
            print 'test years: ',year
            
            try: preda_tr = preda_seas[train,:]
            except IndexError: preda_tr = preda_seas[train[:-1],:]
            if MLR_PRED:
                try: predo_prad_tr = predo_prad[:,train,:]
                except IndexError: predo_prad_tr = predo_prad[:,train[:-1],:]
                
            else: predo_prad_tr = []
            


            t0 = time.time()
            regr_loop(predo_tr,predo_te,preda_tr,year,m,False,FORECAST,DYN_MONAVG,MLR_PRED,n,ens_size,latx,lonx,bdnc,predictand,predictorz[p],resolution,latz,lonz,predo_trend_tr=predo_trend_tr,predo_prad_tr=predo_prad_tr,predo_trend_te=predo_trend_te,stvalyear=stvalyear,PLOT_PREDCOR=PLOT_PREDCOR)
            print 'time regr_loop = ',time.time()-t0
        
        if HINDCAST and CAUSAL:
            print 'Hindcasting mode, causal ',str(stvalyear),'-current'
            for y in range(stvalyear,years[-1]):
                train = np.zeros((predo_seas3.shape[1]),dtype='bool')
                train[:np.argmin(np.abs(years-y))] = True

                test = np.zeros((predo_seas3.shape[1]),dtype='bool')
                test[np.argmin(np.abs(years-y))] = True

                #print y
                year = years[test]
                print 'train data is 1901 ... ',years[train][-1]
                print 'test data is : ',year
                predo_tr = predo_seas3[:,train,:,:]
                predo_te = predo_seas3[:,test,:,:]
                #preda_tr = preda_seas[train,:,:]
                try: preda_tr = preda_seas[train,:]
                except IndexError: preda_tr = preda_seas[train[:-1],:]
                try: preda_te = preda_seas[test,:,:]
                except IndexError: preda_te = preda_seas[test[:-1],:]
                print predo_tr.shape,preda_tr.shape
                if predo_tr.shape[1] != preda_tr.shape[0]:
                    sys.exit()
                if MLR_PRED:
                    try: predo_prad_tr = predo_prad[:,train,:]
                    except IndexError: predo_prad_tr = predo_prad[:,train[:-1],:]
                    try: predo_prad_te = predo_prad[:,test,:]
                    except IndexError: predo_prad_te = predo_prad[:,test[:-1],:]
                else: predo_prad_te = []; predo_prad_tr = []
                
                n=1
                regr_loop(predo_tr,predo_te,preda_tr,year,m,HINDCAST,False,DYN_MONAVG,MLR_PRED,n,ens_size,latx,lonx,bdnc,predictand,predictorz[p],resolution,latz,lonz,preda_te=preda_te,predo_trend_tr=predo_trend_tr,predo_prad_tr=predo_prad_tr,predo_trend_te=predo_trend_te,predo_prad_te=predo_prad_te,stvalyear=stvalyear)
                
            
        elif HINDCAST and CROSVAL:
            n = cv_years
            print 'Hindcasting mode, leave ',str(n),' out cross-validation'
            if predo_seas3.shape[1] % n > 0:
                #cvl_range = range(predo_seas3.shape[1]/n+1)
                cvl_range = range((years[-1]-stvalyear)/n+1)
            else:     
                #cvl_range = range(predo_seas3.shape[1]/n)
                cvl_range = range((years[-1]-stvalyear)/n)
            for cvl in cvl_range:
                samesize = predo_seas3.shape[1] == preda_seas.shape[0]
                train = np.ones((predo_seas3.shape[1]),dtype='bool')
                stvalyear_idx = np.argmin(np.abs(years-stvalyear))
                train[cvl*n+stvalyear_idx:cvl*n+n+stvalyear_idx] = False
                test = ~train
                train[-1]=False
                test[-1]=False
                #if predictand == 'GPCCcom':
                #    train[:48] = False
                #if not samesize: test[-2] = False
                if predictand == 'GPCCcom':
                    train[:49] = False
                year = years[test]
                print 'train data is: ',years[train][:5],' ... ',years[train][-5:]
                print 'test data is : ',year
                #print 'test years: ',year
                n=np.sum(test)
                if n == 0:
                    continue
                predo_tr = predo_seas3[:,train,:,:]
                predo_te = predo_seas3[:,test,:,:]
                if MLR_PRED:
                    predo_trend_tr = predo_trend[:,train,:,:]
                    predo_trend_te = predo_trend[:,test,:,:]
                else: predo_trend_tr = [];predo_trend_te = []
                #preda_tr = preda_seas[train,:,:]
                try: preda_tr = preda_seas[train,:]
                except IndexError: preda_tr = preda_seas[train[:-1],:]
                try: preda_te = preda_seas[test,:,:]
                except IndexError: preda_te = preda_seas[test[:-1],:]
                print predo_tr.shape,preda_tr.shape
                if predo_tr.shape[1] != preda_tr.shape[0]:
                    sys.exit()
                if MLR_PRED:
                    try: predo_prad_tr = predo_prad[:,train,:]
                    except IndexError: predo_prad_tr = predo_prad[:,train[:-1],:]
                    try: predo_prad_te = predo_prad[:,test,:]
                    except IndexError: predo_prad_te = predo_prad[:,test[:-1],:]
                else: predo_prad_tr = []; predo_prad_te = []
                t0 = time.time()        
                regr_loop(predo_tr,predo_te,preda_tr,year,m,HINDCAST,False,DYN_MONAVG,MLR_PRED,n,ens_size,latx,lonx,bdnc,predictand,predictorz[p],resolution,latz,lonz,preda_te=preda_te,predo_trend_tr=predo_trend_tr,predo_prad_tr=predo_prad_tr,predo_trend_te=predo_trend_te,predo_prad_te=predo_prad_te,stvalyear=stvalyear)
                print 'regr_loop time: ',time.time()-t0
        elif HINDCAST:
            print 'either CROSVAL or CAUSAL should be set to true'
            
                    
            


    if VALIDATION:
       # predictand = 'GCEcom' 
        print 'Start validation for the last year of all months'
        for p,predictand in enumerate(predictands):
            
            dataset=Dataset(bdnc+'pred_v2_'+predictand+'.nc')
            timenz = dataset.variables['time']
            year_nc = num2date(timenz[:][-1],timenz.units).year
            month_nc = num2date(timenz[:][-1],timenz.units).month
            #timen = num2date(timenz[:],timenz.units)
            pre = dataset.variables[predictand+'_fc'][:]
            obs = dataset.variables[predictand+'_obs'][:]
            ref = dataset.variables[predictand+'_ref'][:]
            co2 = dataset.variables[predictand+'_co2'][:]
            dataset.close()
            
            #mon_range = range(12)
            #mon
        
            
            #monthz,yearz = np.zeros(len(timen)),np.zeros(len(timen))
            #for i,d in enumerate(timen):
            #    monthz[i] = d.month
            #    yearz[i] = d.year
                
            if predictand == 'GCEcom':      
                var = 'Surface air temperature'
                clevz = np.array((-2.,-1.,-0.5,-0.2,0.2,0.5,1.,2.))
                cmap1 = matplotlib.colors.ListedColormap(['#000099','#3355ff','#66aaff','#77ffff','#ffffff','#ffff33','#ffaa00','#ff4400','#cc0022'])
                cmap2 = matplotlib.colors.ListedColormap(['#3355ff','#66aaff','#77ffff','#ffffff','#ffff33','#ffaa00','#ff4400'])
                cmap_under = '#000099'
                cmap_over = '#cc0022'
            elif predictand == 'GPCCcom':   
                var = 'Surface precipitation' 
                clevz = np.array((-200.,-100.,-50.,-20.,20.,50.,100.,200.))
                cmap1 = matplotlib.colors.ListedColormap(['#993300','#cc8800','#ffcc00','#ffee99','#ffffff','#ccff66','#33ff00','#009933','#006666'])
                cmap2 = matplotlib.colors.ListedColormap(['#cc8800','#ffcc00','#ffee99','#ffffff','#ccff66','#33ff00','#009933'])
                cmap_under = '#993300'
                cmap_over = '#006666'
                
            elif predictand == '20CRslp':
                var = 'Mean sea level pressure'
                clevz=np.array((-4.,-2.,-1.,-0.5,0.5,1.,2.,4.))
                cmap1 = matplotlib.colors.ListedColormap(['#000099','#3355ff','#66aaff','#77ffff','#ffffff','#ffff33','#ffaa00','#ff4400','#cc0022'])
                cmap2 = matplotlib.colors.ListedColormap(['#3355ff','#66aaff','#77ffff','#ffffff','#ffff33','#ffaa00','#ff4400'])
                cmap_under = '#000099'
                cmap_over = '#cc0022'

            
             
            #for m in (np.unique(timez[0,:])-1).astype(int):
            for m in [month_nc-1]:
            #for m in range(12):
                    mon = str(m+1)
                    if len(str(mon)) == 1: mon = '0'+mon
                    #year = str(int(yearz[m::12][-1]-y))
                    year = str(year_nc)
                    season = monthzz[m+1:m+4]
                    
                    print 'validation for '+season+' '+year
                    
                    bdpo = bdp+predictand+'/'+str(resolution)+'/'+year+mon+'/'
                    if not os.path.exists(bdpo):
                        os.makedirs(bdpo)
                    
                    pref = pre[m::12,:][-1,:]
                    prem = pre[m::12,:][:-1,:]
                    obsm = obs[m::12,:][:-1,:]
                    refm = ref[m::12,:][:-1,:]
                    co2m = co2[m::12,:][:-1,:]
                    
                    rmse_pred =     f_rmse(prem,obsm,SS=True,ref=refm)
                    crps_pred_co2 = f_crps(prem,obsm,SS=True,ref=co2m)
                    crps_pred =     f_crps(prem,obsm,SS=True,ref=refm)
                    corr_pred,corrp_pred = linregrez(np.nanmean(prem,axis=1),obsm,COR=True)
                    tercile = tercile_category(prem,pref)
                    
                    tmp = np.nanmean(pref,axis=0)
                    posneg = tmp > 0.
                    above = 1.-(np.sum(pref>0,axis=0)/51.)
                    below = 1.-(np.sum(pref<0,axis=0)/51.)
                    sig_ensmean = np.ones_like(crps_pred)
                    sig_ensmean[posneg] = above[posneg]
                    sig_ensmean[~posneg] = below[~posneg]
                    
                    
                    plot_climexp(rmse_pred,
                                 'RMSESS hindcasts, climatology as reference (1961-current)',
                                 'SPECS Empirical Seasonal Forecast: '+var+' ('+season+' '+year+')',
                                 'Ensemble size: 51 | Forecast generation date: '+dt.strftime("%d/%m/%Y")+' | base time: '+months[m]+' '+year,
                                 predictand = predictand,
                                 fname=bdpo+predictand+'_rmsess_'+year+mon+'.png',
                                 clevs = np.array((-0.5,-0.35,-0.2,-0.1,0.1,0.2,0.35,0.5)),
                                 cmap=cmap2,
                                 cmap_under = cmap_under,
                                 cmap_over = cmap_over,
                                 )
                    plot_climexp(crps_pred,
                                 'CRPSS hindcasts, reference: climatology (1961-'+str(year_nc-1)+')',
                                 'SPECS Empirical Seasonal Forecast: '+var+' ('+season+' '+year+')',
                                 'Ensemble size: 51 | Forecast generation date: '+dt.strftime("%d/%m/%Y")+' | base time: '+months[m]+' '+year,
                                 predictand = predictand,
                                 cmap=cmap2,
                                 cmap_under = cmap_under,
                                 cmap_over = cmap_over,
                                 fname=bdpo+predictand+'_crpss_'+year+mon+'.png',
                                 clevs = np.array((-0.5,-0.35,-0.2,-0.1,0.1,0.2,0.35,0.5)),
                                 )    
                    plot_climexp(crps_pred_co2,
                                 'CRPSS hindcasts, reference: hindcasts with only CO2 as predictor (1961-'+str(year_nc-1)+')',
                                 'SPECS Empirical Seasonal Forecast: '+var+' ('+season+' '+year+')',
                                 'Ensemble size: 51 | Forecast generation date: '+dt.strftime("%d/%m/%Y")+' | base time: '+months[m]+' '+year,
                                 predictand = predictand,
                                 cmap = cmap2,
                                 fname=bdpo+predictand+'_crpss_detrended_clim_'+year+mon+'.png',
                                 clevs = np.array((-0.5,-0.35,-0.2,-0.1,0.1,0.2,0.35,0.5)),
                                 cmap_under = cmap_under,
                                 cmap_over = cmap_over,
                                 )    
                    plot_climexp(np.nanmean(pref,axis=0),
                                 'Ensemble mean anomaly (wrt 1980-2010)',
                                 'SPECS Empirical Seasonal Forecast: '+var+' ('+season+' '+year+')',
                                 'Ensemble size: 51 | Forecast generation date: '+dt.strftime("%d/%m/%Y")+'  |  Stippled where NOT significant at 10% level'+' | base time: '+months[m]+' '+year,
                                 sig=sig_ensmean,
                                 cmap=cmap2,
                                 predictand = predictand,
                                 cmap_under = cmap_under,
                                 cmap_over = cmap_over,
                                 fname=bdpo+predictand+'_ensmean_'+year+mon+'.png', 
                                 clevs = clevz,
                                 MEAN=True,
                                 )
                    plot_climexp(corr_pred,
                                 'Correlation between hindcast anomaly and observations (1961-'+str(year_nc-1)+'',
                                 'SPECS Empirical Seasonal Forecast: '+var+' ('+season+' '+year+')',
                                 'Ensemble size: 51 | Forecast generation date: '+dt.strftime("%d/%m/%Y")+'  |  Stippled where signficant at 5% level'+' | base time: '+months[m]+' '+year,
                                 sig = corrp_pred,
                                 predictand = predictand,
                                 fname=bdpo+predictand+'_correlation_'+year+mon+'.png', 
                                 clevs = np.arange(-1.,1.01,0.2),
                                 )
                    plot_climexp(tercile,
                                 'Probabilty (most likely tercile of '+var+'), based on 1961-'+str(year_nc-1)+' hindcasts',
                                 'SPECS Empirical Seasonal Forecast: '+var+' ('+season+' '+year+')',
                                 'Ensemble size: 51 | Forecast generation date: '+dt.strftime("%d/%m/%Y")+' | base time: '+months[m]+' '+year,
                                 cmap = cmap1,
                                 predictand = predictand,
                                 fname=bdpo+predictand+'_tercile_'+year+mon+'.png', 
                                 clevs = np.array((-100,-70,-60,-50,-40,40,50,60,70,100)),
                                 barticks = ['100%','70%','60%', '50%', '40%', '40%', '50%','60%', '70%', '100%'],
                                 TERCILE=True,
                                 )
                                 #plt.annotate('<---- below lower tercile        

                
            
                
    import time        
    print 'Total time taken is: ',np.int((time.time()-start0)//60),' minutes and ',np.int((time.time()-start0)%60), 'seconds'


## Regression loop function
TEST = False
if TEST:
    
    # Data from 1981 onwards..
    # S5 data are note anomalies, hence either make them anomalies or add climatology to our data..
    # First try to add climatology (1980-2010) to our data..
    
    obs_seas = nans_like(predodata[p,-682:,:])
    for m in range(12):
        tmp = seazon_prad(predadata[p,-682:,:]+clim_gcecom[-682:,:],m)
        obs_seas[m::12,:][:tmp.shape[0],:] = tmp
    clim_seas = clim(obs_seas,1980,2010,1961,keepdims=True)    
    
    nc1 = Dataset('ncfiles/pred_v2_GCEcom_causal.nc')  # Baseline forecast
    fc1 = nc1.variables['GCEcom_fc'][:] #+ clim_seas[:,None,:,:]
    ref = nc1.variables['GCEcom_ref'][:] #+ clim_seas[:,None,:,:]
    obs = nc1.variables['GCEcom_obs'][:] #+ clim_seas
    lons = nc1.variables['longitude'][:]
    lats = nc1.variables['latitude'][:]
    nc1.close()

    nc2 = Dataset('ncfiles/pred_v2_GCEcom_l3o.nc') # Fit trend and mean of predictor on predictand
    fc2 = nc2.variables['GCEcom_fc'][:] #+ clim_seas[:,None,:,:]
    ref2 = nc2.variables['GCEcom_ref'][:] #+ clim_seas[:,None,:,:]
    #obs2 = nc2.variables['GCEcom_obs'][:]
    nc2.close()

    nc3 = Dataset('ncfiles/pred_v2_GCEcom.nc') # Fit trend and mean of predictor on predictor (Seems physically better)
    fc3 = nc3.variables['GCEcom_fc'][:] #+ clim_seas[:,None,:,:]
    ref3 = nc3.variables['GCEcom_ref'][:]
    #obs3 = nc3.variables['GCEcom_obs'][:]
    nc3.close()

    #nc4 = Dataset('ncfiles/pred_v2_GCEcom_l3o_mlr_noPERS.nc') # Use trend and mean as separate predictors
    #fc4 = nc4.variables['GCEcom_fc'][:] #+ clim_seas[:,None,:,:]
    #ref4 = nc4.variables['GCEcom_ref'][:]
    #nc4.close()

    #nc5 = Dataset('ncfiles/pred_v2_GCEcom_l3o_mlr_noPERS_ens.nc') # Use trend and mean as separate predictors
    #fc5 = nc5.variables['GCEcom_fc'][:] #+ clim_seas[:,None,:,:]
    #ref4 = nc4.variables['GCEcom_ref'][:]
    #nc5.close()

    #nc5 = Dataset('ncfiles/pred_GCEcom_dmavg.nc') # Use either 1, 3 or 5 monthly average for predictors
    #fc5 = nc5.variables['GCEcom_fc'][:] 
    #nc5.close()

    #nc6 = Dataset('ncfiles/pred_GCEcom_dmavg_trend.nc') # Use either 1, 3 or 5 monthly average for predictors
    #fc6 = nc6.variables['GCEcom_fc'][:] 
    #nc6.close()

    for m in [1,4,7,10]:
        # Load ecmwf data
        s5 = load_ecmwf2(var='t2m',m=m,anom=True)
        #bias = np.nanmean(np.nanmean(s5 - clim_seas[m::12,None,:,:][20:-1,:],axis=0),axis=0)
        
        
        #crps1 = f_crps(fc1[m::12,:][-37:-1,:],obs[m::12,:][-37:-1,:],SS=True,ref=ref[m::12,:][-37:-1,:])
        #crps2 = f_crps(fc2[m::12,:][-37:-1,:],obs[m::12,:][-37:-1,:],SS=True,ref=ref2[m::12,:][-37:-1,:])
        #crps3 = f_crps(fc3[m::12,:][-37:-1,:],obs[m::12,:][-37:-1,:],SS=True,ref=ref2[m::12,:][-37:-1,:])
        #crps4 = f_crps(fc4[m::12,:][-37:-1,:],obs[m::12,:][-37:-1,:],SS=True,ref=ref3[m::12,:][-37:-1,:])
        #crps5 = f_crps(fc5[m::12,:][:-1,:],obs[m::12,:][:-1,:],SS=True,ref=ref[m::12,:][:-1,:])
        #crps6 = f_crps(fc6[m::12,:][:-1,:],obs[m::12,:][:-1,:],SS=True,ref=ref[m::12,:][:-1,:])
        #crps_s5 = f_crps(s5,obs[m::12,:][-37:-1,:],SS=True,ref=ref2[m::12,:][-37:-1,:])
        #crps1_2 = f_crps(fc1[m::12,:][:-1,:],obs[m::12,:][:-1,:],SS=True,ref=fc2[m::12,:][:-1,:])
        #crps1_3 = f_crps(fc1[m::12,:][:-1,:],obs[m::12,:][:-1,:],SS=True,ref=fc3[m::12,:][:-1,:])
        #crps5_4 = f_crps(fc5[m::12,:][:-1,:],obs[m::12,:][:-1,:],SS=True,ref=fc4[m::12,:][:-1,:])
        crps3_s5 = f_crps(fc3[m::12,:][-37:-1,:],obs[m::12,:][-37:-1,:],SS=True,ref=s5[:])
        #crpss5 = f_crps(s5,obs[m::12,:],SS=True,ref=ref3[m::12,:][-37:-1,:])
        #crps2_s5 = f_crps(fc2[m::12,:][-37:-1,:],obs[m::12,:][-37:-1,:],SS=True,ref=s5[:])
        #crps2_1 = f_crps(fc2[m::12,:][:-1,:],obs[m::12,:][59:-1,:],SS=True,ref=fc1[m::12,:][59:-1,:])
        #crps3_2 = f_crps(fc3[m::12,:][:-1,:],obs[m::12,:][59:-1,:],SS=True,ref=fc2[m::12,:][:-1,:])
        cmap2 = matplotlib.colors.ListedColormap(['#3355ff','#66aaff','#77ffff','#ffffff','#ffff33','#ffaa00','#ff4400'])
        cmap_u = '#000099'
        cmap_o = '#cc0022'
        clev1 = np.array((-0.5,-0.35,-0.2,-0.1, 0.1, 0.2,0.35,0.5))
        clev2 = np.array((-0.2,-0.1, -0.05,-0.025,0.025,0.05,0.1, 0.2))
        #plotdata(crps1,lons=lons,lats=lats,clev=clev1,cmap=cmap2,cmap_u=cmap_u,cmap_o=cmap_o,title='causal',fname=str(m)+'crps1.png',PLOT=False,extend=True)
        #plotdata(crps2,lons=lons,lats=lats,clev=clev1,cmap=cmap2,cmap_u=cmap_u,cmap_o=cmap_o,title='leave-3-out',fname=str(m)+'crps2.png',PLOT=False,extend=True)
        #plotdata(crps3,lons=lons,lats=lats,clev=clev1,cmap=cmap2,cmap_u=cmap_u,cmap_o=cmap_o,title='leave-3-out mlr',fname=str(m)+'crps3.png',PLOT=False,extend=True)
        #plotdata(crps4,lons=lons,lats=lats,clev=clev1,cmap=cmap2,cmap_u=cmap_u,cmap_o=cmap_o,title='l3o mlr noPERS',fname=str(m)+'crps4.png',PLOT=False,extend=True)
        #plotdata(crps_s5,lons=lons,lats=lats,clev=clev1,cmap=cmap2,cmap_u=cmap_u,cmap_o=cmap_o,title='s5',fname=str(m)+'crps_s5.png',PLOT=False,extend=True)
        #plotdata(crps6,lons=lons,lats=lats,clev=clev1,cmap=cmap2,cmap_u=cmap_u,cmap_o=cmap_o,title='dmavg_pred',fname=str(m)+'crps6.png',PLOT=False,extend=True)

        #plotdata(crps2_1,lons=lons,lats=lats,clev=clev2,cmap=cmap2,cmap_u=cmap_u,cmap_o=cmap_o,title='crps2_1',fname=str(m)+'crps1_2.png',PLOT=False,extend=True)
        #plotdata(crps3_2,lons=lons,lats=lats,clev=clev2,cmap=cmap2,cmap_u=cmap_u,cmap_o=cmap_o,title='crps3_2',fname=str(m)+'crps2_3.png',PLOT=False,extend=True)
        #plotdata(crps2_s5,lons=lons,lats=lats,clev=clev2,cmap=cmap2,cmap_u=cmap_u,cmap_o=cmap_o,title='crps2_s5',fname=str(m)+'crps2_s5.png',PLOT=False,extend=True)
        plotdata(crps3_s5,lons=lons,lats=lats,clev=clev2,cmap=cmap2,cmap_u=cmap_u,cmap_o=cmap_o,title='crps3_s5',fname=str(m)+'crps3_s5.png',PLOT=False,extend=True)
        #plotdata(crps1_6,lons=lons,lats=lats,clev=clev2,cmap=cmap2,cmap_u=cmap_u,cmap_o=cmap_o,title='crps1_6',fname=str(m)+'crps1_6.png',PLOT=False,extend=True)
        #plotdata(crps5_4,lons=lons,lats=lats,clev=clev2,cmap=cmap2,cmap_u=cmap_u,cmap_o=cmap_o,title='crps5_4',fname=str(m)+'crps5_4.png',PLOT=False,extend=True)



    for m in range(12):
        po,pa = predo_seas3,preda_seas = seazon(predodata[:,:-1,:],predadata[0,:-1,:],m)
        mask = get_boreal_mask()
        
        if po.shape[1] != pa.shape[0]:
            po = po[:,:-1,:]
        
        sie_noco2 = remove_co2(po[8,:-1,:],po[0,:-1,:])
        tas_noco2 = remove_co2(pa[:-1,:],po[0,:-1,:])
        
        
        mask[:,:80]=False
        tas_bor = np.nanmean(tas_noco2[:,mask],axis=1)
        #plot_regr(sie_noco2[:,20,20],tas_bor)
        print m
        plotcor2d(tas_noco2,sie_noco2,CLICK_R=True)
        #plotcor2d(tas_noco2,sie_noco2,CLICK_R=True)

