#
#
#    Updates:
#    v10d:
#       * Debug: E_ORGJ is missing in output.
#       *        correction : ORG ----> ORGJ
#       *        ORG -> ORGJ correction is done also in GFED_FIRE
#       *        Modified GFED files:
#       *        1. EmissionFactors_SAPRC_Summary.txt
#       *        2. EmissionFactors.txt
#       * Debug: In GFED, only OC is given. Its OM equivalent is OM = 1.4 * OC
#       * 
#       * Global Attribute "TITLE" is added for WRF v4 
#
#    v10c:
#       * ECLIPSE v6b : when GFED is used. sector awb is not used.
#       * ECLIPSE v6b : Monthly factors are given by "ECLIPSE_V6a_monthly_pattern.nc"
#       * ECLIPSEinUse = True:
#           1. if ELIPSE V6b data (ECLIPSE_ANTH_VOC and ECLIPSE_ANTH_Other) are used,
#              ECLIPSEinUse must be True.
#       * modified functions :
#           1. AddDict_ECLIPSE_Anth_Other_OnWRF_v2 ('_v2' to be removed later)
#           2. AddDict_ECLIPSE_Anth_VOC_OnWRF_v2   ('_v2' to be removed later)
#           3. AddDict_GFED_Fire_OnWRF
#              a. Argument (ECLIPSEinUse) is added. If True, sector 'AGRI' is not included.
#
#
#    v10b:
#       * "area_common_array" (numpy array) is replaced by "area_common_dict" (dictionary)
#         Info of only relevent cells are stored.
#         "area_common_array" was not ideal for a polar stereographic projection.
#         It stored many emission grid cells,
#         including non-overlapping cells with a relevent WRF gric cell.
#         The number of such irrelevent cells in a polar stereographic projection
#         can be very large.
#       * "area_common_dict" (dictionary) contains only relevent cells.
#       * pickle.dump and pickle.load are used to save "area_common_dict"
#         without losing its data type of dictionary.
#         numpy.save and numpy.load change the data type from dictionary to numpy array.
#       * GFED emissions are added.
#       * format of keyname in emis_dict : <sec>_<spec> ---> <sec>-<<Inventory>>-<spec>
#                                    e.g.) tra_CO ---------> tra-ECLIPSE-CO
#
#

from mpi4py import MPI
import sys,getopt
sys.path.append("/home/onishi/Python3Codes")
import glob
from math import fsum
from math import ceil
from netCDF4 import Dataset
import matplotlib
#matplotlib.use('tkagg')
matplotlib.use('agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
#import coltbls as coltbls
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid import make_axes_locatable
import matplotlib.axes as maxes
from decimal import *
#from speciate_temp import *
import time as tm
#import datetime
import re
from shapely.geometry import Polygon
from calendar import monthrange
#from sidereal import *
from read_REAS import read_REAS
from read_GFED41s import read_GFED41s
import h5py
from lonlat2timezone import *
from tzwhere import tzwhere

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print('rank = ',rank)
print('size = ',size)

def print_rank0(text):
  if rank == 0:
    print(datetime.now(), ' : ', text)

def Create_Hourly_Keyname(keyname,dt):
  year  = dt.year
  month = dt.month
  day   = dt.day
  hour  = dt.hour
  format_string = "{:04d}-{:02d}-{:02d}_{:02d}:00:00"
  date_string   = keyname+'-'+format_string.format(dt.year,dt.month,dt.day,dt.hour)
  return date_string

def Add_EmisData_Dictionary(emis_dict,keyname,emis_wrf, units, \
                            XLON, XLAT, time=None):
  ng_sn_wrf = XLON.shape[0]
  ng_we_wrf = XLAT.shape[1]
  emis_dict[keyname]={}         # Create a dictionary entry with a 'shp_NMVOC'iable name ''shp_NMVOC''
  emis_dict[keyname]['dimensions']={}
  emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
  emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
  emis_dict[keyname]['dimensions']['time']       = 367        # of points in time series> e.g.:12
  emis_dict[keyname]['west_east']={}
  emis_dict[keyname]['west_east']['dtype']='i4'
  emis_dict[keyname]['west_east']['dims' ]=['west_east']
  emis_dict[keyname]['west_east']['units']=''
  emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
  emis_dict[keyname]['south_north']={}
  emis_dict[keyname]['south_north']['dtype']='i4'
  emis_dict[keyname]['south_north']['dims' ]=['south_north']
  emis_dict[keyname]['south_north']['units']=''
  emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
  emis_dict[keyname]['longitude']={}
  emis_dict[keyname]['longitude']['dtype']='f4'
  emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
  emis_dict[keyname]['longitude']['units']='degrees_east'
  emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
  emis_dict[keyname]['latitude']={}
  emis_dict[keyname]['latitude']['dtype']='f4'
  emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
  emis_dict[keyname]['latitude']['units']='degrees_east'
  emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
  if time != None:
    emis_dict[keyname]['time']={}
    if type(time[0]) is datetime:
      emis_dict[keyname]['time']['dtype']='datetime'
    elif type(time[0]) is int:
      emis_dict[keyname]['time']['dtype']='i4'
    emis_dict[keyname]['time']['dims' ]=['time']
    emis_dict[keyname]['time']['data' ]=time[:] # [datetime(2008,1,1),datetime(2008,1,2),....,datetime(2009,1,1)]
  emis_dict[keyname]['voc']={}
  emis_dict[keyname]['voc']['dtype']='f4'
  if time != None:
    emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time']
  else:
    emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
  emis_dict[keyname]['voc']['units']= units
  if time != None:
    emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
  else:
    emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
  
  return emis_dict

def Add_WRFOUT_hourly_dictionary(emis_dict,keyname,dt,emis_data):
  year  = dt.year
  month = dt.month
  day   = dt.day
  hour  = dt.hour
  date  = [dt.year,dt.month,dt.day,dt.hour]
  if rank == 0:
    print(date)
  format_string = "{:04d}-{:02d}-{:02d}_{:02d}:00:00"
  date_string   = format_string.format(dt.year,dt.month,dt.day,dt.hour)

  return 1


def Obtain_factors():
  #===========================================================================
  #
  #                 Parameters used throughout the program
  #
  #---------------------------------------------------------------------------
  # Factors per day. monday to sunday
  
  # DATA STRUCTURE:
  #...daily_factors = [[mon_ene, tue_ene, wed_ene, thu_ene, fri_ene, sat_ene, sun_ene],\
  #                    [mon_dom, tue_dom, wed_dom, thu_dom, fri_dom, sat_dom, sun_dom],\
  #                    [.............................................................],\
  #                    [mon_soi, tue_soi, wed_soi, thu_soi, fri_soi, sat_soi, sun_soi],\
  #                    [mon_vol, tue_vol, wed_vol, thu_vol, fri_vol, sat_vol, sun_vol]]
  #
  daily_factors = {'ene' :[0.15,    0.15,    0.15,    0.15,    0.15,    0.13,    0.12],\
                   'dom' :[0.15,    0.14,    0.14,    0.14,    0.14,    0.14,    0.14],\
                   'ind1':[0.15,    0.15,    0.15,    0.15,    0.15,    0.12,    0.12],\
                   'ind2':[0.15,    0.15,    0.15,    0.15,    0.15,    0.11,    0.11],\
                   'flr' :[0.14,    0.14,    0.14,    0.14,    0.14,    0.14,    0.14],\
                   'slv' :[0.17,    0.17,    0.17,    0.17,    0.17,    0.09,    0.04],\
                   'tra' :[0.15,    0.14,    0.15,    0.15,    0.16,    0.13,    0.13],\
                   'shp' :[0.14,    0.14,    0.14,    0.14,    0.14,    0.14,    0.14],\
                   'wst' :[0.14,    0.14,    0.14,    0.14,    0.14,    0.14,    0.14],\
                   'awb' :[0.14,    0.14,    0.14,    0.14,    0.14,    0.14,    0.14],\
                   'agr' :[0.14,    0.14,    0.14,    0.14,    0.14,    0.14,    0.14],\
                   'oth' :[0.14,    0.14,    0.14,    0.14,    0.14,    0.14,    0.14],\
                   'soi' :[0.14,    0.14,    0.14,    0.14,    0.14,    0.14,    0.14],\
                   'vol' :[0.14,    0.14,    0.14,    0.14,    0.14,    0.14,    0.14]}
  
  # Factors per hour 0:23
  # ene, dom, ind1, ind2, flr, slv, tra, shp, wst, awb, agr, oth, soil
  #
  # DATA STRUCTURE:
  #...hourly_factors = [[00hr_ene, 01hr_ene, 02hr_ene, ...., 22hr_ene, 23hr_ene],\
  #                     [00hr_dom, 01hr_dom, 02hr_dom, ...., 22hr_dom, 23hr_dom],\
  #                     [......................................................],\
  #                     [00hr_soi, 01hr_soi, 02hr_soi, ...., 22hr_soi, 23hr_soi],\
  #                     [00hr_vol, 01hr_vol, 02hr_vol, ...., 22hr_vol, 23hr_vol]]
  #
  hourly_factors = {'ene' :[0.03,0.03,0.03,0.03,0.03,0.04,0.04,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.04,0.04,0.04,0.04,0.04,0.04,0.04],\
                    'dom' :[0.02,0.02,0.02,0.02,0.02,0.03,0.05,0.06,0.06,0.06,0.05,0.05,0.04,0.04,0.04,0.04,0.04,0.05,0.06,0.06,0.06,0.05,0.03,0.02],\
                    'ind1':[0.03,0.03,0.03,0.04,0.04,0.04,0.04,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.04,0.04,0.04,0.04,0.03,0.03,0.03,0.03],\
                    'ind2':[0.03,0.03,0.03,0.04,0.04,0.04,0.04,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.04,0.04,0.04,0.04,0.03,0.03,0.03,0.03],\
                    'flr' :[0.03,0.03,0.03,0.04,0.04,0.04,0.04,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.04,0.04,0.04,0.04,0.03,0.03,0.03,0.03],\
                    'slv' :[0.02,0.01,0.01,0.01,0.01,0.02,0.04,0.05,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.05,0.04,0.04,0.04,0.03,0.03],\
                    'tra' :[0.02,0.00,0.00,0.00,0.01,0.02,0.05,0.07,0.07,0.06,0.05,0.05,0.06,0.06,0.06,0.07,0.08,0.08,0.06,0.04,0.03,0.02,0.02,0.02],\
                    'shp' :[0.02,0.00,0.00,0.00,0.01,0.02,0.05,0.07,0.07,0.06,0.05,0.05,0.06,0.06,0.06,0.07,0.08,0.08,0.06,0.04,0.03,0.02,0.02,0.02],\
                    'wst' :[0.02,0.00,0.00,0.00,0.01,0.02,0.05,0.07,0.07,0.06,0.05,0.05,0.06,0.06,0.06,0.07,0.08,0.08,0.06,0.04,0.03,0.02,0.02,0.02],\
                    'awb' :[0.02,0.02,0.02,0.03,0.03,0.03,0.03,0.04,0.05,0.06,0.06,0.07,0.07,0.07,0.07,0.06,0.05,0.04,0.04,0.03,0.03,0.02,0.02,0.02],\
                    'agr' :[0.02,0.02,0.02,0.03,0.03,0.03,0.03,0.04,0.05,0.06,0.06,0.07,0.07,0.07,0.07,0.06,0.05,0.04,0.04,0.03,0.03,0.02,0.02,0.02],\
                    'oth' :[0.02,0.02,0.02,0.03,0.03,0.03,0.03,0.04,0.05,0.06,0.06,0.07,0.07,0.07,0.07,0.06,0.05,0.04,0.04,0.03,0.03,0.02,0.02,0.02],\
                    'soi' :[0.02,0.02,0.02,0.03,0.03,0.03,0.03,0.04,0.05,0.06,0.06,0.07,0.07,0.07,0.07,0.06,0.05,0.04,0.04,0.03,0.03,0.02,0.02,0.02],\
                    'vol' :[0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04]}
  
  # ene, dom, ind1, ind2, flr, slv, tra, shp, wst, awb, agr, oth, soil, vol
  #
  # DATA STRUCTURE:
  #...monthly_factors = [[jan_ene, feb_ene, mar_ene, ...., nov_ene, dec_ene],\
  #                      [jan_dom, feb_dom, mar_dom, ...., nov_dom, dec_dom],\
  #                      [.................................................],\
  #                      [jan_soi, feb_soi, mar_soi, ...., nov_soi, dec_soi],\
  #                      [jan_vol, feb_vol, mar_vol, ...., nov_vol, dec_vol]]
  #
  monthly_factors = {'ene' :[0.11,    0.1 ,    0.09,    0.09,    0.08,    0.06,    0.06,    0.06,    0.06,    0.08,    0.09,    0.11],\
                     'dom' :[0.17,    0.19,    0.15,    0.1 ,    0.05,    0.03,    0.02,    0.02,    0.02,    0.05,    0.08,    0.13],\
                     'ind1':[0.09,    0.1 ,    0.1 ,    0.09,    0.08,    0.08,    0.07,    0.06,    0.07,    0.08,    0.09,    0.09],\
                     'ind2':[0.08,    0.09,    0.09,    0.09,    0.09,    0.09,    0.08,    0.07,    0.08,    0.09,    0.09,    0.08],\
                     'flr' :[0.08,    0.08,    0.08,    0.08,    0.08,    0.08,    0.08,    0.08,    0.08,    0.08,    0.08,    0.08],\
                     'slv' :[0.08,    0.08,    0.08,    0.09,    0.09,    0.09,    0.08,    0.07,    0.08,    0.09,    0.09,    0.09],\
                     'tra' :[0.08,    0.08,    0.08,    0.09,    0.09,    0.09,    0.08,    0.08,    0.08,    0.09,    0.08,    0.08],\
                     'shp' :[0.07,    0.08,    0.08,    0.09,    0.09,    0.09,    0.09,    0.08,    0.08,    0.09,    0.08,    0.08],\
                     'wst' :[0.08,    0.08,    0.08,    0.08,    0.08,    0.08,    0.08,    0.08,    0.08,    0.08,    0.08,    0.08],\
                     'awb' :[0.01,    0.02,    0.09,    0.13,    0.08,    0.06,    0.05,    0.06,    0.14,    0.19,    0.13,    0.04],\
                     'agr' :[0.01,    0.02,    0.09,    0.13,    0.08,    0.06,    0.05,    0.06,    0.14,    0.19,    0.13,    0.04],\
                     'oth' :[0.01,    0.02,    0.09,    0.13,    0.08,    0.06,    0.05,    0.06,    0.14,    0.19,    0.13,    0.04],\
                     'soi' :[0.01,    0.02,    0.09,    0.13,    0.08,    0.06,    0.05,    0.06,    0.14,    0.19,    0.13,    0.04],\
                     'vol' :[0.08,    0.08,    0.08,    0.08,    0.08,    0.08,    0.08,    0.08,    0.08,    0.08,    0.08,    0.08]}
 
  return hourly_factors, daily_factors, monthly_factors
  
  #------------------------------------------------------------------------------------
  #
  #                   END OF Parameters 
  #
  #====================================================================================




 
def plot_emis_dict(keyname, *itime):
  lons = emis_dict[keyname]['longitude']['data' ]             # <WRF longitude grid>
  lats = emis_dict[keyname]['latitude']['data' ]              # <WRF latitude grid>
  data = emis_dict[keyname]['voc']['data' ]                   # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
  #debug#data = np.array(data)
  #
  if data.ndim == 3:
    itime = itime[0]
    #debug#data = np.squeeze(data[:,:,itime])
    data = data[:,:,itime]
  #
  data[data == 0.0] = np.nan

  lon_min = np.amin(lons)-5
  lon_max = np.amax(lons)+5
  lat_min = np.amin(lats)-5
  lat_max = np.amax(lats)+5
  #
  plt.figure()
  # m = Basemap(projection='robin', lon_0=0, resolution='c')
  m = Basemap(projection='cyl', llcrnrlat=lat_min, urcrnrlat=lat_max, \
              llcrnrlon=lon_min, urcrnrlon=lon_max, resolution='l')
  m.drawcountries()
  m.bluemarble()
  m.drawcoastlines(linewidth=1.0,color='white')
  #m.drawmapboundary(fill_color='aqua')
  m.drawmapboundary()
  m.drawmeridians(np.arange(-180,360, 30))
  m.drawparallels(np.arange( -90, 90, 30))
  
  x, y = m(lons,lats)
  #clev = np.concatenate((np.linspace(0.0001,0.001,10),np.linspace(0.001,0.01,10),np.array([
  cmap = plt.get_cmap('jet')
  #norm = colors.BoundaryNorm(clev,cmap.N)
  #cs   = m.contourf(x,y,data,cmap=cmap,norm=norm, spacing='uniform', levels=clev)
  if 1 == 1:
    cs   = m.contourf(x,y,data,25,cmap=cmap,spacing='uniform',norm=LogNorm())
  else:
    clevs = np.logspace(-8,-2,41)
    ticks = np.logspace(-8,-2,7)
    cs   = m.pcolormesh(x,y,data/365.0/24.0/3600.0,vmin=clevs[0],vmax=clevs[-1],norm=LogNorm())

  cbar = m.colorbar(cs,location='bottom',pad="5%")
  m.plot(x[ 0, :],y[ 0, :],color='yellow')
  m.plot(x[-1, :],y[-1, :],color='yellow')
  m.plot(x[ :, 0],y[ :, 0],color='yellow')
  m.plot(x[ :,-1],y[ :,-1],color='yellow')
  
  title = emis_dict[keyname]['voc']['units']
  cbar.set_label(title)
  plt.title(keyname)
  plt.show()
 




print(getcontext())
#
#----------------------------------------------------------------------------
#   FUNCTION: Surface area of latitude-longitude quadrangle
#   .............................................
#   Input   : lon1 [deg] : lower left longitude 
#           : lat1 [deg] : lower left latitude   
#           : lon2 [deg] : upper right longitude
#           : lat2 [deg] : upper right latitude 
#   Output  : area [m2]  : Surface area
#
#   Const   : R (Earth R.) = 6371 [km] 
#             4 x pi x R^2 = 5.10066e14 [m2]
# 
def areaquad(lon1,lat1,lon2,lat2):
    area = (lon2-lon1)*np.pi/180.0\
          *(np.sin(np.pi/180.0*lat2)-np.sin(np.pi/180.0*lat1))/(4.0*np.pi)
    area =5.10066*1e14*np.fabs(area) 
    return area
#
#-----------------------------------------------------------------------------
def area_rect(X,Y):
    area1 = 0.5*np.fabs((X[1]-X[0])*(Y[2]-Y[1])-(X[2]-X[1])*(Y[1]-Y[0]))
    area2 = 0.5*np.fabs((X[2]-X[0])*(Y[3]-Y[0])-(X[3]-X[0])*(Y[2]-Y[0])) 
    return area1+area2

def area_poly(X,Y):
    lenX = len(X)
    area = 0.0
    for i in np.linspace(1,lenX-3,lenX-3):
      area += \
           0.5*np.fabs((X[i]-X[0])*(Y[i+1]-Y[0])-(X[i+1]-X[0])*(Y[i]-Y[0]))   
    return area

def poly2cw(X,Y,dir):
    if X[0] != X[-1] or Y[0] != Y[-1]:
       if rank == 0:
         print('(X1,Y1) must be equal to (X2,Y2)')
       sys.exit()
    if type(X) != type(np.zeros(0)) or type(Y) != type(np.zeros(0)):
       if rank == 0:
         print('X is not numpy array')
       sys.exit()
    lenX  = len(X)
    np_arr = np.zeros(lenX-2)
    for i in range(lenX)[1:-1]:
      np_temp = 0
      for j in range(lenX)[1:-1]:
        V01 = np.array([X[i],Y[i]])-np.array([X[0],Y[0]])
        V02 = np.array([X[j],Y[j]])-np.array([X[0],Y[0]])
        XP  = np.cross(V01,V02)
        if dir*XP < 0.0:
         np_temp += 1
      np_arr[i-1] = np_temp
    np_ind = [i[0]+1 for i in sorted(enumerate(np_arr), key=lambda x:x[1])]
    np_ind = np.append(0,np_ind)
    np_ind = np.append(np_ind,0)
    X = X[np_ind]
    Y = Y[np_ind]
    return X, Y

def area_common3(X,Y,XX,YY):

    P1 = Polygon([( X[0], Y[0]),( X[1], Y[1]),( X[2], Y[2]),( X[3], Y[3])])
    P2 = Polygon([(XX[0],YY[0]),(XX[1],YY[1]),(XX[2],YY[2]),(XX[3],YY[3])])
    if P1.intersects(P2):
      try:
        area1 = P1.intersection(P2).area
      except:
        print(P1)
        print(P2)
        
        plt.figure
        plt.plot(X, Y)
        plt.plot(XX,YY,color='red')
        plt.savefig('./test.png')  
        sys.exit(1)
      area2 = P1.area
      ### if rank == 0:
      ###   print(area1, area2)
      ###   print(X, Y)
      ###   print(XX,YY)
      return area1/area2
    else:
      return 0.0

def area_common2(X,Y,XX,YY):
    count = 0
    nb    = 10
    if len(X) != 5 or len(Y) != 5:
      if rank == 0:
        print(X)
        print(Y)
      sys.exit() 
    for ii in np.arange(nb):
      for jj in np.arange(nb):
        i = ii+0.5
        j = jj+0.5
        XP = (float(nb-i)*float(nb-j)*X[0]+float(i)*float(nb-j)*X[1]+float(i)*float(j)*X[2]+float(nb-i)*float(j)*X[3])/float(nb*nb)
        YP = (float(nb-i)*float(nb-j)*Y[0]+float(i)*float(nb-j)*Y[1]+float(i)*float(j)*Y[2]+float(nb-i)*float(j)*Y[3])/float(nb*nb)
        xtemp = XP  
        ytemp = YP
        temp1 = np.sign((xtemp-XX[0])*(YY[1]-YY[0])-(XX[1]-XX[0])*(ytemp-YY[0]))
        temp2 = np.sign((xtemp-XX[1])*(YY[2]-YY[1])-(XX[2]-XX[1])*(ytemp-YY[1]))
        temp3 = np.sign((xtemp-XX[2])*(YY[3]-YY[2])-(XX[3]-XX[2])*(ytemp-YY[2]))
        temp4 = np.sign((xtemp-XX[3])*(YY[4]-YY[3])-(XX[4]-XX[3])*(ytemp-YY[3]))
        temp  = temp1+temp2+temp3+temp4
        if temp == 4 or temp == -4:
          count += 1
    return count/float(nb*nb)

def area_common(X,Y,XX,YY):
    X   = np.array(X)
    Y   = np.array(Y)
    XX  = np.array(XX)
    YY  = np.array(YY)

    XXX = np.array([])
    YYY = np.array([])
    XXXold = np.array([])
    YYYold = np.array([])
    ZZZ = np.array([])

    # IF one rectagle is entirely inside the other

    ind = []
    area= 0.0
    for i, xtemp in enumerate(XX[:-1]):
      ytemp = YY[i]
      temp1 = np.sign((xtemp-X[0])*(Y[1]-Y[0])-(X[1]-X[0])*(ytemp-Y[0]))
      temp2 = np.sign((xtemp-X[1])*(Y[2]-Y[1])-(X[2]-X[1])*(ytemp-Y[1]))
      temp3 = np.sign((xtemp-X[2])*(Y[3]-Y[2])-(X[3]-X[2])*(ytemp-Y[2]))
      temp4 = np.sign((xtemp-X[3])*(Y[4]-Y[3])-(X[4]-X[3])*(ytemp-Y[3]))
      temp  = temp1+temp2+temp3+temp4
      if temp == 4 or temp == -4:
        XXXold = np.append(XXXold,xtemp)
        YYYold = np.append(YYYold,ytemp)
        ind.append(i+10)
    
    if len(ind) == 4:
      area  = area_poly(XX,YY)
      return area 

    for i, xtemp in enumerate(X[:-1]):
      ytemp = Y[i]
      temp1 = np.sign((xtemp-XX[0])*(YY[1]-YY[0])-(XX[1]-XX[0])*(ytemp-YY[0]))
      temp2 = np.sign((xtemp-XX[1])*(YY[2]-YY[1])-(XX[2]-XX[1])*(ytemp-YY[1]))
      temp3 = np.sign((xtemp-XX[2])*(YY[3]-YY[2])-(XX[3]-XX[2])*(ytemp-YY[2]))
      temp4 = np.sign((xtemp-XX[3])*(YY[4]-YY[3])-(XX[4]-XX[3])*(ytemp-YY[3]))
      temp  = temp1+temp2+temp3+temp4
      if temp == 4 or temp == -4:
        XXXold = np.append(XXXold,xtemp)
        YYYold = np.append(YYYold,ytemp)
        ind.append(i)
    
    if len(ind) == 4:   
      area = area_poly(X,Y)
      return area

#    if len(ind) == 0:
#      area = 0.0
#      return area
   
#    for i in ind:
    for i in range(4):
    #  if i < 10:
        ip = i+1
        if ip == 5:
          ip = 0
        im = i-1
        if im == -1:
          im = 3
        # between i and ip
        a = X[ip]-X[i]
        b = Y[ip]-Y[i]
        for j in range(4):
          aa = XX[j+1]-XX[j]
          bb = YY[j+1]-YY[j]
          t  = (aa*(Y[i]-YY[j])-bb*(X[i]-XX[j]))
          tt = (a*(YY[j]-Y[i])-b*(XX[j]-X[i]))
          if aa*b-a*bb != 0.0:
            t  = -t/(aa*b-a*bb)
            tt = -tt/(a*bb-aa*b)
          else:
            t  = 10.0
            tt = 10.0
          if t >= 0 and t <= 1 and tt >= 0 and tt <= 1:
            Xnew = a*t+X[i]
            Ynew = b*t+Y[i]
            ind_temp = [iii for iii,xtemp in enumerate(XXX) 
                        if np.fabs(xtemp-Xnew) <  1.e-8 
                       and np.fabs(YYY[iii]-Ynew) < 1.e-8]
            if len(ind_temp) == 0:
              XXX = np.append(XXX,Xnew)
              YYY = np.append(YYY,Ynew)
              ZZZ = np.append(ZZZ,[Xnew,Ynew])
        a = X[im]-X[i]
        b = Y[im]-Y[i]
        for j in range(4):
          aa = XX[j+1]-XX[j]
          bb = YY[j+1]-YY[j]
          t  = (aa*(Y[i]-YY[j])-bb*(X[i]-XX[j]))
          tt = (a*(YY[j]-Y[i])-b*(XX[j]-X[i]))
          if aa*b-a*bb != 0:
            t  = -t/(aa*b-a*bb)
            tt = -tt/(a*bb-aa*b)
          else:
            t  = 10.0
            tt = 10.0
          if t >= 0 and t <= 1 and tt >= 0 and tt <= 1:
            Xnew = a*t+X[i]
            Ynew = b*t+Y[i]
            ind_temp = [iii for iii,xtemp in enumerate(XXX) 
                        if np.fabs(xtemp-Xnew) <  1.e-8 
                       and np.fabs(YYY[iii]-Ynew) < 1.e-8]
            if len(ind_temp) == 0:
              XXX = np.append(XXX,Xnew)
              YYY = np.append(YYY,Ynew)
              ZZZ = np.append(ZZZ,[Xnew,Ynew])

    for ii in range(4):
#      if i >= 10:
#        ii = i-10
        ip = ii+1
        if ip == 5:
          ip = 0
        im = ii-1
        if im == -1:
          im = 3
        # between i and ip
        a = XX[ip]-XX[ii]
        b = YY[ip]-YY[ii]
        for j in range(4):
          #
          aa = X[j+1]-X[j]
          bb = Y[j+1]-Y[j]
          t  = aa*(YY[ii]-Y[j])-bb*(XX[ii]-X[j])
          tt = a*(Y[j]-YY[ii])-b*(X[j]-XX[ii])
          if aa*b-a*bb != 0.0:
            t  = -t/(aa*b-a*bb)
            tt = -tt/(a*bb-aa*b)
          else:
            t  = 10.0
            tt = 10.0
          if t >= 0 and t <= 1 and tt >= 0 and tt <= 1:
            Xnew = a*t+XX[ii]
            Ynew = b*t+YY[ii]
            ind_temp = [iii for iii,xtemp in enumerate(XXX) 
                        if np.fabs(xtemp-Xnew) <  1.e-8 
                       and np.fabs(YYY[iii]-Ynew) < 1.e-8]
            if len(ind_temp) == 0:
              XXX = np.append(XXX,Xnew)
              YYY = np.append(YYY,Ynew)
              ZZZ = np.append(ZZZ,[Xnew,Ynew])
        a = XX[im]-XX[ii]
        b = YY[im]-YY[ii]
        for j in range(4):
          #
          aa = X[j+1]-X[j]
          bb = Y[j+1]-Y[j]
          t  = aa*(YY[ii]-Y[j])-bb*(XX[ii]-X[j])
          tt = a*(Y[j]-YY[ii])-b*(X[j]-XX[ii])
          if aa*b-a*bb != 0.0:
            t  = -t/(aa*b-a*bb)
            tt = -tt/(a*bb-aa*b)
          else:
            t  = 10.0
            tt = 10.0
          if t >= 0 and t <= 1 and tt >= 0 and tt <= 1:
            Xnew = a*t+XX[ii]
            Ynew = b*t+YY[ii]
            ind_temp = [iii for iii,xtemp in enumerate(XXX) 
                        if np.fabs(xtemp-Xnew) <  1.e-8 
                       and np.fabs(YYY[iii]-Ynew) < 1.e-8]
            if len(ind_temp) == 0:
              XXX = np.append(XXX,Xnew)
              YYY = np.append(YYY,Ynew)
              ZZZ = np.append(ZZZ,[Xnew,Ynew])

#    print 'X      = ',X
#    print 'Y      = ',Y
#    print 'XX     = ',XX
#    print 'YY     = ',YY
#    print 'XXX    = ', XXX
#    print 'YYY    = ', YYY
#    print 'XXXold = ',XXXold
#    print 'YYYold = ',YYYold
#    print 'len(XXX) = ',len(XXX)

    if len(XXX) == 0:
      area = 0.0
      return area

    for i,xtemp in enumerate(XXXold):
       XXX = np.append(XXX,xtemp)
       YYY = np.append(YYY,YYYold[i])
    XXX = np.append(XXX,XXX[0])
    YYY = np.append(YYY,YYY[0])

#    print 'X      = ',X
#    print 'Y      = ',Y
#    print 'XX     = ',XX
#    print 'YY     = ',YY
#    print 'XXX    = ', XXX
#    print 'YYY    = ', YYY
#    print 'XXXold = ',XXXold
#    print 'YYYold = ',YYYold
    if len(XXX) != 5:
      area = 0.0
      return area
    XXX, YYY = poly2cw(XXX,YYY,1)
    area = area_poly(XXX,YYY)
    #print area
    return area

#####--- Tests of sub-programs --------------------------      
####X = np.array([ 0.0, 1.0, 2.0, 3.0, 2.5, 1.0, 0.0])
####Y = np.array([ 0.0, 1.0, 0.0, 0.0, 1.0,-1.0, 0.0])
####X, Y = poly2cw(X, Y, 1)
#####print 'X   = ',X
#####print 'Y   = ',Y
####X = np.array([ 0.0, 1.0, 1.0, 0.0, 0.0])
####Y = np.array([ 0.0, 0.0, 1.0, 1.0, 0.0])
####XX= np.array([ 0.5, 1.0, 0.5, 0.0, 0.5])
####YY= np.array([ 0.5, 1.2, 2.0, 1.2, 0.5])
####area = area_common(X,Y,XX,YY)
#####print 'X      = ',X
#####print 'Y      = ',Y
#####print 'XX     = ',XX
#####print 'YY     = ',YY
#####print 'area = ', area
#####--- END of 'Tests of sub-programs' ------------------


#-------------------------------------------------------------------
#
#  Read POLMIP Daily Volcanic SO2 emission data
#  and interpolate it on to WRF grid
#
def AddDict_GFED_OnWRF( emis_dict, dname,\
                    area_common_array,area_common_dict,emis_cell_area,\
                    XLON, XLAT, start_dt, end_dt):
  #import datetime
  keyname  = 'DMS_OC'
  #
  month_txt = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
  #
  emis_tmp  = np.zeros(shape=(360,180,12))
  for imonth, month in enumerate(month_txt):
    filename = dname+'DMSclim_'+month+'.csv'
    #
    with open(filename) as f:
      for iline, line in enumerate(f):
        line = re.sub(r'\s','',line) 
        line = re.sub('NaN','0.0',line)
        line_value = line.split(',')
        line_value = [float(i) for i in line_value]
        emis_tmp[:,iline,imonth] = line_value[:]
        #print iline, line_value
        if iline == 179:
          break
  
  #======================================================================
  #----------------------------------------------------------------------
  #
  #  Expand longitude range from 0 < lon < 360 to -180 < lon < 360
  #
  #  Output: emis_tmp[nlon+nlon/2,nlat,12]
  #          emis_lon[nlon+nlon/2,nlat] (only used here)
  #          emis_lat[nlon+nlon/2,nlat] (only used here)
  #
  #emis_tmp   = np.concatenate((emis_tmp[-emis_tmp.shape[0]//2:,:,:],emis_tmp[:,:,:]),axis=0)
  emis_tmp   = np.concatenate((emis_tmp[:,:,:],emis_tmp[:emis_tmp.shape[0]//2,:,:]),axis=0)
  emis_tmp   = emis_tmp[:,::-1,:]

  ### plt.figure()
  ### # m = Basemap(projection='robin', lon_0=0, resolution='c')
  ### m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, \
  ###             llcrnrlon=-180, urcrnrlon=360, resolution='l')
  ### m.drawcountries()
  ### #m.bluemarble()
  ### m.drawcoastlines()
  ### #m.drawmapboundary(fill_color='aqua')
  ### m.drawmapboundary()
  ### m.drawmeridians(np.arange(-180,360, 30))
  ### m.drawparallels(np.arange( -90, 90, 30))
  ### 
  ### x, y = m(lon_2D,lat_2D)
  ### #clev = np.concatenate((np.linspace(0.0001,0.001,10),np.linspace(0.001,0.01,10),np.array([
  ### cmap = plt.get_cmap('jet')
  ### #norm = colors.BoundaryNorm(clev,cmap.N)
  ### #cs   = m.contourf(x,y,data,cmap=cmap,norm=norm, spacing='uniform', levels=clev)
  ### cs   = m.contourf(x,y,emis_tmp[:,:,0],cmap=cmap,spacing='uniform')
  ### cbar = m.colorbar(cs,location='bottom',pad="5%")
  ### 
  ### plt.show()
  ### raw_input()


  #
  #=====================================================================
  #---------------------------------------------------------------------
  #
  #  assign variables : 'lon' and 'lat' 
  ### lon   = np.array(nc.variables["lon"])   #    0 < lon < 360
  ### lat   = np.array(nc.variables["lat"])   #  -90 < lat <  90
  ### #
  ### #  assign variable : 'soil'
  ### emis_tmp = np.array(nc.variables['volcano']) 
  ### #
  ### nanind = np.where(np.isnan(emis_tmp))              # Indices of NaN in emis_tmp
  ### #emis_vname.append(var)                             # Create a list of variable name 
  ### emis_tmp[nanind] = np.zeros_like(emis_tmp)[nanind] # Replace NaN with 0
  ### #emis_nmvoc[:,:,iv-2] = np.transpose(emis_tmp[:,:]) # Set data to 'emis_nmvoc' [lon,lat,nvar]
  ### #
  ### #----------------------------------------------------------------------
  ### #
  ### #  Change the order of indexing
  ### #
  ### emis_tmp   = np.transpose(emis_tmp)     # [time,lat,lon] ---> [lon,lat,time]
  ### #  
  ### #======================================================================
  ### #----------------------------------------------------------------------
  ### #
  ### #  Expand longitude range from 0 < lon < 360 to -180 < lon < 360
  ### #
  ### #  Output: emis_tmp[nlon+nlon/2,nlat,12]
  ### #          emis_lon[nlon+nlon/2,nlat] (only used here)
  ### #          emis_lat[nlon+nlon/2,nlat] (only used here)
  ### #
  ### emis_tmp   = np.concatenate((emis_tmp[-emis_tmp.shape[0]/2:,:,:],emis_tmp[:,:,:]),axis=0)
  ### #
  ### #===========================================================================================
  ### #-------------------------------------------------------------------------------------------
  ### #  variable : 'soil'
  ### #
  # New Grid Dimensions
  ng_we_wrf = XLON.shape[0]
  ng_sn_wrf = XLON.shape[1]
  #
  emis_wrf  = np.zeros(shape=(ng_we_wrf,ng_sn_wrf,12))
  #
  dt  = [] 
  ### start_time = datetime.datetime(2008,start_dt.month,start_dt.day,0,0)
  ### end_time   = datetime.datetime(2008,  end_dt.month,  end_dt.day,0,0)
  ### #
  for imonth in np.arange(12):
    #dt_temp = datetime.datetime(2008,1,1,0,0) + datetime.timedelta(iday)
    #if dt_temp < start_time or dt_temp > end_time:
    #  continue
    #dt.append(dt_temp)
    #print iday, '/367'
    for c in np.arange(ng_we_wrf):                          # Loop over west_east index of wrf grid (wrfinput_d01)
      for r in np.arange(ng_sn_wrf):                        # Loop over south_north index of wrf grid (wrfinput_d01)
        if 1 == 0:
          #... Common to WRF grid ........................
          #    Index ranges of 'west-east' and 'south-north' variables of emission grid. 
          #    
          ind_lon_min = int(area_common_array[c,r,0])
          ind_lon_max = int(area_common_array[c,r,1])
          ind_lat_min = int(area_common_array[c,r,2])
          ind_lat_max = int(area_common_array[c,r,3])
          #
          count            = 1           # index counter for 3rd index of area_common_array[c,r,3+count]
          emis_wrf_temp    = 0.0         # temp. variable of emission on a wrf grid
          ## area_all    = 0.0         # checking parameter : should be 1.0 after the loop
          #
          # Loops over "ilon" 'west-east (longitude)' and "ilat" 'south-north (latitude)' 
          # index of emission grid 
          # Only those containing a wrf cell (XLON[c,r],XLAT[c,r]) defined by 4 points [X,Y]
          #
          if np.amax(emis_tmp[ind_lon_min:ind_lon_max,ind_lat_min:ind_lat_max,imonth]) != 0.0:
            for ilon in np.arange(ind_lon_min,ind_lon_max):      
              for ilat in np.arange(ind_lat_min,ind_lat_max):    
                if emis_tmp[ilon,ilat,imonth] != 0.0 :     # Just to avoid a useless computation 
                  area_temp     = area_common_array[c,r,3+count]     # Area ratio (area of emission grid (ilon,ilat) shared with a wrf grid)/(area of emission grid)
                  emis_wrf_temp = emis_wrf_temp+area_temp*emis_tmp[ilon,ilat,imonth]
                #area_all    = area_all + area_temp
                count += 1
            # End of for ilon and ilat
            #print 'area_all = ',area_all
            #print c,'/',ng_we_wrf,' ',r,'/',ng_sn_wrf,emis_wrf
        #
        #--- using area_common_dict -------------------------------------------
        #
        if 1 == 1:
          emis_wrf_temp = 0.0
          for idict in np.arange(area_common_dict[c][r]['total_count']):
            ilon = area_common_dict[c][r][idict]['ilon']
            ilat = area_common_dict[c][r][idict]['ilat']
            emis_wrf_temp += area_common_dict[c][r][idict]['area']*emis_tmp[ilon,ilat,imonth]
        emis_wrf[c,r,imonth] = emis_wrf_temp

        # Unit : [mol/cm3] --> [mol/m3] 
        emis_wrf[c,r,imonth] *= 1.e9 
      # End of loop r
    # End of loop c
  # End of loop iday
  #
  #-----------------------------------------------------------
  #
  emis_dict[keyname]={}         # Create a dictionary entry with a 'shp_NMVOC'iable name ''shp_NMVOC''
  emis_dict[keyname]['dimensions']={}
  emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
  emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
  emis_dict[keyname]['dimensions']['time']       = 12        # of points in time series> e.g.:12
  emis_dict[keyname]['west_east']={}
  emis_dict[keyname]['west_east']['dtype']='i4'
  emis_dict[keyname]['west_east']['dims' ]=['west_east']
  emis_dict[keyname]['west_east']['units']=''
  emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
  emis_dict[keyname]['south_north']={}
  emis_dict[keyname]['south_north']['dtype']='i4'
  emis_dict[keyname]['south_north']['dims' ]=['south_north']
  emis_dict[keyname]['south_north']['units']=''
  emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
  emis_dict[keyname]['longitude']={}
  emis_dict[keyname]['longitude']['dtype']='f4'
  emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
  emis_dict[keyname]['longitude']['units']='degrees_east'
  emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
  emis_dict[keyname]['latitude']={}
  emis_dict[keyname]['latitude']['dtype']='f4'
  emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
  emis_dict[keyname]['latitude']['units']='degrees_east'
  emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
  emis_dict[keyname]['time']={}
  emis_dict[keyname]['time']['dtype']='i4'
  emis_dict[keyname]['time']['dims' ]=['time']
  emis_dict[keyname]['time']['data' ]=[1,2,3,4,5,6,7,8,9,10,11,12] # Month
  emis_dict[keyname]['voc']={}
  emis_dict[keyname]['voc']['dtype']='f4'
  emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time']
  #emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
  emis_dict[keyname]['voc']['units']='mol/m3'
  emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
  
  return emis_dict


#-------------------------------------------------------------------
#
#  Read POLMIP Daily Volcanic SO2 emission data
#  and interpolate it on to WRF grid
#
def AddDict_LANA_DMS_Ocean_OnWRF( emis_dict, dname,\
                    area_common_array, area_common_dict, emis_cell_area,\
                    XLON, XLAT, start_dt, end_dt):
  #import datetime
  keyname  = 'DMS_OC-LANA'
  #
  month_txt = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
  #
  emis_tmp  = np.zeros(shape=(360,180,12))
  for imonth, month in enumerate(month_txt):
    filename = dname+'DMSclim_'+month+'.csv'
    #
    with open(filename) as f:
      for iline, line in enumerate(f):
        line = re.sub(r'\s','',line) 
        line = re.sub('NaN','0.0',line)
        line_value = line.split(',')
        line_value = [float(i) for i in line_value]
        emis_tmp[:,iline,imonth] = line_value[:]
        #print iline, line_value
        if iline == 179:
          break
  
  #======================================================================
  #----------------------------------------------------------------------
  #
  #  Expand longitude range from 0 < lon < 360 to -180 < lon < 360
  #
  #  Output: emis_tmp[nlon+nlon/2,nlat,12]
  #          emis_lon[nlon+nlon/2,nlat] (only used here)
  #          emis_lat[nlon+nlon/2,nlat] (only used here)
  #
  #emis_tmp   = np.concatenate((emis_tmp[-emis_tmp.shape[0]//2:,:,:],emis_tmp[:,:,:]),axis=0)
  emis_tmp   = np.concatenate((emis_tmp[:,:,:],emis_tmp[:emis_tmp.shape[0]//2,:,:]),axis=0)
  emis_tmp   = emis_tmp[:,::-1,:]

  ### plt.figure()
  ### # m = Basemap(projection='robin', lon_0=0, resolution='c')
  ### m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, \
  ###             llcrnrlon=-180, urcrnrlon=360, resolution='l')
  ### m.drawcountries()
  ### #m.bluemarble()
  ### m.drawcoastlines()
  ### #m.drawmapboundary(fill_color='aqua')
  ### m.drawmapboundary()
  ### m.drawmeridians(np.arange(-180,360, 30))
  ### m.drawparallels(np.arange( -90, 90, 30))
  ### 
  ### x, y = m(lon_2D,lat_2D)
  ### #clev = np.concatenate((np.linspace(0.0001,0.001,10),np.linspace(0.001,0.01,10),np.array([
  ### cmap = plt.get_cmap('jet')
  ### #norm = colors.BoundaryNorm(clev,cmap.N)
  ### #cs   = m.contourf(x,y,data,cmap=cmap,norm=norm, spacing='uniform', levels=clev)
  ### cs   = m.contourf(x,y,emis_tmp[:,:,0],cmap=cmap,spacing='uniform')
  ### cbar = m.colorbar(cs,location='bottom',pad="5%")
  ### 
  ### plt.show()
  ### raw_input()


  #
  #=====================================================================
  #---------------------------------------------------------------------
  #
  #  assign variables : 'lon' and 'lat' 
  ### lon   = np.array(nc.variables["lon"])   #    0 < lon < 360
  ### lat   = np.array(nc.variables["lat"])   #  -90 < lat <  90
  ### #
  ### #  assign variable : 'soil'
  ### emis_tmp = np.array(nc.variables['volcano']) 
  ### #
  ### nanind = np.where(np.isnan(emis_tmp))              # Indices of NaN in emis_tmp
  ### #emis_vname.append(var)                             # Create a list of variable name 
  ### emis_tmp[nanind] = np.zeros_like(emis_tmp)[nanind] # Replace NaN with 0
  ### #emis_nmvoc[:,:,iv-2] = np.transpose(emis_tmp[:,:]) # Set data to 'emis_nmvoc' [lon,lat,nvar]
  ### #
  ### #----------------------------------------------------------------------
  ### #
  ### #  Change the order of indexing
  ### #
  ### emis_tmp   = np.transpose(emis_tmp)     # [time,lat,lon] ---> [lon,lat,time]
  ### #  
  ### #======================================================================
  ### #----------------------------------------------------------------------
  ### #
  ### #  Expand longitude range from 0 < lon < 360 to -180 < lon < 360
  ### #
  ### #  Output: emis_tmp[nlon+nlon/2,nlat,12]
  ### #          emis_lon[nlon+nlon/2,nlat] (only used here)
  ### #          emis_lat[nlon+nlon/2,nlat] (only used here)
  ### #
  ### emis_tmp   = np.concatenate((emis_tmp[-emis_tmp.shape[0]/2:,:,:],emis_tmp[:,:,:]),axis=0)
  ### #
  ### #===========================================================================================
  ### #-------------------------------------------------------------------------------------------
  ### #  variable : 'soil'
  ### #
  # New Grid Dimensions
  ng_we_wrf = XLON.shape[0]
  ng_sn_wrf = XLON.shape[1]
  #
  emis_wrf  = np.zeros(shape=(ng_we_wrf,ng_sn_wrf,12))
  #
  dt  = [] 
  ### start_time = datetime.datetime(2008,start_dt.month,start_dt.day,0,0)
  ### end_time   = datetime.datetime(2008,  end_dt.month,  end_dt.day,0,0)
  ### #
  for imonth in np.arange(12):
    #dt_temp = datetime.datetime(2008,1,1,0,0) + datetime.timedelta(iday)
    #if dt_temp < start_time or dt_temp > end_time:
    #  continue
    #dt.append(dt_temp)
    #print iday, '/367'
    for c in np.arange(ng_we_wrf):                          # Loop over west_east index of wrf grid (wrfinput_d01)
      for r in np.arange(ng_sn_wrf):                        # Loop over south_north index of wrf grid (wrfinput_d01)
        if 1 == 0:
          #... Common to WRF grid ........................
          #    Index ranges of 'west-east' and 'south-north' variables of emission grid. 
          #    
          ind_lon_min = int(area_common_array[c,r,0])
          ind_lon_max = int(area_common_array[c,r,1])
          ind_lat_min = int(area_common_array[c,r,2])
          ind_lat_max = int(area_common_array[c,r,3])
          #
          count            = 1           # index counter for 3rd index of area_common_array[c,r,3+count]
          emis_wrf_temp    = 0.0         # temp. variable of emission on a wrf grid
          ## area_all    = 0.0         # checking parameter : should be 1.0 after the loop
          #
          # Loops over "ilon" 'west-east (longitude)' and "ilat" 'south-north (latitude)' 
          # index of emission grid 
          # Only those containing a wrf cell (XLON[c,r],XLAT[c,r]) defined by 4 points [X,Y]
          #
          if np.amax(emis_tmp[ind_lon_min:ind_lon_max,ind_lat_min:ind_lat_max,imonth]) != 0.0:
            for ilon in np.arange(ind_lon_min,ind_lon_max):      
              for ilat in np.arange(ind_lat_min,ind_lat_max):    
                if emis_tmp[ilon,ilat,imonth] != 0.0 :     # Just to avoid a useless computation 
                  area_temp     = area_common_array[c,r,3+count]     # Area ratio (area of emission grid (ilon,ilat) shared with a wrf grid)/(area of emission grid)
                  emis_wrf_temp = emis_wrf_temp+area_temp*emis_tmp[ilon,ilat,imonth]
                #area_all    = area_all + area_temp
                count += 1
            # End of for ilon and ilat
            #print 'area_all = ',area_all
            #print c,'/',ng_we_wrf,' ',r,'/',ng_sn_wrf,emis_wrf
        #
        #---- Using area_common_dict ------------------------------------------------
        #
        if 1==1:
          emis_wrf_temp = 0.0
          for idict in np.arange(area_common_dict[c][r]['total_count']):
            ilon = area_common_dict[c][r][idict]['ilon']
            ilat = area_common_dict[c][r][idict]['ilat']
            emis_wrf_temp += area_common_dict[c][r][idict]['area']*emis_tmp[ilon,ilat,imonth]
        #
        #---- End of Using area_common_dict ------------------------------------------
        # 
        emis_wrf[c,r,imonth] = emis_wrf_temp
        # Unit : [mol/cm3] --> [mol/m3] 
        emis_wrf[c,r,imonth] *= 1.e9 
      # End of loop r
    # End of loop c
  # End of loop iday
  #
  #-----------------------------------------------------------
  #
  emis_dict[keyname]={}         # Create a dictionary entry with a 'shp_NMVOC'iable name ''shp_NMVOC''
  emis_dict[keyname]['dimensions']={}
  emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
  emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
  emis_dict[keyname]['dimensions']['time']       = 12        # of points in time series> e.g.:12
  emis_dict[keyname]['west_east']={}
  emis_dict[keyname]['west_east']['dtype']='i4'
  emis_dict[keyname]['west_east']['dims' ]=['west_east']
  emis_dict[keyname]['west_east']['units']=''
  emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
  emis_dict[keyname]['south_north']={}
  emis_dict[keyname]['south_north']['dtype']='i4'
  emis_dict[keyname]['south_north']['dims' ]=['south_north']
  emis_dict[keyname]['south_north']['units']=''
  emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
  emis_dict[keyname]['longitude']={}
  emis_dict[keyname]['longitude']['dtype']='f4'
  emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
  emis_dict[keyname]['longitude']['units']='degrees_east'
  emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
  emis_dict[keyname]['latitude']={}
  emis_dict[keyname]['latitude']['dtype']='f4'
  emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
  emis_dict[keyname]['latitude']['units']='degrees_east'
  emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
  emis_dict[keyname]['time']={}
  emis_dict[keyname]['time']['dtype']='i4'
  emis_dict[keyname]['time']['dims' ]=['time']
  emis_dict[keyname]['time']['data' ]=[1,2,3,4,5,6,7,8,9,10,11,12] # Month
  emis_dict[keyname]['voc']={}
  emis_dict[keyname]['voc']['dtype']='f4'
  emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time']
  #emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
  emis_dict[keyname]['voc']['units']='mol/m3'
  emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
  
  return emis_dict

#-------------------------------------------------------------------
#
#  Read POLMIP Daily Volcanic SO2 emission data
#  and interpolate it on to WRF grid
#
def AddDict_POLMIP_Daily_vol_SO2_OnWRF( emis_dict, fname,\
                    area_common_array, area_common_dict, emis_cell_area,\
                    XLON, XLAT, start_dt, end_dt):
  #
  # Input  unit : [molecules/cm2/sec]
  # Output unit : [mol/sec/m2]
  #
  #import datetime
  keyname  = 'vol-POLMIP-SO2'
  #
  nc    = Dataset(fname,'r',format='NETCDF4')
  #
  #====================================================================
  #--------------------------------------------------------------------
  #
  #  Get a list of variable names 
  #  vars = [u'lon', u'lat', u'time', u'date', u'anthro', u'bb', u'soil']
  #  nvars= 7
  #  dimension = [367,180,360]
  #  lon.shape[0] = 360
  #  lat.shape[0] = 180
  #  
  #  Output: nvars : # of variables 
  #          vars  : list of variable names
  #
  vars  = list(nc.variables.keys())
  nvars = len(vars)
  vars  = list(vars)
  #
  #=====================================================================
  #---------------------------------------------------------------------
  #
  #  assign variables : 'lon' and 'lat' 
  lon   = np.array(nc.variables["lon"])   #    0 < lon < 360
  lat   = np.array(nc.variables["lat"])   #  -90 < lat <  90
  #
  #  assign variable : 'soil'
  emis_tmp = np.array(nc.variables['volcano']) 
  #
  nanind = np.where(np.isnan(emis_tmp))              # Indices of NaN in emis_tmp
  #emis_vname.append(var)                             # Create a list of variable name 
  emis_tmp[nanind] = np.zeros_like(emis_tmp)[nanind] # Replace NaN with 0
  #emis_nmvoc[:,:,iv-2] = np.transpose(emis_tmp[:,:]) # Set data to 'emis_nmvoc' [lon,lat,nvar]
  #
  #----------------------------------------------------------------------
  #
  #  Change the order of indexing
  #
  emis_tmp   = np.transpose(emis_tmp)     # [time,lat,lon] ---> [lon,lat,time]
  #  
  #======================================================================
  #----------------------------------------------------------------------
  #
  #  Expand longitude range from 0 < lon < 360 to -180 < lon < 360
  #
  #  Output: emis_tmp[nlon+nlon/2,nlat,12]
  #          emis_lon[nlon+nlon/2,nlat] (only used here)
  #          emis_lat[nlon+nlon/2,nlat] (only used here)
  #
  emis_tmp   = np.concatenate((emis_tmp[-emis_tmp.shape[0]//2:,:,:],emis_tmp[:,:,:]),axis=0)
  #
  #===========================================================================================
  #-------------------------------------------------------------------------------------------
  #  variable : 'soil'
  #
  # New Grid Dimensions
  ng_we_wrf = XLON.shape[0]
  ng_sn_wrf = XLON.shape[1]
  #
  emis_wrf  = np.zeros(shape=(ng_we_wrf,ng_sn_wrf,367))
  #
  dt  = [] 
  start_time = datetime(2008,start_dt.month,start_dt.day,0,0)
  end_time   = datetime(2008,  end_dt.month,  end_dt.day,0,0)
  #
  for iday in np.arange(367):
    dt_temp = datetime(2008,1,1,0,0) + timedelta(days=int(iday))
    if dt_temp < start_time or dt_temp > end_time:
      continue
    dt.append(dt_temp)
    if rank == 0:
      print(iday, '/367')
    for c in np.arange(ng_we_wrf):                          # Loop over west_east index of wrf grid (wrfinput_d01)
      for r in np.arange(ng_sn_wrf):                        # Loop over south_north index of wrf grid (wrfinput_d01)
        if 1==0:
          #... Common to WRF grid ........................
          #    Index ranges of 'west-east' and 'south-north' variables of emission grid. 
          #    
          ind_lon_min = int(area_common_array[c,r,0])
          ind_lon_max = int(area_common_array[c,r,1])
          ind_lat_min = int(area_common_array[c,r,2])
          ind_lat_max = int(area_common_array[c,r,3])
          #
          count            = 1           # index counter for 3rd index of area_common_array[c,r,3+count]
          emis_wrf_temp    = 0.0         # temp. variable of emission on a wrf grid
          ## area_all    = 0.0         # checking parameter : should be 1.0 after the loop
          #
          # Loops over "ilon" 'west-east (longitude)' and "ilat" 'south-north (latitude)' 
          # index of emission grid 
          # Only those containing a wrf cell (XLON[c,r],XLAT[c,r]) defined by 4 points [X,Y]
          #
          if np.amax(emis_tmp[ind_lon_min:ind_lon_max,ind_lat_min:ind_lat_max,iday]) != 0.0:
            for ilon in np.arange(ind_lon_min,ind_lon_max):      
              for ilat in np.arange(ind_lat_min,ind_lat_max):    
                if emis_tmp[ilon,ilat,iday] != 0.0 :     # Just to avoid a useless computation 
                  area_temp     = area_common_array[c,r,3+count]     # Area ratio (area of emission grid (ilon,ilat) shared with a wrf grid)/(area of emission grid)
                  emis_wrf_temp = emis_wrf_temp+area_temp*emis_tmp[ilon,ilat,iday]
                #area_all    = area_all + area_temp
                count += 1
            # End of for ilon and ilat
            #print 'area_all = ',area_all
            #print c,'/',ng_we_wrf,' ',r,'/',ng_sn_wrf,emis_wrf
        #
        #---- Using area_common_dict ------------------------------------------------
        #
        if 1==1:
          emis_wrf_temp = 0.0
          for idict in np.arange(area_common_dict[c][r]['total_count']):
            ilon = area_common_dict[c][r][idict]['ilon']
            ilat = area_common_dict[c][r][idict]['ilat']
            emis_wrf_temp += area_common_dict[c][r][idict]['area']*emis_tmp[ilon,ilat,iday]
        #
        #---- End of Using area_common_dict ------------------------------------------
        # 
        emis_wrf[c,r,iday] = emis_wrf_temp
        # Unit : [molecules/cm2/sec] --> [mol/sec/m2] 
        emis_wrf[c,r,iday] *= 1.e4/(6.022*1.e23) 
      # End of loop r
    # End of loop c
  # End of loop iday
  #
  #-----------------------------------------------------------
  #
  emis_dict[keyname]={}         # Create a dictionary entry with a 'shp_NMVOC'iable name ''shp_NMVOC''
  emis_dict[keyname]['dimensions']={}
  emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
  emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
  emis_dict[keyname]['dimensions']['time']       = 367        # of points in time series> e.g.:12
  emis_dict[keyname]['west_east']={}
  emis_dict[keyname]['west_east']['dtype']='i4'
  emis_dict[keyname]['west_east']['dims' ]=['west_east']
  emis_dict[keyname]['west_east']['units']=''
  emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
  emis_dict[keyname]['south_north']={}
  emis_dict[keyname]['south_north']['dtype']='i4'
  emis_dict[keyname]['south_north']['dims' ]=['south_north']
  emis_dict[keyname]['south_north']['units']=''
  emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
  emis_dict[keyname]['longitude']={}
  emis_dict[keyname]['longitude']['dtype']='f4'
  emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
  emis_dict[keyname]['longitude']['units']='degrees_east'
  emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
  emis_dict[keyname]['latitude']={}
  emis_dict[keyname]['latitude']['dtype']='f4'
  emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
  emis_dict[keyname]['latitude']['units']='degrees_east'
  emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
  emis_dict[keyname]['time']={}
  emis_dict[keyname]['time']['dtype']='datetime'
  emis_dict[keyname]['time']['dims' ]=['time']
  emis_dict[keyname]['time']['data' ]=dt[:] # [datetime(2008,1,1),datetime(2008,1,2),....,datetime(2009,1,1)]
  emis_dict[keyname]['voc']={}
  emis_dict[keyname]['voc']['dtype']='f4'
  emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time']
  #emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
  emis_dict[keyname]['voc']['units']='mol/sec/m2'
  emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
  
  return emis_dict

#-------------------------------------------------------------------
#
#  Read POLMIP Daily Soil NOx Surface emission data
#  and interpolate it on to WRF grid
#
def AddDict_POLMIP_Daily_Soil_NO_OnWRF( emis_dict, fname,\
                    area_common_array, area_common_dict, emis_cell_area,\
                    XLON, XLAT, start_dt, end_dt):
  #
  # Input  unit : [molecules/cm2/sec]
  # Output unit : [mol/m2/sec]
  #
  #import datetime
  keyname  = 'soil-POLMIP-NO'
  if rank == 0:
    print(keyname)
  #
  nc    = Dataset(fname,'r',format='NETCDF4')
  #
  #====================================================================
  #--------------------------------------------------------------------
  #
  #  Get a list of variable names 
  #  vars = [u'lon', u'lat', u'time', u'date', u'anthro', u'bb', u'soil']
  #  nvars= 7
  #  dimension = [367,180,360]
  #  lon.shape[0] = 360
  #  lat.shape[0] = 180
  #  
  #  Output: nvars : # of variables 
  #          vars  : list of variable names
  #
  vars  = list(nc.variables.keys())
  nvars = len(vars)
  vars  = list(vars)
  if rank == 0:
    print(vars)
  #
  #=====================================================================
  #---------------------------------------------------------------------
  #
  #  assign variables : 'lon' and 'lat' 
  lon   = np.array(nc.variables["lon"])   #    0 < lon < 360
  lat   = np.array(nc.variables["lat"])   #  -90 < lat <  90
  #
  #  assign variable : 'soil'
  emis_tmp = np.array(nc.variables['soil']) 
  #
  nanind = np.where(np.isnan(emis_tmp))              # Indices of NaN in emis_tmp
  #emis_vname.append(var)                             # Create a list of variable name 
  emis_tmp[nanind] = np.zeros_like(emis_tmp)[nanind] # Replace NaN with 0
  #emis_nmvoc[:,:,iv-2] = np.transpose(emis_tmp[:,:]) # Set data to 'emis_nmvoc' [lon,lat,nvar]
  #
  #----------------------------------------------------------------------
  #
  #  Change the order of indexing
  #
  emis_tmp   = np.transpose(emis_tmp)     # [time,lat,lon] ---> [lon,lat,time]
  #  
  #======================================================================
  #----------------------------------------------------------------------
  #
  #  Expand longitude range from 0 < lon < 360 to -180 < lon < 360
  #
  #  Output: emis_tmp[nlon+nlon/2,nlat,12]
  #          emis_lon[nlon+nlon/2,nlat] (only used here)
  #          emis_lat[nlon+nlon/2,nlat] (only used here)
  #
  emis_tmp   = np.concatenate((emis_tmp[-emis_tmp.shape[0]//2:,:,:],emis_tmp[:,:,:]),axis=0)
  #
  #===========================================================================================
  #-------------------------------------------------------------------------------------------
  #  variable : 'soil'
  #
  # New Grid Dimensions
  ng_we_wrf = XLON.shape[0]
  ng_sn_wrf = XLON.shape[1]
  #
  emis_wrf  = np.zeros(shape=(ng_we_wrf,ng_sn_wrf,367))
  #
  dt  = [] 
  start_time = datetime(2008,start_dt.month,start_dt.day,0,0)
  end_time   = datetime(2008,  end_dt.month,  end_dt.day,0,0)
  #
  for iday in np.arange(367):
    dt_temp = datetime(2008,1,1,0,0) + timedelta(days=int(iday))
    if dt_temp < start_time or dt_temp > end_time:
      continue
    dt.append(dt_temp)
    if rank == 0:
      print(iday, '/367')
    for c in np.arange(ng_we_wrf):                          # Loop over west_east index of wrf grid (wrfinput_d01)
      for r in np.arange(ng_sn_wrf):                        # Loop over south_north index of wrf grid (wrfinput_d01)
        if 1==0:
          #... Common to WRF grid ........................
          #    Index ranges of 'west-east' and 'south-north' variables of emission grid. 
          #    
          ind_lon_min = int(area_common_array[c,r,0])
          ind_lon_max = int(area_common_array[c,r,1])
          ind_lat_min = int(area_common_array[c,r,2])
          ind_lat_max = int(area_common_array[c,r,3])
          #
          count            = 1           # index counter for 3rd index of area_common_array[c,r,3+count]
          emis_wrf_temp    = 0.0         # temp. variable of emission on a wrf grid
          ## area_all    = 0.0         # checking parameter : should be 1.0 after the loop
          #
          # Loops over "ilon" 'west-east (longitude)' and "ilat" 'south-north (latitude)' 
          # index of emission grid 
          # Only those containing a wrf cell (XLON[c,r],XLAT[c,r]) defined by 4 points [X,Y]
          #
          if np.amax(emis_tmp[ind_lon_min:ind_lon_max,ind_lat_min:ind_lat_max,iday]) != 0.0:
            for ilon in np.arange(ind_lon_min,ind_lon_max):      
              for ilat in np.arange(ind_lat_min,ind_lat_max):    
                if emis_tmp[ilon,ilat,iday] != 0.0 :     # Just to avoid a useless computation 
                  area_temp     = area_common_array[c,r,3+count]     # Area ratio (area of emission grid (ilon,ilat) shared with a wrf grid)/(area of emission grid)
                  emis_wrf_temp = emis_wrf_temp+area_temp*emis_tmp[ilon,ilat,iday]
                #area_all    = area_all + area_temp
                count += 1
            # End of for ilon and ilat
            #print 'area_all = ',area_all
            #print c,'/',ng_we_wrf,' ',r,'/',ng_sn_wrf,emis_wrf
        #
        #---- Using area_common_dict ------------------------------------------------
        #
        if 1==1:
          emis_wrf_temp = 0.0
          for idict in np.arange(area_common_dict[c][r]['total_count']):
            ilon = area_common_dict[c][r][idict]['ilon']
            ilat = area_common_dict[c][r][idict]['ilat']
            emis_wrf_temp += area_common_dict[c][r][idict]['area']*emis_tmp[ilon,ilat,iday]
        #
        #---- End of Using area_common_dict ------------------------------------------
        # 
        emis_wrf[c,r,iday] = emis_wrf_temp
        # Unit : [molecules/cm2/sec] --> [mol/sec/m2] 
        emis_wrf[c,r,iday] = emis_wrf[c,r,iday]*1.e4/(6.022*1.e23) 
      # End of loop r
    # End of loop c
  # End of loop iday
  #
  #-----------------------------------------------------------
  #
  emis_dict[keyname]={}                                  # Create a dictionary entry with a 'shp_NMVOC'iable name ''shp_NMVOC''
  emis_dict[keyname]['dimensions']={}
  emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
  emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
  emis_dict[keyname]['dimensions']['time']       = 367        # of points in time series> e.g.:12
  emis_dict[keyname]['west_east']={}
  emis_dict[keyname]['west_east']['dtype']='i4'
  emis_dict[keyname]['west_east']['dims' ]=['west_east']
  emis_dict[keyname]['west_east']['units']=''
  emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
  emis_dict[keyname]['south_north']={}
  emis_dict[keyname]['south_north']['dtype']='i4'
  emis_dict[keyname]['south_north']['dims' ]=['south_north']
  emis_dict[keyname]['south_north']['units']=''
  emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
  emis_dict[keyname]['longitude']={}
  emis_dict[keyname]['longitude']['dtype']='f4'
  emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
  emis_dict[keyname]['longitude']['units']='degrees_east'
  emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
  emis_dict[keyname]['latitude']={}
  emis_dict[keyname]['latitude']['dtype']='f4'
  emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
  emis_dict[keyname]['latitude']['units']='degrees_east'
  emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
  emis_dict[keyname]['time']={}
  emis_dict[keyname]['time']['dtype']='datetime'
  emis_dict[keyname]['time']['dims' ]=['time']
  emis_dict[keyname]['time']['data' ]=dt[:] # [datetime(2008,1,1),datetime(2008,1,2),....,datetime(2009,1,1)]
  emis_dict[keyname]['voc']={}
  emis_dict[keyname]['voc']['dtype']='f4'
  emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time']
  #emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
  emis_dict[keyname]['voc']['units']='mol/m2/sec'
  emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
  
  return emis_dict
#-------------------------------------------------------------------
#
#  Read ECLIPSE RCP60 NMVOC emission data
#  and interpolate it on to WRF grid
#
def AddDict_RCP60_SHP_OnWRF( emis_dict, fname_voc,\
                    area_common_array, area_common_dict, emis_cell_area,\
                    XLON, XLAT):
  #
  # Input unit : [kg/m2/sec]
  # Output unit: [kg/m2/sec]
  #
  basename = os.path.basename(fname_voc)
  if rank == 0:
    print(basename)
  spec     = basename.split('_')[3]
  keyname  = 'shp-RCP60-'+spec
  if rank == 0:
    print(keyname)
  #
  nc    = Dataset(fname_voc,'r',format='NETCDF4')
  #
  #====================================================================
  #--------------------------------------------------------------------
  #
  #  Get a list of variable names 
  #  vars = [u'lat', u'lon', u'date', u'time', u'emiss_shp', u'molecular_weight']
  #  nvars= 6
  #  dimension = [12,360,720]
  #  lon.shape[0] = 720
  #  lat.shape[0] = 360
  #  
  #  Output: nvars : # of variables 
  #          vars  : list of variable names
  #
  vars  = list(nc.variables.keys())
  nvars = len(vars)
  vars  = list(vars)
  if rank == 0:
    print(vars)
  #
  #=====================================================================
  #---------------------------------------------------------------------
  #
  #  assign variables : 'lon' and 'lat' 
  lon   = np.array(nc.variables["lon"])   #    0 < lon < 360
  lat   = np.array(nc.variables["lat"])   #  -90 < lat <  90
  #
  #  assign variable : 'emis_shp'
  emis_tmp = np.array(nc.variables['emiss_shp']) 
  #
  nanind = np.where(np.isnan(emis_tmp))              # Indices of NaN in emis_tmp
  #emis_vname.append(var)                             # Create a list of variable name 
  emis_tmp[nanind] = np.zeros_like(emis_tmp)[nanind] # Replace NaN with 0
  #emis_nmvoc[:,:,iv-2] = np.transpose(emis_tmp[:,:]) # Set data to 'emis_nmvoc' [lon,lat,nvar]
  #
  #----------------------------------------------------------------------
  #
  #  Change the order of indexing
  #
  emis_tmp   = np.transpose(emis_tmp)     # [time,lat,lon] ---> [lon,lat,time]
  #  
  #======================================================================
  #----------------------------------------------------------------------
  #
  #  Expand longitude range from 0 < lon < 360 to -180 < lon < 360
  #
  #  Output: emis_tmp[nlon+nlon/2,nlat,12]
  #          emis_lon[nlon+nlon/2,nlat] (only used here)
  #          emis_lat[nlon+nlon/2,nlat] (only used here)
  #
  emis_tmp   = np.concatenate((emis_tmp[-emis_tmp.shape[0]//2:,:,:],emis_tmp[:,:,:]),axis=0)
  #
  #===========================================================================================
  #-------------------------------------------------------------------------------------------
  #  variable : 'emis_tmp'
  #
  # New Grid Dimensions
  ng_we_wrf = XLON.shape[0]
  ng_sn_wrf = XLON.shape[1]
  #
  emis_wrf  = np.zeros(shape=(ng_we_wrf,ng_sn_wrf,12))
  #
  for imonth in np.arange(12):
    for c in np.arange(ng_we_wrf):                          # Loop over west_east index of wrf grid (wrfinput_d01)
      for r in np.arange(ng_sn_wrf):                        # Loop over south_north index of wrf grid (wrfinput_d01)
        if 1==0:
          #... Common to WRF grid ........................
          #    Index ranges of 'west-east' and 'south-north' variables of emission grid. 
          #    
          ind_lon_min = int(area_common_array[c,r,0])
          ind_lon_max = int(area_common_array[c,r,1])
          ind_lat_min = int(area_common_array[c,r,2])
          ind_lat_max = int(area_common_array[c,r,3])
          #
          count            = 1           # index counter for 3rd index of area_common_array[c,r,3+count]
          emis_wrf_temp    = 0.0         # temp. variable of emission on a wrf grid
          ## area_all    = 0.0         # checking parameter : should be 1.0 after the loop
          #
          # Loops over "ilon" 'west-east (longitude)' and "ilat" 'south-north (latitude)' 
          # index of emission grid 
          # Only those containing a wrf cell (XLON[c,r],XLAT[c,r]) defined by 4 points [X,Y]
          #
          if np.amax(emis_tmp[ind_lon_min:ind_lon_max,ind_lat_min:ind_lat_max,imonth]) != 0.0:
            for ilon in np.arange(ind_lon_min,ind_lon_max):      
              for ilat in np.arange(ind_lat_min,ind_lat_max):    
                if emis_tmp[ilon,ilat,imonth] != 0.0 :     # Just to avoid a useless computation 
                  area_temp     = area_common_array[c,r,3+count]     # Area ratio (area of emission grid (ilon,ilat) shared with a wrf grid)/(area of emission grid)
                  emis_wrf_temp = emis_wrf_temp+area_temp*emis_tmp[ilon,ilat,imonth]
                #area_all    = area_all + area_temp
                count += 1
            # End of for ilon and ilat
            #print 'area_all = ',area_all
            #print c,'/',ng_we_wrf,' ',r,'/',ng_sn_wrf,emis_wrf
        #
        #---- Using area_common_dict ------------------------------------------------
        #
        if 1==1:
          emis_wrf_temp = 0.0
          for idict in np.arange(area_common_dict[c][r]['total_count']):
            ilon = area_common_dict[c][r][idict]['ilon']
            ilat = area_common_dict[c][r][idict]['ilat']
            emis_wrf_temp += area_common_dict[c][r][idict]['area']*emis_tmp[ilon,ilat,imonth]
        #
        #---- End of Using area_common_dict ------------------------------------------
        # 
        emis_wrf[c,r,imonth] = emis_wrf_temp
      # End of loop r
    # End of loop c
  # End of loop imonth
  #
  #-----------------------------------------------------------
  #
  emis_dict[keyname]={}                                  # Create a dictionary entry with a 'shp_NMVOC'iable name ''shp_NMVOC''
  emis_dict[keyname]['dimensions']={}
  emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
  emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
  emis_dict[keyname]['dimensions']['time']       = 12        # of points in time series> e.g.:12
  emis_dict[keyname]['west_east']={}
  emis_dict[keyname]['west_east']['dtype']='i4'
  emis_dict[keyname]['west_east']['dims' ]=['west_east']
  emis_dict[keyname]['west_east']['units']=''
  emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
  emis_dict[keyname]['south_north']={}
  emis_dict[keyname]['south_north']['dtype']='i4'
  emis_dict[keyname]['south_north']['dims' ]=['south_north']
  emis_dict[keyname]['south_north']['units']=''
  emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
  emis_dict[keyname]['longitude']={}
  emis_dict[keyname]['longitude']['dtype']='f4'
  emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
  emis_dict[keyname]['longitude']['units']='degrees_east'
  emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
  emis_dict[keyname]['latitude']={}
  emis_dict[keyname]['latitude']['dtype']='f4'
  emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
  emis_dict[keyname]['latitude']['units']='degrees_east'
  emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
  emis_dict[keyname]['time']={}
  emis_dict[keyname]['time']['dtype']='i4'
  emis_dict[keyname]['time']['dims' ]=['time']
  emis_dict[keyname]['time']['data' ]=np.arange(12) # [0,1,2,3,4,...,11]
  emis_dict[keyname]['voc']={}
  emis_dict[keyname]['voc']['dtype']='f4'
  emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time']
  #emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
  emis_dict[keyname]['voc']['units']='kg/sec/m2'
  emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
  
  return emis_dict

#-------------------------------------------------------------------
#
#  Read Huang BC emission data
#  and interpolate it on to WRF grid
#
def AddDict_Huang_BC_OnWRF( emis_dict, fname,\
                    area_common_array, area_common_dict, emis_cell_area,\
                    XLON, XLAT):
  #
  # Input Unit : [kg/m2/sec]
  #
  Mmol      = {'CO':28,'CH4':16,'SO2':64,'NH3':17,'NO':30,'NO2':46}
  ind_ratio = {'CO' :[0.55,0.45],'CH4':[0.92,0.08],'BC'  :[0.5 ,0.5 ],'OM':[0.5,0.5],\
               'SO2':[0.71,0.29],'NH3':[0.08,0.92],'PM25':[0.43,0.57],'NO':[0.73,0.27],'NO2':[0.73,0.27]}
  #
  nc    = Dataset(fname,'r',format='NETCDF4')
  #
  #====================================================================
  #--------------------------------------------------------------------
  #
  #  Get a list of variable names 
  #  vars = ['lat','lon','cell_area','RUS_BC_FLARE','RUS_BC_INDUSTRY','RUS_BC_RESIDENTIAL','RUS_BC_TRANSPORT']
  #  dimension = [1800,3600]
  #  lon.shape[0] = 3600
  #  lat.shape[0] = 1800
  #  
  #  Output: nvars : # of variables 
  #          vars  : list of variable names
  #
  vars  = list(nc.variables.keys())
  nvars = len(vars)
  vars  = list(vars)
  # print vars
  #
  #=====================================================================
  #---------------------------------------------------------------------
  #
  #  assign variables : 'lon' and 'lat' 
  lon   = np.array(nc.variables["lon"])   #    0 < lon < 360
  lat   = np.array(nc.variables["lat"])   #  -90 < lat <  90
  #
  for iv, var in enumerate(vars):
    # print iv, var
    #
    try:
      emis_tmp = np.array(nc.variables[var]) 
    except:
      print(var)
      print(nc.variables[var])
      raise
    #
    # iv = 0 : 'lat'
    # iv = 1 : 'lon'
    # iv = 2 : 'cell_area'
    #
    nanind = np.where(np.isnan(emis_tmp))              # Indices of NaN in emis_tmp
    #emis_vname.append(var)                             # Create a list of variable name 
    emis_tmp[nanind] = np.zeros_like(emis_tmp)[nanind] # Replace NaN with 0
    #emis_nmvoc[:,:,iv-2] = np.transpose(emis_tmp[:,:]) # Set data to 'emis_nmvoc' [lon,lat,nvar]
    #
    #----------------------------------------------------------------------
    #
    #  Change the order of indexing
    #
    emis_tmp   = np.transpose(emis_tmp)     # [lat,lon] ---> [lon,lat]
    #  
    #======================================================================
    #----------------------------------------------------------------------
    #
    #  Expand longitude range from 0 < lon < 360 to -180 < lon < 360
    #
    #  Output: emis_nmvoc[nlon+nlon/2,nlat,nvars-3]
    #          emis_lon[nlon+nlon/2,nlat]
    #          emis_lat[nlon+nlon/2,nlat]
    #
    #
    print(emis_tmp.shape)
    emis_tmp   = np.concatenate((emis_tmp[emis_tmp.shape[0]//2:,:],emis_tmp[:,:]),axis=0)
    lon_temp   = np.concatenate((lon[lon.shape[0]//2:]-360.0,lon))
    emis_lon1  = lon_temp
    emis_lat1  = lat
    emis_lon, emis_lat = np.meshgrid(lon_temp,lat,indexing='ij')
    #
    #===========================================================================================
    #-------------------------------------------------------------------------------------------
    #  variable : 'RUS_BC_FLARE' etc
    # 
    #  unit : [kg/m2/sec]--> [ug/m2/sec]
    #
    units     = 'ug/m2/sec'
    emis_tmp *= 1.e9
    # 
    # New Grid Dimensions
    ng_we_wrf = XLON.shape[0]
    ng_sn_wrf = XLON.shape[1]
    #
    emis_wrf  = np.zeros(shape=(ng_we_wrf,ng_sn_wrf))
    # 
    for c in np.arange(ng_we_wrf):                          # Loop over west_east index of wrf grid (wrfinput_d01)
      for r in np.arange(ng_sn_wrf):                        # Loop over south_north index of wrf grid (wrfinput_d01)
        if 1==0:
          #... Common to WRF grid ........................
          #    Index ranges of 'west-east' and 'south-north' variables of emission grid. 
          #    
          ind_lon_min = int(area_common_array[c,r,0])
          ind_lon_max = int(area_common_array[c,r,1])
          ind_lat_min = int(area_common_array[c,r,2])
          ind_lat_max = int(area_common_array[c,r,3])
          #
          count            = 1           # index counter for 3rd index of area_common_array[c,r,3+count]
          emis_wrf_temp    = 0.0         # temp. variable of emission on a wrf grid
          ## area_all    = 0.0         # checking parameter : should be 1.0 after the loop
          #
          # Loops over "ilon" 'west-east (longitude)' and "ilat" 'south-north (latitude)' 
          # index of emission grid 
          # Only those containing a wrf cell (XLON[c,r],XLAT[c,r]) defined by 4 points [X,Y]
          #
          if np.amax(emis_tmp[ind_lon_min:ind_lon_max,ind_lat_min:ind_lat_max]) != 0.0:
            for ilon in np.arange(ind_lon_min,ind_lon_max):      
              for ilat in np.arange(ind_lat_min,ind_lat_max):    
                if emis_tmp[ilon,ilat] != 0.0 :     # Just to avoid a useless computation 
                  area_temp     = area_common_array[c,r,3+count]     # Area ratio (area of emission grid (ilon,ilat) shared with a wrf grid)/(area of emission grid)
                  emis_wrf_temp = emis_wrf_temp+area_temp*emis_tmp[ilon,ilat]
                #area_all    = area_all + area_temp
                count += 1
            # End of for ilon and ilat
            #print 'area_all = ',area_all
            #print c,'/',ng_we_wrf,' ',r,'/',ng_sn_wrf,emis_wrf
        #
        #---- Using area_common_dict ------------------------------------------------
        #
        if 1==1:
          emis_wrf_temp = 0.0
          for idict in np.arange(area_common_dict[c][r]['total_count']):
            ilon = area_common_dict[c][r][idict]['ilon']
            ilat = area_common_dict[c][r][idict]['ilat']
            emis_wrf_temp += area_common_dict[c][r][idict]['area']*emis_tmp[ilon,ilat]
        #
        #---- End of Using area_common_dict ------------------------------------------
        # 
        emis_wrf[c,r] = emis_wrf_temp
      # End of loop r
    # End of loop c
    #
    #-----------------------------------------------------------
    #
    sec_temp  = var.split('_')[2]
    if sec_temp == 'FLARE':
      sec = 'flr'
      #... 'uncomment continue to exclude 'flr' from wrfchemi ....
      #continue    

    elif sec_temp == 'INDUSTRY':
      sec = 'ind'
      #... 'uncomment continue to exclude 'ind' from wrfchemi ....
      #continue
    elif sec_temp == 'RESIDENTIAL':
      sec = 'dom'
      #... 'uncomment continue to exclude 'dom' from wrfchemi ....
      #continue
    elif sec_temp == 'TRANSPORT':
      sec = 'tra'
      #... 'uncomment continue to exclude 'tra' from wrfchemi ....
      #continue
    spec = 'BC'
    sec_arr = [sec]
    spec_arr= [spec]
    coef_arr= [1.0]
    if sec == 'ind':
      sec_arr = ['ind1','ind2']
    if spec == 'NOx':
      spec_arr = ['NO','NO2']
      coef_arr = [0.9,0.1]        # NO/NO2 ratio 0.9:0.1

    for isec, sec in enumerate(sec_arr):
      for ispec, spec in enumerate(spec_arr):
        Mm = 1.0  # for BC
        # Molecular mass (g/mol) 
        if spec in Mmol:
          Mm = Mmol[spec]
        keyname = sec+'-Huang-'+spec
        if rank == 0:
          print(keyname)
        emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
        emis_dict[keyname]['dimensions']={}
        emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
        emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
        # emis_dict[keyname]['dimensions']['time']       =<# of points in time series> e.g.:12
        emis_dict[keyname]['west_east']={}
        emis_dict[keyname]['west_east']['dtype']='i4'
        emis_dict[keyname]['west_east']['dims' ]=['west_east']
        emis_dict[keyname]['west_east']['units']=''
        emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
        emis_dict[keyname]['south_north']={}
        emis_dict[keyname]['south_north']['dtype']='i4'
        emis_dict[keyname]['south_north']['dims' ]=['south_north']
        emis_dict[keyname]['south_north']['units']=''
        emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
        emis_dict[keyname]['longitude']={}
        emis_dict[keyname]['longitude']['dtype']='f4'
        emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
        emis_dict[keyname]['longitude']['units']='degrees_east'
        emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
        emis_dict[keyname]['latitude']={}
        emis_dict[keyname]['latitude']['dtype']='f4'
        emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
        emis_dict[keyname]['latitude']['units']='degrees_east'
        emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
        # emis_dict[keyname]['time']={}
        # emis_dict[keyname]['time']['dtype']='i4'
        # emis_dict[keyname]['time']['dims' ]=['time']
        # emis_dict[keyname]['time']['data' ]=np.arange(12) : [0,1,2,3,4,...,11]
        emis_dict[keyname]['voc']={}
        emis_dict[keyname]['voc']['dtype']='f4'
        # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] or ['west_east','south_north']
        emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
        emis_dict[keyname]['voc']['units']= units
        # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
        if 'ind' in sec:
          # unit [g] --> [mol] for chemical species
          emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:]/Mm*ind_ratio[spec][isec]*coef_arr[ispec]
        else:
          emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:]/Mm*coef_arr[ispec] 
  return emis_dict

#-------------------------------------------------------------------
#
#  Read REAS Anthoropogenic emission data
#  and interpolate it on to WRF grid
#
def AddDict_REAS_Anth_VOC_OnWRF( emis_dict, fname,\
                    area_common_array, area_common_dict, emis_cell_area,\
                    XLON, XLAT,sector):
  #
  # Input  Unit : [t/month]
  # Output Unit : [kg/m2/sec] 
  #
  #
  lon, lat, emis_tmp = read_REAS(fname)
  #  
  #======================================================================
  #----------------------------------------------------------------------
  #
  #  Output: emis_nmvoc[nlon,nlat,12]
  #          emis_lon[nlon,nlat]
  #          emis_lat[nlon,nlat]
  #
  #
  emis_lon, emis_lat = np.meshgrid(lon,lat,indexing='ij')
  #
  #===========================================================================================
  #-------------------------------------------------------------------------------------------
  #  variable : 'emis_awb'
  day_of_month = np.array([31,29,31,30,31,30,31,31,30,31,30,31])
  #  unit : [ton/month] --> [kg/sec/m2]
  units    = 'kg/sec/m2'
  for imonth in np.arange(12):
    emis_tmp[:,:,imonth] = np.divide(emis_tmp[:,:,imonth],emis_cell_area)
    #emis_tmp[:,:,imonth] /= emis_cell_area
    #emis_tmp[:,:,imonth] = emis_tmp[:,:,imonth]*1.e3/(60.0*60.0*24.0*day_of_month[imonth])
    emis_tmp[:,:,imonth] *= (1.e3/(60.0*60.0*24.0*day_of_month[imonth]))
  ### if spec in ['CO','NH3','NOx','SO2','CH4']:
  ###   #  unit : [ton/month] --> [g/sec/m2]
  ###   units    = 'mol/sec/m2'
  ###   for imonth in np.arange(12):
  ###     emis_tmp[:,:,imonth] = np.divide(emis_tmp[:,:,imonth],emis_cell_area)
  ###     emis_tmp[:,:,imonth] = emis_tmp[:,:,imonth]*1.e6/(60.0*60.0*24.0*day_of_month[imonth])
  ### if spec in ['OM','PM25','BC']:
  ###   #  unit : [ton/month] --> [ug/sec/m2]
  ###   units    = 'ug/sec/m2'
  ###   for imonth in np.arange(12):
  ###     emis_tmp[:,:,imonth] = np.divide(emis_tmp[:,:,imonth],emis_cell_area)
  ###     emis_tmp[:,:,imonth] = emis_tmp[:,:,imonth]*1.e15/(60.0*60.0*24.0*day_of_month[imonth])
  # 
  # New Grid Dimensions
  ng_we_wrf = XLON.shape[0]
  ng_sn_wrf = XLON.shape[1]
  #
  emis_wrf_total  = np.zeros(shape=(ng_we_wrf,ng_sn_wrf,12))
  #
  for imonth in np.arange(12):
    emis_wrf  = np.zeros(shape=(ng_we_wrf,ng_sn_wrf))
    for c in np.arange(ng_we_wrf):                          # Loop over west_east index of wrf grid (wrfinput_d01)
      for r in np.arange(ng_sn_wrf):                        # Loop over south_north index of wrf grid (wrfinput_d01)
        if 1==0:
          #... Common to WRF grid ........................
          #    Index ranges of 'west-east' and 'south-north' variables of emission grid. 
          #    
          ind_lon_min = int(area_common_array[c,r,0])
          ind_lon_max = int(area_common_array[c,r,1])
          ind_lat_min = int(area_common_array[c,r,2])
          ind_lat_max = int(area_common_array[c,r,3])
          #
          count            = 1           # index counter for 3rd index of area_common_array[c,r,3+count]
          emis_wrf_temp    = 0.0         # temp. variable of emission on a wrf grid
          ## area_all    = 0.0         # checking parameter : should be 1.0 after the loop
          #
          # Loops over "ilon" 'west-east (longitude)' and "ilat" 'south-north (latitude)' 
          # index of emission grid 
          # Only those containing a wrf cell (XLON[c,r],XLAT[c,r]) defined by 4 points [X,Y]
          #
          if np.amax(emis_tmp[ind_lon_min:ind_lon_max,ind_lat_min:ind_lat_max,imonth]) != 0.0:
            for ilon in np.arange(ind_lon_min,ind_lon_max):      
              for ilat in np.arange(ind_lat_min,ind_lat_max):    
                if emis_tmp[ilon,ilat,imonth] != 0.0 :     # Just to avoid a useless computation 
                  area_temp     = area_common_array[c,r,3+count]     # Area ratio (area of emission grid (ilon,ilat) shared with a wrf grid)/(area of emission grid)
                  emis_wrf_temp += area_temp*emis_tmp[ilon,ilat,imonth]
                #area_all    = area_all + area_temp
                count += 1
            # End of for ilon and ilat
            #print 'area_all = ',area_all
            #print c,'/',ng_we_wrf,' ',r,'/',ng_sn_wrf,emis_wrf
        #
        #---- Using area_common_dict ------------------------------------------------
        #
        if 1==1:
          emis_wrf_temp = 0.0
          for idict in np.arange(area_common_dict[c][r]['total_count']):
            ilon = area_common_dict[c][r][idict]['ilon']
            ilat = area_common_dict[c][r][idict]['ilat']
            emis_wrf_temp += area_common_dict[c][r][idict]['area']*emis_tmp[ilon,ilat,imonth]
        #
        #---- End of Using area_common_dict ------------------------------------------
        # 
        emis_wrf[c,r] = emis_wrf_temp
      # End of loop r
    # End of loop c
    emis_wrf_total[:,:,imonth] = emis_wrf[:,:]
  # End of loop imonth
  #
  #-----------------------------------------------------------
  #
  # print var
  sec  = sector
  keyname = sec+'-REAS-VOC'
  if rank == 0:
    print(keyname)
  if emis_dict.get(keyname) == None:
    emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
    emis_dict[keyname]['dimensions']={}
    emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
    emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
    emis_dict[keyname]['dimensions']['time']       = 12        # of points in time series> e.g.:12
    emis_dict[keyname]['west_east']={}
    emis_dict[keyname]['west_east']['dtype']='i4'
    emis_dict[keyname]['west_east']['dims' ]=['west_east']
    emis_dict[keyname]['west_east']['units']=''
    emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
    emis_dict[keyname]['south_north']={}
    emis_dict[keyname]['south_north']['dtype']='i4'
    emis_dict[keyname]['south_north']['dims' ]=['south_north']
    emis_dict[keyname]['south_north']['units']=''
    emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
    emis_dict[keyname]['longitude']={}
    emis_dict[keyname]['longitude']['dtype']='f4'
    emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
    emis_dict[keyname]['longitude']['units']='degrees_east'
    emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
    emis_dict[keyname]['latitude']={}
    emis_dict[keyname]['latitude']['dtype']='f4'
    emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
    emis_dict[keyname]['latitude']['units']='degrees_east'
    emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
    emis_dict[keyname]['time']={}
    emis_dict[keyname]['time']['dtype']='i4'
    emis_dict[keyname]['time']['dims' ]=['time']
    emis_dict[keyname]['time']['data' ]=np.arange(12) #: [0,1,2,3,4,...,11]
    emis_dict[keyname]['voc']={}
    emis_dict[keyname]['voc']['dtype']='f4'
    emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] or ['west_east','south_north']
    # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
    units    = 'kg/sec/m2'
    emis_dict[keyname]['voc']['units']= units
    # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
    emis_dict[keyname]['voc']['data' ]= emis_wrf_total[:,:,:] 
  else:
    emis_dict[keyname]['voc']['data' ]+= emis_wrf_total[:,:,:] 
    
  return emis_dict


#-------------------------------------------------------------------
#-------------------------------------------------------------------
#
#  Read GFED Fire emission data          
#  and interpolate it on to WRF grid
#
def AddDict_GFED_Fire_OnWRF( emis_dict, dname, \
                    area_common_array, area_common_dict, emis_cell_area, \
                    XLON, XLAT, start_dt, end_dt, ECLIPSEinUse=False):


  # GFED SAPRC spec:
  #             ['ACET', 'ALK3', 'ALK4', 'ALK5', 
  #              'ARO1', 'ARO2', 'BACL', 'C2H2', 
  #              'C2H6', 'C3H6', 'C3H8', 'CCHO', 
  #              'CH4' , 'CO'  , 'EC'  , 'ETHENE', 
  #              'HCHO', 'HCOOH', 'HONO', 'MEK', 
  #              'MEOH', 'MVK', 'NH3', 'NO', 
  #              'OLE1', 'OLE2', 'ORG', 'PHEN', 
  #              'RCHO', 'SO2', 'TERP', 'cco_oh', 
  #              'isoprene', 'isoprod', 'methacro', 
  #              'no3j', 'pm25j', 'so4j']
  
  GFED_SAPRC_spec = \
               ['ACET', 'ALK3', 'ALK4', 'ALK5', \
                'ARO1', 'ARO2', 'BACL', 'C2H2', \
                'C2H6', 'C3H6', 'C3H8', 'CCHO', \
                'CH4' , 'CO'  , 'EC'  , 'ETHENE', \
                'HCHO', 'HCOOH', 'HONO', 'MEK', \
                'MEOH', 'MVK', 'NH3', 'NO', \
                'OLE1', 'OLE2', 'ORGJ', 'PHEN', \
                'RCHO', 'SO2', 'TERP', 'CCO_OH', \
                'ISOPRENE', 'ISOPROD', 'METHACRO', \
                'NO3J', 'PM25J', 'SO4J']

  #--- Read GFED monthly data ----
  # emis_GFED[keyname]['data'].shape = (720,1440)
  # unit : kg DM/sec/m2 
  # keyname : e.g. SAVA-GFED-DM-2015010103 
  #
  emis_GFED, lons, lats = read_GFED41s(start_dt,end_dt,WithEFs=False)

  EF_filename = '/data/onishi/GFEDv4/SAPRC/EmissionFactors_SAPRC_Summary.txt'
  EFs  = {}
  sec  = ['DEFO','SAVA','AGRI','BORF','TEMF','PEAT']
  
  spec_list = []
  f_EF = open(EF_filename,'r')
  for iline, line in enumerate(f_EF):
    line_split = line.split(';')
    if iline > 2:
      spec_temp = line_split[2].strip()
      if spec_temp != 'XXX' and spec_temp not in spec_list:
        spec_list.append(spec_temp)
  f_EF.close()

  f_EF = open(EF_filename,'r')
  for iline, line in enumerate(f_EF):
    line_split = line.split(';')
    if iline > 2:
      spec_temp = line_split[2].strip()
      Mmol_temp = line_split[1].strip()
      if spec_temp != 'XXX':
        #
        #--- Initialize ----
        #
        if spec_temp not in EFs:
          EFs[spec_temp]={}
          # Initialize Emission Factors 
          for isec, sec_temp in enumerate(sec):
            # Emission Factors : e.g. EFs['CO']['SAVA'] (g CO/kg DM)
            EFs[spec_temp][sec_temp]=0.0
            #print('initializing....EFs[',spec_temp,sec_temp,']')
        #
        #--- Accumulate Emission Factor for specific species and sector ---
        #
        for isec, sec_temp in enumerate(sec):
          if ECLIPSEinUse and sec is 'AGRI':
            continue
          EF_temp = line_split[isec+3].strip()
          if EF_temp != 'XXX':
            if spec_temp == 'ORGJ':
              EF_temp2 = 1.4*float(EF_temp)
            else:
              EF_temp2 = float(EF_temp)
            #print(EF_temp,Mmol_temp)
            if Mmol_temp != 'XXX':
              # Emission Factors : e.g. EFs['CO']['SAVA'] (mol CO/kg DM)
              EFs[spec_temp][sec_temp]+=EF_temp2/float(Mmol_temp)
              EFs[spec_temp]['unit'] = 'mol/kg'
            else:
              # Emission Factors : e.g. EFs['BC']['SAVA'] (g EC/kg DM)
              EFs[spec_temp][sec_temp]+=EF_temp2
              EFs[spec_temp]['unit'] = 'g/kg'
  f_EF.close()
  
  #======================================================================
  #----------------------------------------------------------------------
  #
  #  Output: emis_wrf[nlon,nlat,ntime] (ntime:hourly output from start_datetime to end_datetime)
  #          emis_lon[nlon,nlat]
  #          emis_lat[nlon,nlat]
  #
  emis_lon       = np.transpose(lons)
  emis_lat       = np.transpose(lats)
  #
  time0= datetime.now()
  for ikey, key in enumerate(emis_GFED.keys()):
    datetime_str   = key.split('-GFED-DM-')[-1]
    hour_temp      = int(datetime_str[11:13])
    #
    #-- If hour is NOT 0, 3, 6, 9, 12, 15 or 18 --------
    #
    if int(hour_temp/3)*3 != hour_temp:
      datetime_str_temp = datetime_str[0:11]+str(int(hour_temp/3)*3).zfill(2)+datetime_str[13:]
      #print(datetime_str_temp)
      for spec_temp in spec_list:
        keyname      = spec_temp+'-GFED-'+datetime_str
        keyname_temp = spec_temp+'-GFED-'+datetime_str_temp
        #print('copying ',keyname_temp,' to ',keyname)
        emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
        emis_dict[keyname] = emis_dict[keyname_temp]
        if 'NO-GFED' == keyname[:7]:
          keyname2 = keyname.replace('NO-','NO2-')
          keyname2_temp = keyname_temp.replace('NO-','NO2-')
          emis_dict[keyname2]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
          emis_dict[keyname2] = emis_dict[keyname2_temp]
        ## END OF if 'NO-GFED' == keyname.split('-')[0]:
      ## END OF for spec_temp in spec_list:
      continue
    ## END OF if int(hour_temp/3)*3 != hour_temp:
         

    #print('datetime_str : ',datetime_str)
    #print('hour_temp    : ',hour_temp)
    sec_name       = key.split('-GFED-DM-')[0]
    #GFED_unit      = emis_GFED[key]['unit']
    time0 = datetime.now()
    #
    #----------------------------------------------------------------------
    #
    emis_GFED_temp = np.transpose(emis_GFED[key]["data"])
    #----------------------------------------------------------------------
    #
    #  Expand longitude range from 0 < lon < 360 to -180 < lon < 360
    #
    emis_tmp       = np.concatenate((emis_GFED_temp[:,:],emis_GFED_temp[:emis_GFED_temp.shape[0]//2,:]),axis=0)
    #
    # New Grid Dimensions
    #
    ng_we_wrf = XLON.shape[0]
    ng_sn_wrf = XLON.shape[1]
    #
    emis_wrf_total = np.zeros(shape=(ng_we_wrf,ng_sn_wrf))
    for c in np.arange(ng_we_wrf):                          # Loop over west_east index of wrf grid (wrfinput_d01)
      for r in np.arange(ng_sn_wrf):                        # Loop over south_north index of wrf grid (wrfinput_d01)
        if 1==0:
          #... Common to WRF grid ........................
          #    Index ranges of 'west-east' and 'south-north' variables of emission grid. 
          #    
          ind_lon_min = int(area_common_array[c,r,0])
          ind_lon_max = int(area_common_array[c,r,1])
          ind_lat_min = int(area_common_array[c,r,2])
          ind_lat_max = int(area_common_array[c,r,3])
          #
          count            = 1           # index counter for 3rd index of area_common_array[c,r,3+count]
          emis_wrf_temp    = 0.0         # temp. variable of emission on a wrf grid
          ## area_all    = 0.0         # checking parameter : should be 1.0 after the loop
          #
          # Loops over "ilon" 'west-east (longitude)' and "ilat" 'south-north (latitude)' 
          # index of emission grid 
          # Only those containing a wrf cell (XLON[c,r],XLAT[c,r]) defined by 4 points [X,Y]
          #
          if np.amax(emis_tmp[ind_lon_min:ind_lon_max,ind_lat_min:ind_lat_max]) != 0.0:
            for ilon in np.arange(ind_lon_min,ind_lon_max):      
              for ilat in np.arange(ind_lat_min,ind_lat_max):    
                if emis_tmp[ilon,ilat] != 0.0 :     # Just to avoid a useless computation 
                  area_temp     = area_common_array[c,r,3+count]     # Area ratio (area of emission grid (ilon,ilat) shared with a wrf grid)/(area of emission grid)
                  emis_wrf_temp += area_temp*emis_tmp[ilon,ilat]
                #area_all    = area_all + area_temp
                count += 1
            # End of for ilon and ilat
            #print 'area_all = ',area_all
            #print c,'/',ng_we_wrf,' ',r,'/',ng_sn_wrf,emis_wrf
        #
        #---- Using area_common_dict ------------------------------------------------
        #
        if 1==1:
          emis_wrf_temp = 0.0
          #print(area_common_dict[c][r]['total_count'],'xxx')
          #area_temp = 0.0
          for idict in np.arange(area_common_dict[c][r]['total_count']):
            ilon = area_common_dict[c][r][idict]['ilon']
            ilat = area_common_dict[c][r][idict]['ilat']
            emis_wrf_temp += area_common_dict[c][r][idict]['area']*emis_tmp[ilon,ilat]
            #area_temp += area_common_dict[c][r][idict]['area']
          #print('area_temp = ',area_temp)
        #
        #---- End of Using area_common_dict ------------------------------------------
        # 
        emis_wrf_total[c,r] = emis_wrf_temp
      # End of loop r
    # End of loop c
    #
    #-----------------------------------------------------------
    #
    # print('key = ',datetime.now(),':',key)

    #
    #--- Create keyname : <spec>-GM-<datetime> ------
    #
    for spec_temp in spec_list:

      keyname = spec_temp+'-GFED-'+datetime_str
      GFED_unit = EFs[spec_temp]['unit'].split('/kg')[0]+'/m2/sec'

      if keyname not in emis_dict.keys():
        #print('keyname (GFED) no1= ',datetime.now(),':',keyname)

        emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
        emis_dict[keyname]['dimensions']={}
        emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
        emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
        #emis_dict[keyname]['dimensions']['time']       = 1         # of points in time series> e.g.:12
        emis_dict[keyname]['west_east']={}
        emis_dict[keyname]['west_east']['dtype']='i4'
        emis_dict[keyname]['west_east']['dims' ]=['west_east']
        emis_dict[keyname]['west_east']['units']=''
        emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
        emis_dict[keyname]['south_north']={}
        emis_dict[keyname]['south_north']['dtype']='i4'
        emis_dict[keyname]['south_north']['dims' ]=['south_north']
        emis_dict[keyname]['south_north']['units']=''
        emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
        emis_dict[keyname]['longitude']={}
        emis_dict[keyname]['longitude']['dtype']='f4'
        emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
        emis_dict[keyname]['longitude']['units']='degrees_east'
        emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
        emis_dict[keyname]['latitude']={}
        emis_dict[keyname]['latitude']['dtype']='f4'
        emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
        emis_dict[keyname]['latitude']['units']='degrees_east'
        emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
        #emis_dict[keyname]['time']={}
        #emis_dict[keyname]['time']['dtype']='i4'
        #emis_dict[keyname]['time']['dims' ]=['time']
        #emis_dict[keyname]['time']['data' ]=np.arange(12) #: [0,1,2,3,4,...,11]
        emis_dict[keyname]['voc']={}
        emis_dict[keyname]['voc']['dtype']='f4'
        #...emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] or ['west_east','south_north']
        # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] 
        emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
        # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
        if 'mol' in GFED_unit:
          units    = 'mol/hour/km2'
          emis_dict[keyname]['voc']['data' ]= emis_wrf_total[:,:]*EFs[spec_temp][sec_name]*3600.0*1.e6
        else:
          units    = 'ug/sec/m2'
          emis_dict[keyname]['voc']['data' ]= emis_wrf_total[:,:]*EFs[spec_temp][sec_name]*1.e6
        emis_dict[keyname]['voc']['units']= units
        
        if 'NO-GFED' == keyname[:7]:
          keyname2 = keyname.replace('NO-','NO2-')
          emis_dict[keyname2]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
          emis_dict[keyname2]['dimensions']={}
          emis_dict[keyname2]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
          emis_dict[keyname2]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
          #emis_dict[keyname2]['dimensions']['time']       = 1         # of points in time series> e.g.:12
          emis_dict[keyname2]['west_east']={}
          emis_dict[keyname2]['west_east']['dtype']='i4'
          emis_dict[keyname2]['west_east']['dims' ]=['west_east']
          emis_dict[keyname2]['west_east']['units']=''
          emis_dict[keyname2]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
          emis_dict[keyname2]['south_north']={}
          emis_dict[keyname2]['south_north']['dtype']='i4'
          emis_dict[keyname2]['south_north']['dims' ]=['south_north']
          emis_dict[keyname2]['south_north']['units']=''
          emis_dict[keyname2]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
          emis_dict[keyname2]['longitude']={}
          emis_dict[keyname2]['longitude']['dtype']='f4'
          emis_dict[keyname2]['longitude']['dims' ]=['west_east','south_north']
          emis_dict[keyname2]['longitude']['units']='degrees_east'
          emis_dict[keyname2]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
          emis_dict[keyname2]['latitude']={}
          emis_dict[keyname2]['latitude']['dtype']='f4'
          emis_dict[keyname2]['latitude']['dims' ]=['west_east','south_north']
          emis_dict[keyname2]['latitude']['units']='degrees_east'
          emis_dict[keyname2]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
          #emis_dict[keyname2]['time']={}
          #emis_dict[keyname2]['time']['dtype']='i4'
          #emis_dict[keyname2]['time']['dims' ]=['time']
          #emis_dict[keyname2]['time']['data' ]=np.arange(12) #: [0,1,2,3,4,...,11]
          emis_dict[keyname2]['voc']={}
          emis_dict[keyname2]['voc']['dtype']='f4'
          #...emis_dict[keyname2]['voc']['dims' ]=['west_east','south_north','time'] or ['west_east','south_north']
          # emis_dict[keyname2]['voc']['dims' ]=['west_east','south_north','time'] 
          emis_dict[keyname2]['voc']['dims' ]=['west_east','south_north']
          # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
          units    = 'mol/hour/km2'
          emis_dict[keyname2]['voc']['data' ]= 0.1/0.9*emis_wrf_total[:,:]*EFs[spec_temp][sec_name]*30.0/46.0*3600.0*1.e6
          emis_dict[keyname2]['voc']['units']= units
        ## END OF if 'NO-GFED' == keyname.split('-')[0]:
      else:
        #print('keyname (GFED) no2 = ',datetime.now(),':',keyname)
        if 'mol' in GFED_unit:
          emis_dict[keyname]['voc']['data' ]+= emis_wrf_total[:,:]*EFs[spec_temp][sec_name]*3600.0*1.e6
          print('EFs test check : ',EFs[spec_temp][sec_name], spec_temp, sec_name)
        else:
          emis_dict[keyname]['voc']['data' ]+= emis_wrf_total[:,:]*EFs[spec_temp][sec_name]*1.e6
        if 'NO-GFED' == keyname[:7]:
          keyname2 = keyname.replace('NO-','NO2-')
          emis_dict[keyname2]['voc']['data' ]+= 0.1/0.9*emis_wrf_total[:,:]*EFs[spec_temp][sec_name]*30.0/46.0*3600.0*1.e6
      ## END OF if keyname not in emis_dict.keys():
      
  return emis_dict

#-------------------------------------------------------------------
#-------------------------------------------------------------------
#
#  Read REAS Anthoropogenic emission data
#  and interpolate it on to WRF grid
#
def AddDict_REAS_Anth_Other_OnWRF( emis_dict, fname,\
                    area_common_array, area_common_dict, emis_cell_area,\
                    XLON, XLAT,sector,spec):
  #
  # Input  Unit : [t/month]
  # Output Unit : [g/m2/sec] or [ug/m2/sec] (BC,OC,PM25)
  #
  Mmol      = {'CO':28,'CH4':16,'SO2':64,'NH3':17,'NO':30,'NO2':46}
  ind_ratio = {'CO' :[0.55,0.45],'CH4':[0.92,0.08],'BC'  :[0.5 ,0.5 ],'OM':[0.5,0.5],'OC':[0.5,0.5],\
               'SO2':[0.71,0.29],'NH3':[0.08,0.92],'PM25':[0.43,0.57],'NO':[0.73,0.27],'NO2':[0.73,0.27]}
  #
  lon, lat, emis_tmp = read_REAS(fname)
  #
  #  
  #======================================================================
  #----------------------------------------------------------------------
  #
  #  Output: emis_nmvoc[nlon,nlat,12]
  #          emis_lon[nlon,nlat]
  #          emis_lat[nlon,nlat]
  #
  #
  emis_lon, emis_lat = np.meshgrid(lon,lat,indexing='ij')
  #
  #===========================================================================================
  #-------------------------------------------------------------------------------------------
  #  variable : 'emis_awb'
  day_of_month = np.array([31,29,31,30,31,30,31,31,30,31,30,31])
  if spec in ['CO','NH3','NOx','SO2','CH4']:
    #  unit : [ton/month] --> [g/sec/m2]
    units    = 'mol/sec/m2'
    for imonth in np.arange(12):
      emis_tmp[:,:,imonth] = np.divide(emis_tmp[:,:,imonth],emis_cell_area)
      #emis_tmp[:,:,imonth] /= emis_cell_area
      #emis_tmp[:,:,imonth] = emis_tmp[:,:,imonth]*1.e6/(60.0*60.0*24.0*day_of_month[imonth])
      emis_tmp[:,:,imonth] *= (1.e6/(60.0*60.0*24.0*day_of_month[imonth]))
  if spec in ['OM','OC','PM25','BC']:
    #  unit : [ton/month] --> [ug/sec/m2]
    units    = 'ug/sec/m2'
    for imonth in np.arange(12):
      emis_tmp[:,:,imonth] = np.divide(emis_tmp[:,:,imonth],emis_cell_area)
      #emis_tmp[:,:,imonth] /= emis_cell_area
      #emis_tmp[:,:,imonth] = emis_tmp[:,:,imonth]*1.e12/(60.0*60.0*24.0*day_of_month[imonth])
      emis_tmp[:,:,imonth] *= (1.e12/(60.0*60.0*24.0*day_of_month[imonth]))
  # 
  # New Grid Dimensions
  ng_we_wrf = XLON.shape[0]
  ng_sn_wrf = XLON.shape[1]
  #
  emis_wrf_total  = np.zeros(shape=(ng_we_wrf,ng_sn_wrf,12))
  #
  for imonth in np.arange(12):
    emis_wrf  = np.zeros(shape=(ng_we_wrf,ng_sn_wrf))
    for c in np.arange(ng_we_wrf):                          # Loop over west_east index of wrf grid (wrfinput_d01)
      for r in np.arange(ng_sn_wrf):                        # Loop over south_north index of wrf grid (wrfinput_d01)
        if 1==0:
          #... Common to WRF grid ........................
          #    Index ranges of 'west-east' and 'south-north' variables of emission grid. 
          #    
          ind_lon_min = int(area_common_array[c,r,0])
          ind_lon_max = int(area_common_array[c,r,1])
          ind_lat_min = int(area_common_array[c,r,2])
          ind_lat_max = int(area_common_array[c,r,3])
          #
          count            = 1           # index counter for 3rd index of area_common_array[c,r,3+count]
          emis_wrf_temp    = 0.0         # temp. variable of emission on a wrf grid
          ## area_all    = 0.0         # checking parameter : should be 1.0 after the loop
          #
          # Loops over "ilon" 'west-east (longitude)' and "ilat" 'south-north (latitude)' 
          # index of emission grid 
          # Only those containing a wrf cell (XLON[c,r],XLAT[c,r]) defined by 4 points [X,Y]
          #
          if np.amax(emis_tmp[ind_lon_min:ind_lon_max,ind_lat_min:ind_lat_max,imonth]) != 0.0:
            for ilon in np.arange(ind_lon_min,ind_lon_max):      
              for ilat in np.arange(ind_lat_min,ind_lat_max):    
                if emis_tmp[ilon,ilat,imonth] != 0.0 :     # Just to avoid a useless computation 
                  area_temp     = area_common_array[c,r,3+count]     # Area ratio (area of emission grid (ilon,ilat) shared with a wrf grid)/(area of emission grid)
                  emis_wrf_temp += area_temp*emis_tmp[ilon,ilat,imonth]
                #area_all    = area_all + area_temp
                count += 1
            # End of for ilon and ilat
            #print 'area_all = ',area_all
            #print c,'/',ng_we_wrf,' ',r,'/',ng_sn_wrf,emis_wrf
        #
        #---- Using area_common_dict ------------------------------------------------
        #
        if 1==1:
          emis_wrf_temp = 0.0
          for idict in np.arange(area_common_dict[c][r]['total_count']):
            ilon = area_common_dict[c][r][idict]['ilon']
            ilat = area_common_dict[c][r][idict]['ilat']
            emis_wrf_temp += area_common_dict[c][r][idict]['area']*emis_tmp[ilon,ilat,imonth]
        #
        #---- End of Using area_common_dict ------------------------------------------
        # 
        emis_wrf[c,r] = emis_wrf_temp
      # End of loop r
    # End of loop c
    emis_wrf_total[:,:,imonth] = emis_wrf[:,:]
  # End of loop imonth
  #
  #-----------------------------------------------------------
  #
  # print var
  sec  = sector
  sec_arr = [sec]
  spec_arr= [spec]
  coef_arr= [1.0]
  if sec == 'ind':
    sec_arr = ['ind1','ind2']
  if spec == 'NOx':
    spec_arr = ['NO','NO2']
    coef_arr = [0.9,0.1]        # NO/NO2 ratio 0.9:0.1

  for isec, sec in enumerate(sec_arr):
    for ispec, spec_temp in enumerate(spec_arr):
      Mm = 1.0
      # Molecular mass (g/mol) 
      if rank == 0:
        print(spec_temp, Mmol)
      if spec_temp in Mmol:
        Mm = Mmol[spec_temp]
      keyname = sec+'-REAS-'+spec_temp
      if rank == 0:
        print(keyname)
      if emis_dict.get(keyname) == None:
        emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
        emis_dict[keyname]['dimensions']={}
        emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
        emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
        emis_dict[keyname]['dimensions']['time']       = 12        # of points in time series> e.g.:12
        emis_dict[keyname]['west_east']={}
        emis_dict[keyname]['west_east']['dtype']='i4'
        emis_dict[keyname]['west_east']['dims' ]=['west_east']
        emis_dict[keyname]['west_east']['units']=''
        emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
        emis_dict[keyname]['south_north']={}
        emis_dict[keyname]['south_north']['dtype']='i4'
        emis_dict[keyname]['south_north']['dims' ]=['south_north']
        emis_dict[keyname]['south_north']['units']=''
        emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
        emis_dict[keyname]['longitude']={}
        emis_dict[keyname]['longitude']['dtype']='f4'
        emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
        emis_dict[keyname]['longitude']['units']='degrees_east'
        emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
        emis_dict[keyname]['latitude']={}
        emis_dict[keyname]['latitude']['dtype']='f4'
        emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
        emis_dict[keyname]['latitude']['units']='degrees_east'
        emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
        emis_dict[keyname]['time']={}
        emis_dict[keyname]['time']['dtype']='i4'
        emis_dict[keyname]['time']['dims' ]=['time']
        emis_dict[keyname]['time']['data' ]=np.arange(12) #: [0,1,2,3,4,...,11]
        emis_dict[keyname]['voc']={}
        emis_dict[keyname]['voc']['dtype']='f4'
        emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] or ['west_east','south_north']
        # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
        if spec_temp in ['OC','OM','PM25','BC']:
          units    = 'ug/sec/m2'
        else:
          units    = 'mol/sec/m2'
        emis_dict[keyname]['voc']['units']= units
        # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
        if 'ind' in sec:
          # unit [g] --> [mol] for chemical species
          emis_dict[keyname]['voc']['data' ]= emis_wrf_total[:,:,:]/Mm*ind_ratio[spec_temp][isec]*coef_arr[ispec]
        else:
          emis_dict[keyname]['voc']['data' ]= emis_wrf_total[:,:,:]/Mm*coef_arr[ispec] 
      else:
        if 'ind' in sec:
          # unit [g] --> [mol] for chemical species
          emis_dict[keyname]['voc']['data' ]+= emis_wrf_total[:,:,:]/Mm*ind_ratio[spec_temp][isec]*coef_arr[ispec] 
        else:
          emis_dict[keyname]['voc']['data' ]+= emis_wrf_total[:,:,:]/Mm*coef_arr[ispec] 
        
  return emis_dict

#-------------------------------------------------------------------
#
#  Read ECLIPSE Anthoropogenic emission data
#  and interpolate it on to WRF grid
#
def AddDict_ECLIPSE_Anth_Other_OnWRF_v2( emis_dict, fname, fname_monthly_partition, \
                    area_common_array, area_common_dict, emis_cell_area,\
                    XLON, XLAT):
  #
  # Year from fname e.g.) /directory/ETP_base_CLE_V6_CO_2005.nc
  #
  basename = os.path.basename(fname)
  year     = int(basename.split('.')[0].split('_')[-1])
  #
  #
  # Input Unit : [kt/year]
  #
  Mmol      = {'CO':28,'CH4':16,'SO2':64,'NH3':17,'NO':30,'NO2':46}
  ind_ratio = {'CO' :[0.55,0.45],'CH4':[0.92,0.08],'BC'  :[0.5 ,0.5 ],'OM':[0.5,0.5],\
               'SO2':[0.71,0.29],'NH3':[0.08,0.92],'PM25':[0.43,0.57],'NO':[0.73,0.27],'NO2':[0.73,0.27]}
  #
  temp_text = fname.split('_')
  #UPD_index = temp_text.index('UPD')
  V6_index  = temp_text.index('V6')
  spec_text = temp_text[V6_index+1]
  if spec_text in ['VOC','OC','PM10']:
    return emis_dict
  #
  nc    = Dataset(fname,'r',format='NETCDF4')
  #
  #-------------------------------------------------------------------
  # 
  #  Monthly pattern from ECLIPSE V6a
  #  file : /data/onishi/ECLIIPSE_V6b/ECLIPSE_V6a_monthly_pattern.nc
  #
  #      dimensions:
  #         lat = 360 ;
  #         lon = 720 ;
  #         time = 12 ;
  #      variables:
  #         double agr(time, lat, lon) ;
  #                 agr:long_name = "Monthly weights - Agriculture (animals, rice, soil)" ;
  #                 agr:sector = "Agriculture (animals, rice, soil)" ;
  #         double agr_NH3(time, lat, lon) ;
  #                 agr_NH3:long_name = "Monthly weights - Agriculture (NH3 from livestock and fertilizer use)" ;
  #                 agr_NH3:sector = "Agriculture (NH3 from livestock and fertilizer use)" ;
  #         double awb(time, lat, lon) ;
  #                 awb:long_name = "Monthly weights - Agriculture (waste burning on fields)" ;
  #                 awb:sector = "Agriculture (waste burning on fields)" ;
  #         double dom(time, lat, lon) ;
  #                 dom:long_name = "Monthly weights - Residential and commercial" ;
  #                 dom:sector = "Residential and commercial" ;
  #         double ene(time, lat, lon) ;
  #                 ene:long_name = "Monthly weights - Power plants, energy conversion, extraction" ;
  #                 ene:sector = "Power plants, energy conversion, extraction" ;
  #         double flr(time, lat, lon) ;
  #                 flr:long_name = "Monthly weights - Upstream production field flaring or venting" ;
  #                 flr:sector = "Upstream production field flaring or venting" ;
  #         double ind(time, lat, lon) ;
  #                 ind:long_name = "Monthly weights - Industry (combustion and processing)" ;
  #                 ind:sector = "Industry (combustion and processing)" ;
  #         double shp(time, lat, lon) ;
  #                 shp:long_name = "Monthly weights - International shipping" ;
  #                 shp:sector = "International shipping" ;
  #         double slv(time, lat, lon) ;
  #                 slv:long_name = "Monthly weights - Solvents" ;
  #                 slv:sector = "Solvents" ;
  #         double tra(time, lat, lon) ;
  #                 tra:long_name = "Monthly weights - Surface transportation" ;
  #                 tra:sector = "Surface transportation" ;
  #         double wst(time, lat, lon) ;
  #                 wst:long_name = "Monthly weights - Waste" ;
  #                 wst:sector = "Waste" ;
  #
  #--------------------------------------------------------------------
  #
  nc_m  = Dataset(fname_monthly_partition,'r',format='NETCDF4')
  mp_agr     = np.array(nc_m.variables['agr'])
  if 'NH3' in spec_text:
    mp_agr   = np.array(nc_m.variables['agr_NH3'])
  mp_awb     = np.array(nc_m.variables['awb'])
  mp_dom     = np.array(nc_m.variables['dom'])
  mp_ene     = np.array(nc_m.variables['ene'])
  mp_flr     = np.array(nc_m.variables['flr'])
  mp_ind     = np.array(nc_m.variables['ind'])
  mp_shp     = np.array(nc_m.variables['shp'])
  mp_slv     = np.array(nc_m.variables['slv'])
  mp_tra     = np.array(nc_m.variables['tra'])
  mp_wst     = np.array(nc_m.variables['wst'])
  nc_m.close()
  #
  #====================================================================
  #--------------------------------------------------------------------
  #
  #  Get a list of variable names 
  #  vars = ['lat','lon','emis_agr','emis_awb',,emis_dom','emis_ene','emis_flr','emis_ind','emis_tra','emis_wst','emis_all']
  #  dimension = [1,360,720]
  #  lon.shape[0] = 720
  #  lat.shape[0] = 360
  #  
  #  Output: nvars : # of variables 
  #          vars  : list of variable names
  #
  vars  = list(nc.variables.keys())
  nvars = len(vars)
  vars  = list(vars)
  # print vars
  #
  #=====================================================================
  #---------------------------------------------------------------------
  #
  #  assign variables : 'lon' and 'lat' 
  lon   = np.array(nc.variables["lon"])   # -180 < lon < 180
  lat   = np.array(nc.variables["lat"])   #  -90 < lat <  90
  #
  for iv, var in enumerate(vars):
    # print iv, var
    #
    # iv = 0 : 'lat'
    # iv = 1 : 'lon'
    # iv = 2 : 'time'
    if iv <= 2:
      continue
    #
    try:
      emis_tmp   = np.squeeze(np.array(nc.variables[var]))
      emis_tmp12 = np.zeros((12,emis_tmp.shape[0],emis_tmp.shape[1])) # [time,lat,lon] 
      for imonth in np.arange(12):
        emis_tmp12[imonth,:,:] = emis_tmp[:,:]
      # 
      # [kt/year] --> [kt/month]
      #
      if 'agr' in var:
        emis_tmp12 *= mp_agr
      if 'awb' in var:
        emis_tmp12 *= mp_awb
      if 'dom' in var:
        emis_tmp12 *= mp_dom
      if 'ene' in var:
        emis_tmp12 *= mp_ene
      if 'flr' in var:
        emis_tmp12 *= mp_flr
      if 'ind' in var:
        emis_tmp12 *= mp_ind
      if 'shp' in var:
        emis_tmp12 *= mp_shp
      if 'slv' in var:
        emis_tmp12 *= mp_slv
      if 'tra' in var:
        emis_tmp12 *= mp_tra
      if 'wst' in var:
        emis_tmp12 *= mp_wst
      #
      emis_tmp   = emis_tmp12
    except:
      print(var)
      print(nc.variables[var])
      raise
    #
    nanind = np.where(np.isnan(emis_tmp))               # Indices of NaN in emis_tmp
    #emis_vname.append(var)                             # Create a list of variable name 
    emis_tmp[nanind] = np.zeros_like(emis_tmp)[nanind]  # Replace NaN with 0
    #emis_nmvoc[:,:,iv-2] = np.transpose(emis_tmp[:,:]) # Set data to 'emis_nmvoc' [lon,lat,nvar]
    #
    #----------------------------------------------------------------------
    #
    #  Change the order of indexing
    #
    emis_tmp   = np.transpose(emis_tmp)     # [time,lat,lon] ---> [lon,lat,time]
    #  
    #======================================================================
    #----------------------------------------------------------------------
    #
    #  Expand longitude range from -180 < lon < 180 to -180 < lon < 360
    #
    #  Output: emis_nmvoc[nlon+nlon/2,nlat,nvars-2]
    #          emis_lon[nlon+nlon/2,nlat]
    #          emis_lat[nlon+nlon/2,nlat]
    #
    #
    emis_tmp   = np.concatenate((emis_tmp[:,:,:],emis_tmp[:emis_tmp.shape[0]//2,:,:]),axis=0)
    lon_temp   = np.concatenate((lon,lon[:lon.shape[0]//2]+360.0))
    emis_lon1  = lon_temp
    emis_lat1  = lat
    emis_lon, emis_lat = np.meshgrid(lon_temp,lat,indexing='ij')
    #
    #===========================================================================================
    #-------------------------------------------------------------------------------------------
    #  variable : 'emis_awb'
    if spec_text in ['CO','NH3','NOx','SO2','CH4']:
      #  unit : [kt/month] --> [g/sec/m2]
      #
      units    = 'mol/sec/m2'
      for imonth in np.arange(12):
        ndays    = monthrange(year,imonth+1)[1]
        emis_tmp[:,:,imonth] = np.divide(emis_tmp[:,:,imonth],emis_cell_area)
        emis_tmp[:,:,imonth] *= 1.e9/(60.0*60.0*24.0*ndays)
    if spec_text in ['OM','PM25','BC']:
      #  unit : [kt/month] --> [ug/sec/m2]
      units    = 'ug/sec/m2'
      for imonth in np.arange(12):
        ndays    = monthrange(year,imonth+1)[1]
        emis_tmp[:,:,imonth] = np.divide(emis_tmp[:,:,imonth],emis_cell_area)
        emis_tmp[:,:,imonth]*= 1.e15/(60.0*60.0*24.0*ndays)
    # 
    # New Grid Dimensions
    ng_we_wrf = XLON.shape[0]
    ng_sn_wrf = XLON.shape[1]
    #
    emis_wrf  = np.zeros(shape=(ng_we_wrf,ng_sn_wrf,12))
    # 
    #
    for imonth in np.arange(12):
      for c in np.arange(ng_we_wrf):                          # Loop over west_east index of wrf grid (wrfinput_d01)
        for r in np.arange(ng_sn_wrf):                        # Loop over south_north index of wrf grid (wrfinput_d01)
          ### if 1==0:
          ###   #... Common to WRF grid ........................
          ###   #    Index ranges of 'west-east' and 'south-north' variables of emission grid. 
          ###   #    
          ###   ind_lon_min = int(area_common_array[c,r,0])
          ###   ind_lon_max = int(area_common_array[c,r,1])
          ###   ind_lat_min = int(area_common_array[c,r,2])
          ###   ind_lat_max = int(area_common_array[c,r,3])
          ###   #
          ###   count            = 1           # index counter for 3rd index of area_common_array[c,r,3+count]
          ###   emis_wrf_temp    = 0.0         # temp. variable of emission on a wrf grid
          ###   ## area_all    = 0.0         # checking parameter : should be 1.0 after the loop
          ###   #
          ###   # Loops over "ilon" 'west-east (longitude)' and "ilat" 'south-north (latitude)' 
          ###   # index of emission grid 
          ###   # Only those containing a wrf cell (XLON[c,r],XLAT[c,r]) defined by 4 points [X,Y]
          ###   #
          ###   if np.amax(emis_tmp[ind_lon_min:ind_lon_max,ind_lat_min:ind_lat_max]) != 0.0:
          ###     for ilon in np.arange(ind_lon_min,ind_lon_max):      
          ###       for ilat in np.arange(ind_lat_min,ind_lat_max):    
          ###         if emis_tmp[ilon,ilat] != 0.0 :     # Just to avoid a useless computation 
          ###           area_temp     = area_common_array[c,r,3+count]     # Area ratio (area of emission grid (ilon,ilat) shared with a wrf grid)/(area of emission grid)
          ###           emis_wrf_temp = emis_wrf_temp+area_temp*emis_tmp[ilon,ilat]
          ###         #area_all    = area_all + area_temp
          ###         count += 1
          ###     # End of for ilon and ilat
          ###     #print 'area_all = ',area_all
          ###     #print c,'/',ng_we_wrf,' ',r,'/',ng_sn_wrf,emis_wrf
          #
          #---- Using area_common_dict ------------------------------------------------
          #
          if 1==1:
            emis_wrf_temp = 0.0
            for idict in np.arange(area_common_dict[c][r]['total_count']):
              ilon = area_common_dict[c][r][idict]['ilon']
              ilat = area_common_dict[c][r][idict]['ilat']
              emis_wrf_temp += area_common_dict[c][r][idict]['area']*emis_tmp[ilon,ilat,imonth]
          #
          #---- End of Using area_common_dict ------------------------------------------
          # 
          emis_wrf[c,r,imonth] = emis_wrf_temp
        # End of loop r
      # End of loop c
    # End of loop imonth
    #
    #-----------------------------------------------------------
    #
    # print var
    sec  = var.split('_')[1]
    spec = spec_text
    sec_arr = [sec]
    spec_arr= [spec]
    coef_arr= [1.0]
    if sec == 'ind':
      sec_arr = ['ind1','ind2']
    if spec == 'NOx':
      spec_arr = ['NO','NO2']
      coef_arr = [0.9,0.1]        # NO/NO2 ratio 0.9:0.1

    for isec, sec2 in enumerate(sec_arr):
      for ispec, spec2 in enumerate(spec_arr):
        Mm = 1.0
        # Molecular mass (g/mol) 
        if spec2 in Mmol:
          Mm = Mmol[spec2]
        keyname = sec2+'-ECLIPSE-'+spec2
        if rank == 0:
          print(keyname)
        emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
        emis_dict[keyname]['dimensions']={}
        emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
        emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
        emis_dict[keyname]['dimensions']['time']       = 12        # of points in time series> e.g.:12
        emis_dict[keyname]['west_east']={}
        emis_dict[keyname]['west_east']['dtype']='i4'
        emis_dict[keyname]['west_east']['dims' ]=['west_east']
        emis_dict[keyname]['west_east']['units']=''
        emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
        emis_dict[keyname]['south_north']={}
        emis_dict[keyname]['south_north']['dtype']='i4'
        emis_dict[keyname]['south_north']['dims' ]=['south_north']
        emis_dict[keyname]['south_north']['units']=''
        emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
        emis_dict[keyname]['longitude']={}
        emis_dict[keyname]['longitude']['dtype']='f4'
        emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
        emis_dict[keyname]['longitude']['units']='degrees_east'
        emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
        emis_dict[keyname]['latitude']={}
        emis_dict[keyname]['latitude']['dtype']='f4'
        emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
        emis_dict[keyname]['latitude']['units']='degrees_east'
        emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
        emis_dict[keyname]['time']={}
        emis_dict[keyname]['time']['dtype']='i4'
        emis_dict[keyname]['time']['dims' ]=['time']
        emis_dict[keyname]['time']['data' ]=np.arange(12) # [0,1,2,3,4,...,11]
        emis_dict[keyname]['voc']={}
        emis_dict[keyname]['voc']['dtype']='f4'
        # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] or ['west_east','south_north']
        emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time']
        if spec2 in ['OM','PM25','BC']:
          units    = 'ug/sec/m2'
        else:
          units    = 'mol/sec/m2'
        emis_dict[keyname]['voc']['units']= units
        # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
        if 'ind' in sec2:
          # unit [g] --> [mol] for chemical species
          emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:,:]/Mm*ind_ratio[spec2][isec]*coef_arr[ispec]
        else:
          emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:,:]/Mm*coef_arr[ispec] 
  return emis_dict

#-------------------------------------------------------------------
#
#  Read ECLIPSE Anthoropogenic emission data
#  and interpolate it on to WRF grid
#
def AddDict_ECLIPSE_Anth_Other_OnWRF( emis_dict, fname,\
                    area_common_array, area_common_dict, emis_cell_area,\
                    XLON, XLAT):
  #
  # Input Unit : [kt/year]
  #
  Mmol      = {'CO':28,'CH4':16,'SO2':64,'NH3':17,'NO':30,'NO2':46}
  ind_ratio = {'CO' :[0.55,0.45],'CH4':[0.92,0.08],'BC'  :[0.5 ,0.5 ],'OM':[0.5,0.5],\
               'SO2':[0.71,0.29],'NH3':[0.08,0.92],'PM25':[0.43,0.57],'NO':[0.73,0.27],'NO2':[0.73,0.27]}
  #
  temp_text = fname.split('_')
  #UPD_index = temp_text.index('UPD')
  V6_index  = temp_text.index('V6')
  spec_text = temp_text[V6_index+1]
  if spec_text in ['VOC','OC','PM10']:
    return emis_dict
  #
  nc    = Dataset(fname,'r',format='NETCDF4')
  #
  #====================================================================
  #--------------------------------------------------------------------
  #
  #  Get a list of variable names 
  #  vars = ['lat','lon','emis_agr','emis_awb',,emis_dom','emis_ene','emis_flr','emis_ind','emis_tra','emis_wst','emis_all']
  #  dimension = [1,360,720]
  #  lon.shape[0] = 720
  #  lat.shape[0] = 360
  #  
  #  Output: nvars : # of variables 
  #          vars  : list of variable names
  #
  vars  = list(nc.variables.keys())
  nvars = len(vars)
  vars  = list(vars)
  # print vars
  #
  #=====================================================================
  #---------------------------------------------------------------------
  #
  #  assign variables : 'lon' and 'lat' 
  lon   = np.array(nc.variables["lon"])   # -180 < lon < 180
  lat   = np.array(nc.variables["lat"])   #  -90 < lat <  90
  #
  for iv, var in enumerate(vars):
    # print iv, var
    #
    # iv = 0 : 'lat'
    # iv = 1 : 'lon'
    # iv = 2 : 'time'
    if iv <= 2:
      continue
    #
    try:
      emis_tmp = np.squeeze(np.array(nc.variables[var]))
    except:
      print(var)
      print(nc.variables[var])
      raise
    #
    nanind = np.where(np.isnan(emis_tmp))              # Indices of NaN in emis_tmp
    #emis_vname.append(var)                             # Create a list of variable name 
    emis_tmp[nanind] = np.zeros_like(emis_tmp)[nanind] # Replace NaN with 0
    #emis_nmvoc[:,:,iv-2] = np.transpose(emis_tmp[:,:]) # Set data to 'emis_nmvoc' [lon,lat,nvar]
    #
    #----------------------------------------------------------------------
    #
    #  Change the order of indexing
    #
    emis_tmp   = np.transpose(emis_tmp)     # [lat,lon] ---> [lon,lat]
    #  
    #======================================================================
    #----------------------------------------------------------------------
    #
    #  Expand longitude range from -180 < lon < 180 to -180 < lon < 360
    #
    #  Output: emis_nmvoc[nlon+nlon/2,nlat,nvars-2]
    #          emis_lon[nlon+nlon/2,nlat]
    #          emis_lat[nlon+nlon/2,nlat]
    #
    #
    emis_tmp   = np.concatenate((emis_tmp[:,:],emis_tmp[:emis_tmp.shape[0]//2,:]),axis=0)
    lon_temp   = np.concatenate((lon,lon[:lon.shape[0]//2]+360.0))
    emis_lon1  = lon_temp
    emis_lat1  = lat
    emis_lon, emis_lat = np.meshgrid(lon_temp,lat,indexing='ij')
    #
    #===========================================================================================
    #-------------------------------------------------------------------------------------------
    #  variable : 'emis_awb'
    if spec_text in ['CO','NH3','NOx','SO2','CH4']:
      #  unit : [kt/year] --> [g/sec/m2]
      #
      units    = 'mol/sec/m2'
      emis_tmp = np.divide(emis_tmp[:,:],emis_cell_area)
      emis_tmp *= 1.e9/(60.0*60.0*24.0*365.0)
    if spec_text in ['OM','PM25','BC']:
      #  unit : [kt/year] --> [ug/sec/m2]
      units    = 'ug/sec/m2'
      emis_tmp = np.divide(emis_tmp[:,:],emis_cell_area)
      emis_tmp *= 1.e15/(60.0*60.0*24.0*365.0)
    # 
    # New Grid Dimensions
    ng_we_wrf = XLON.shape[0]
    ng_sn_wrf = XLON.shape[1]
    #
    emis_wrf  = np.zeros(shape=(ng_we_wrf,ng_sn_wrf))
    # 
    #
    for c in np.arange(ng_we_wrf):                          # Loop over west_east index of wrf grid (wrfinput_d01)
      for r in np.arange(ng_sn_wrf):                        # Loop over south_north index of wrf grid (wrfinput_d01)
        if 1==0:
          #... Common to WRF grid ........................
          #    Index ranges of 'west-east' and 'south-north' variables of emission grid. 
          #    
          ind_lon_min = int(area_common_array[c,r,0])
          ind_lon_max = int(area_common_array[c,r,1])
          ind_lat_min = int(area_common_array[c,r,2])
          ind_lat_max = int(area_common_array[c,r,3])
          #
          count            = 1           # index counter for 3rd index of area_common_array[c,r,3+count]
          emis_wrf_temp    = 0.0         # temp. variable of emission on a wrf grid
          ## area_all    = 0.0         # checking parameter : should be 1.0 after the loop
          #
          # Loops over "ilon" 'west-east (longitude)' and "ilat" 'south-north (latitude)' 
          # index of emission grid 
          # Only those containing a wrf cell (XLON[c,r],XLAT[c,r]) defined by 4 points [X,Y]
          #
          if np.amax(emis_tmp[ind_lon_min:ind_lon_max,ind_lat_min:ind_lat_max]) != 0.0:
            for ilon in np.arange(ind_lon_min,ind_lon_max):      
              for ilat in np.arange(ind_lat_min,ind_lat_max):    
                if emis_tmp[ilon,ilat] != 0.0 :     # Just to avoid a useless computation 
                  area_temp     = area_common_array[c,r,3+count]     # Area ratio (area of emission grid (ilon,ilat) shared with a wrf grid)/(area of emission grid)
                  emis_wrf_temp = emis_wrf_temp+area_temp*emis_tmp[ilon,ilat]
                #area_all    = area_all + area_temp
                count += 1
            # End of for ilon and ilat
            #print 'area_all = ',area_all
            #print c,'/',ng_we_wrf,' ',r,'/',ng_sn_wrf,emis_wrf
        #
        #---- Using area_common_dict ------------------------------------------------
        #
        if 1==1:
          emis_wrf_temp = 0.0
          for idict in np.arange(area_common_dict[c][r]['total_count']):
            ilon = area_common_dict[c][r][idict]['ilon']
            ilat = area_common_dict[c][r][idict]['ilat']
            emis_wrf_temp += area_common_dict[c][r][idict]['area']*emis_tmp[ilon,ilat]
        #
        #---- End of Using area_common_dict ------------------------------------------
        # 
        emis_wrf[c,r] = emis_wrf_temp
      # End of loop r
    # End of loop c
    #
    #-----------------------------------------------------------
    #
    # print var
    sec  = var.split('_')[1]
    spec = spec_text
    sec_arr = [sec]
    spec_arr= [spec]
    coef_arr= [1.0]
    if sec == 'ind':
      sec_arr = ['ind1','ind2']
    if spec == 'NOx':
      spec_arr = ['NO','NO2']
      coef_arr = [0.9,0.1]        # NO/NO2 ratio 0.9:0.1

    for isec, sec2 in enumerate(sec_arr):
      for ispec, spec2 in enumerate(spec_arr):
        Mm = 1.0
        # Molecular mass (g/mol) 
        if spec2 in Mmol:
          Mm = Mmol[spec2]
        keyname = sec2+'-ECLIPSE-'+spec2
        if rank == 0:
          print(keyname)
        emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
        emis_dict[keyname]['dimensions']={}
        emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
        emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
        # emis_dict[keyname]['dimensions']['time']       =<# of points in time series> e.g.:12
        emis_dict[keyname]['west_east']={}
        emis_dict[keyname]['west_east']['dtype']='i4'
        emis_dict[keyname]['west_east']['dims' ]=['west_east']
        emis_dict[keyname]['west_east']['units']=''
        emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
        emis_dict[keyname]['south_north']={}
        emis_dict[keyname]['south_north']['dtype']='i4'
        emis_dict[keyname]['south_north']['dims' ]=['south_north']
        emis_dict[keyname]['south_north']['units']=''
        emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
        emis_dict[keyname]['longitude']={}
        emis_dict[keyname]['longitude']['dtype']='f4'
        emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
        emis_dict[keyname]['longitude']['units']='degrees_east'
        emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
        emis_dict[keyname]['latitude']={}
        emis_dict[keyname]['latitude']['dtype']='f4'
        emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
        emis_dict[keyname]['latitude']['units']='degrees_east'
        emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
        # emis_dict[keyname]['time']={}
        # emis_dict[keyname]['time']['dtype']='i4'
        # emis_dict[keyname]['time']['dims' ]=['time']
        # emis_dict[keyname]['time']['data' ]=np.arange(12) : [0,1,2,3,4,...,11]
        emis_dict[keyname]['voc']={}
        emis_dict[keyname]['voc']['dtype']='f4'
        # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] or ['west_east','south_north']
        emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
        if spec2 in ['OM','PM25','BC']:
          units    = 'ug/sec/m2'
        else:
          units    = 'mol/sec/m2'
        emis_dict[keyname]['voc']['units']= units
        # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
        if 'ind' in sec2:
          # unit [g] --> [mol] for chemical species
          emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:]/Mm*ind_ratio[spec2][isec]*coef_arr[ispec]
        else:
          emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:]/Mm*coef_arr[ispec] 
  return emis_dict

#-------------------------------------------------------------------
#
#  Read ECLIPSE Anthoropogenic emission data
#  and interpolate it on to WRF grid
#
def AddDict_ECLIPSE_Anth_VOC_OnWRF_v2( emis_dict, fname_voc, fname_monthly_partition, \
                    area_common_array, area_common_dict, emis_cell_area,\
                    XLON, XLAT):
  #
  # Year from fname e.g.) /directory/ETP_base_CLE_V6_VOC_2005.nc
  #
  print(fname_voc)
  bname = os.path.basename(fname_voc)
  print(bname)
  year     = int(bname.split('.')[0].split('_')[-1])
  #
  #
  # Input Unit : [kt/year]
  #
  nc    = Dataset(fname_voc,'r',format='NETCDF4')
  #
  #-------------------------------------------------------------------
  # 
  #  Monthly pattern from ECLIPSE V6a
  #  file : /data/onishi/ECLIIPSE_V6b/ECLIPSE_V6a_monthly_pattern.nc
  #
  #      dimensions:
  #         lat = 360 ;
  #         lon = 720 ;
  #         time = 12 ;
  #      variables:
  #         double agr(time, lat, lon) ;
  #                 agr:long_name = "Monthly weights - Agriculture (animals, rice, soil)" ;
  #                 agr:sector = "Agriculture (animals, rice, soil)" ;
  #         double agr_NH3(time, lat, lon) ;
  #                 agr_NH3:long_name = "Monthly weights - Agriculture (NH3 from livestock and fertilizer use)" ;
  #                 agr_NH3:sector = "Agriculture (NH3 from livestock and fertilizer use)" ;
  #         double awb(time, lat, lon) ;
  #                 awb:long_name = "Monthly weights - Agriculture (waste burning on fields)" ;
  #                 awb:sector = "Agriculture (waste burning on fields)" ;
  #         double dom(time, lat, lon) ;
  #                 dom:long_name = "Monthly weights - Residential and commercial" ;
  #                 dom:sector = "Residential and commercial" ;
  #         double ene(time, lat, lon) ;
  #                 ene:long_name = "Monthly weights - Power plants, energy conversion, extraction" ;
  #                 ene:sector = "Power plants, energy conversion, extraction" ;
  #         double flr(time, lat, lon) ;
  #                 flr:long_name = "Monthly weights - Upstream production field flaring or venting" ;
  #                 flr:sector = "Upstream production field flaring or venting" ;
  #         double ind(time, lat, lon) ;
  #                 ind:long_name = "Monthly weights - Industry (combustion and processing)" ;
  #                 ind:sector = "Industry (combustion and processing)" ;
  #         double shp(time, lat, lon) ;
  #                 shp:long_name = "Monthly weights - International shipping" ;
  #                 shp:sector = "International shipping" ;
  #         double slv(time, lat, lon) ;
  #                 slv:long_name = "Monthly weights - Solvents" ;
  #                 slv:sector = "Solvents" ;
  #         double tra(time, lat, lon) ;
  #                 tra:long_name = "Monthly weights - Surface transportation" ;
  #                 tra:sector = "Surface transportation" ;
  #         double wst(time, lat, lon) ;
  #                 wst:long_name = "Monthly weights - Waste" ;
  #                 wst:sector = "Waste" ;
  #
  #--------------------------------------------------------------------
  #
  nc_m  = Dataset(fname_monthly_partition,'r',format='NETCDF4')
  mp_agr     = np.array(nc_m.variables['agr'])
  mp_awb     = np.array(nc_m.variables['awb'])
  mp_dom     = np.array(nc_m.variables['dom'])
  mp_ene     = np.array(nc_m.variables['ene'])
  mp_flr     = np.array(nc_m.variables['flr'])
  mp_ind     = np.array(nc_m.variables['ind'])
  mp_shp     = np.array(nc_m.variables['shp'])
  mp_slv     = np.array(nc_m.variables['slv'])
  mp_tra     = np.array(nc_m.variables['tra'])
  mp_wst     = np.array(nc_m.variables['wst'])
  nc_m.close()
  #
  #====================================================================
  #--------------------------------------------------------------------
  #
  #  Get a list of variable names 
  #  vars = ['lat','lon','emis_awb','emis_dom','emis_ene','emis_ind','emis_slv','emis_tra','emis_wst','emis_all']
  #  nvars= 10
  #  dimension = [1,360,720]
  #  lon.shape[0] = 720
  #  lat.shape[0] = 360
  #  
  #  Output: nvars : # of variables 
  #          vars  : list of variable names
  #
  vars  = list(nc.variables.keys())
  nvars = len(vars)
  vars  = list(vars)
  # print vars
  #
  #=====================================================================
  #---------------------------------------------------------------------
  #
  #  assign variables : 'lon' and 'lat' 
  lon   = np.array(nc.variables["lon"])   # -180 < lon < 180
  lat   = np.array(nc.variables["lat"])   #  -90 < lat <  90
  #
  for iv, var in enumerate(vars):
    # print iv, var
    #
    # iv = 0 : 'lat'
    # iv = 1 : 'lon'
    # iv = 2 : 'time'
    if iv <= 2:
      continue
    #
    #  assign variable : 'emis_awb','emis_dom','emis_ene','emis_ind','emis_slv','emis_tra','emis_wst','emis_all'
    emis_tmp = np.squeeze(np.array(nc.variables[var]))
    emis_tmp12 = np.zeros((12,emis_tmp.shape[0],emis_tmp.shape[1])) # [time,lat,lon] 
    for imonth in np.arange(12):
      emis_tmp12[imonth,:,:] = emis_tmp[:,:]
    # 
    # [kt/year] --> [kt/month]
    #
    if 'agr' in var:
      emis_tmp12 *= mp_agr
    if 'awb' in var:
      emis_tmp12 *= mp_awb
    if 'dom' in var:
      emis_tmp12 *= mp_dom
    if 'ene' in var:
      emis_tmp12 *= mp_ene
    if 'flr' in var:
      emis_tmp12 *= mp_flr
    if 'ind' in var:
      emis_tmp12 *= mp_ind
    if 'shp' in var:
      emis_tmp12 *= mp_shp
    if 'slv' in var:
      emis_tmp12 *= mp_slv
    if 'tra' in var:
      emis_tmp12 *= mp_tra
    if 'wst' in var:
      emis_tmp12 *= mp_wst
    #
    emis_tmp   = emis_tmp12
    #
    ### if np.array(nc.variables[var]).ndim == 3:
    ###   emis_tmp = np.array(nc.variables[var])[0,:,:] 
    #
    try:
      nanind = np.where(np.isnan(emis_tmp))              # Indices of NaN in emis_tmp
      emis_tmp[nanind] = np.zeros_like(emis_tmp)[nanind] # Replace NaN with 0
    except:
      pass
    #
    #----------------------------------------------------------------------
    #
    #  Change the order of indexing
    #
    emis_tmp   = np.transpose(emis_tmp)     # [time,lat,lon] ---> [lon,lat,time]
    #  
    #======================================================================
    #----------------------------------------------------------------------
    #
    #  Expand longitude range from -180 < lon < 180 to -180 < lon < 360
    #
    #  Output: emis_nmvoc[nlon+nlon/2,nlat,nvars-2]
    #          emis_lon[nlon+nlon/2,nlat]
    #          emis_lat[nlon+nlon/2,nlat]
    #
    #
    emis_tmp   = np.concatenate((emis_tmp[:,:,:],emis_tmp[:emis_tmp.shape[0]//2,:,:]),axis=0)
    lon_temp   = np.concatenate((lon,lon[:lon.shape[0]//2]+360.0))
    emis_lon1  = lon_temp
    emis_lat1  = lat
    emis_lon, emis_lat = np.meshgrid(lon_temp,lat,indexing='ij')
    #
    #===========================================================================================
    #-------------------------------------------------------------------------------------------
    #  variable : 'emis_awb'
    #  unit : [kt/month] --> [kg/sec/m2]
    #  emis_cell_area : area [m2] of each cell
    #
    for imonth in np.arange(12):
      ndays     = monthrange(year,imonth+1)[1]
      emis_tmp[:,:,imonth]  = np.divide(emis_tmp[:,:,imonth],emis_cell_area)
      emis_tmp[:,:,imonth] *= 1.e6/(60.0*60.0*24.0*ndays)
    # 
    # New Grid Dimensions
    ng_we_wrf = XLON.shape[0]
    ng_sn_wrf = XLON.shape[1]
    #
    emis_wrf  = np.zeros(shape=(ng_we_wrf,ng_sn_wrf,12))
    # 
    for imonth in np.arange(12):
      for c in np.arange(ng_we_wrf):                          # Loop over west_east index of wrf grid (wrfinput_d01)
        for r in np.arange(ng_sn_wrf):                        # Loop over south_north index of wrf grid (wrfinput_d01)
          ### if 1==0:
          ###   #... Common to WRF grid ........................
          ###   #    Index ranges of 'west-east' and 'south-north' variables of emission grid. 
          ###   #    
          ###   ind_lon_min = int(area_common_array[c,r,0])
          ###   ind_lon_max = int(area_common_array[c,r,1])
          ###   ind_lat_min = int(area_common_array[c,r,2])
          ###   ind_lat_max = int(area_common_array[c,r,3])
          ###   #
          ###   count            = 1           # index counter for 3rd index of area_common_array[c,r,3+count]
          ###   emis_wrf_temp    = 0.0         # temp. variable of emission on a wrf grid
          ###   ## area_all    = 0.0         # checking parameter : should be 1.0 after the loop
          ###   #
          ###   # Loops over "ilon" 'west-east (longitude)' and "ilat" 'south-north (latitude)' 
          ###   # index of emission grid 
          ###   # Only those containing a wrf cell (XLON[c,r],XLAT[c,r]) defined by 4 points [X,Y]
          ###   #
          ###   if np.amax(emis_tmp[ind_lon_min:ind_lon_max,ind_lat_min:ind_lat_max]) != 0.0:
          ###     for ilon in np.arange(ind_lon_min,ind_lon_max):      
          ###       for ilat in np.arange(ind_lat_min,ind_lat_max):    
          ###         if emis_tmp[ilon,ilat] != 0.0 :     # Just to avoid a useless computation 
          ###           area_temp     = area_common_array[c,r,3+count]     # Area ratio (area of emission grid (ilon,ilat) shared with a wrf grid)/(area of emission grid)
          ###           emis_wrf_temp = emis_wrf_temp+area_temp*emis_tmp[ilon,ilat]
          ###         #area_all    = area_all + area_temp
          ###         count += 1
          ###     # End of for ilon and ilat
          ###     #print 'area_all = ',area_all
          ###     #print c,'/',ng_we_wrf,' ',r,'/',ng_sn_wrf,emis_wrf
          #
          #---- Using area_common_dict ------------------------------------------------
          #
          if 1==1:
            emis_wrf_temp = 0.0
            for idict in np.arange(area_common_dict[c][r]['total_count']):
              ilon = area_common_dict[c][r][idict]['ilon']
              ilat = area_common_dict[c][r][idict]['ilat']
              emis_wrf_temp += area_common_dict[c][r][idict]['area']*emis_tmp[ilon,ilat,imonth]
          #
          #---- End of Using area_common_dict ------------------------------------------
          # 
          emis_wrf[c,r,imonth] = emis_wrf_temp
        # End of loop r
      # End of loop c
    # End of loop imonth
    #
    #-----------------------------------------------------------
    #
    keyname = var.split('_')[1]+'-ECLIPSE-VOC'
    if rank == 0: 
      print(keyname)
    emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
    emis_dict[keyname]['dimensions']={}
    emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
    emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
    emis_dict[keyname]['dimensions']['time']       = 12        # of points in time series> e.g.:12
    emis_dict[keyname]['west_east']={}
    emis_dict[keyname]['west_east']['dtype']='i4'
    emis_dict[keyname]['west_east']['dims' ]=['west_east']
    emis_dict[keyname]['west_east']['units']=''
    emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
    emis_dict[keyname]['south_north']={}
    emis_dict[keyname]['south_north']['dtype']='i4'
    emis_dict[keyname]['south_north']['dims' ]=['south_north']
    emis_dict[keyname]['south_north']['units']=''
    emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
    emis_dict[keyname]['longitude']={}
    emis_dict[keyname]['longitude']['dtype']='f4'
    emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
    emis_dict[keyname]['longitude']['units']='degrees_east'
    emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
    emis_dict[keyname]['latitude']={}
    emis_dict[keyname]['latitude']['dtype']='f4'
    emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
    emis_dict[keyname]['latitude']['units']='degrees_east'
    emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
    emis_dict[keyname]['time']={}
    emis_dict[keyname]['time']['dtype']='i4'
    emis_dict[keyname]['time']['dims' ]=['time']
    emis_dict[keyname]['time']['data' ]=np.arange(12) # [0,1,2,3,4,...,11]
    emis_dict[keyname]['voc']={}
    emis_dict[keyname]['voc']['dtype']='f4'
    # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] or ['west_east','south_north']
    emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time']
    emis_dict[keyname]['voc']['units']='kg/sec/m2'
    emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
  
  return emis_dict

#-------------------------------------------------------------------
#-------------------------------------------------------------------
#
#  Read ECLIPSE Anthoropogenic emission data
#  and interpolate it on to WRF grid
#
def AddDict_ECLIPSE_Anth_VOC_OnWRF( emis_dict, fname_voc,\
                    area_common_array, area_common_dict, emis_cell_area,\
                    XLON, XLAT):
  #
  # Input Unit : [kt/year]
  #

  nc    = Dataset(fname_voc,'r',format='NETCDF4')
  #
  #====================================================================
  #--------------------------------------------------------------------
  #
  #  Get a list of variable names 
  #  vars = ['lat','lon','emis_awb','emis_dom','emis_ene','emis_ind','emis_slv','emis_tra','emis_wst','emis_all']
  #  nvars= 10
  #  dimension = [1,360,720]
  #  lon.shape[0] = 720
  #  lat.shape[0] = 360
  #  
  #  Output: nvars : # of variables 
  #          vars  : list of variable names
  #
  vars  = list(nc.variables.keys())
  nvars = len(vars)
  vars  = list(vars)
  # print vars
  #
  #=====================================================================
  #---------------------------------------------------------------------
  #
  #  assign variables : 'lon' and 'lat' 
  lon   = np.array(nc.variables["lon"])   # -180 < lon < 180
  lat   = np.array(nc.variables["lat"])   #  -90 < lat <  90
  #
  for iv, var in enumerate(vars):
    # print iv, var
    #
    # iv = 0 : 'lat'
    # iv = 1 : 'lon'
    # iv = 2 : 'time'
    if iv <= 2:
      continue
    #
    #  assign variable : 'emis_awb','emis_dom','emis_ene','emis_ind','emis_slv','emis_tra','emis_wst','emis_all'
    emis_tmp = np.squeeze(np.array(nc.variables[var]))
    ### if np.array(nc.variables[var]).ndim == 3:
    ###   emis_tmp = np.array(nc.variables[var])[0,:,:] 
    #
    try:
      nanind = np.where(np.isnan(emis_tmp))              # Indices of NaN in emis_tmp
      #emis_vname.append(var)                             # Create a list of variable name 
      emis_tmp[nanind] = np.zeros_like(emis_tmp)[nanind] # Replace NaN with 0
      #emis_nmvoc[:,:,iv-2] = np.transpose(emis_tmp[:,:]) # Set data to 'emis_nmvoc' [lon,lat,nvar]
    except:
      pass
    #
    #----------------------------------------------------------------------
    #
    #  Change the order of indexing
    #
    emis_tmp   = np.transpose(emis_tmp)     # [lat,lon] ---> [lon,lat]
    #  
    #======================================================================
    #----------------------------------------------------------------------
    #
    #  Expand longitude range from -180 < lon < 180 to -180 < lon < 360
    #
    #  Output: emis_nmvoc[nlon+nlon/2,nlat,nvars-2]
    #          emis_lon[nlon+nlon/2,nlat]
    #          emis_lat[nlon+nlon/2,nlat]
    #
    #
    emis_tmp   = np.concatenate((emis_tmp[:,:],emis_tmp[:emis_tmp.shape[0]//2,:]),axis=0)
    lon_temp   = np.concatenate((lon,lon[:lon.shape[0]//2]+360.0))
    emis_lon1  = lon_temp
    emis_lat1  = lat
    emis_lon, emis_lat = np.meshgrid(lon_temp,lat,indexing='ij')
    #
    #===========================================================================================
    #-------------------------------------------------------------------------------------------
    #  variable : 'emis_awb'
    #  unit : [kt/year] --> [kg/sec/m2]
    #  emis_cell_area : area [m2] of each cell
    #
    emis_tmp  = np.divide(emis_tmp[:,:],emis_cell_area)
    emis_tmp *= 1.e6/(60.0*60.0*24.0*365.0)
    # 
    # New Grid Dimensions
    ng_we_wrf = XLON.shape[0]
    ng_sn_wrf = XLON.shape[1]
    #
    emis_wrf  = np.zeros(shape=(ng_we_wrf,ng_sn_wrf))
    # 
    for c in np.arange(ng_we_wrf):                          # Loop over west_east index of wrf grid (wrfinput_d01)
      for r in np.arange(ng_sn_wrf):                        # Loop over south_north index of wrf grid (wrfinput_d01)
        if 1==0:
          #... Common to WRF grid ........................
          #    Index ranges of 'west-east' and 'south-north' variables of emission grid. 
          #    
          ind_lon_min = int(area_common_array[c,r,0])
          ind_lon_max = int(area_common_array[c,r,1])
          ind_lat_min = int(area_common_array[c,r,2])
          ind_lat_max = int(area_common_array[c,r,3])
          #
          count            = 1           # index counter for 3rd index of area_common_array[c,r,3+count]
          emis_wrf_temp    = 0.0         # temp. variable of emission on a wrf grid
          ## area_all    = 0.0         # checking parameter : should be 1.0 after the loop
          #
          # Loops over "ilon" 'west-east (longitude)' and "ilat" 'south-north (latitude)' 
          # index of emission grid 
          # Only those containing a wrf cell (XLON[c,r],XLAT[c,r]) defined by 4 points [X,Y]
          #
          if np.amax(emis_tmp[ind_lon_min:ind_lon_max,ind_lat_min:ind_lat_max]) != 0.0:
            for ilon in np.arange(ind_lon_min,ind_lon_max):      
              for ilat in np.arange(ind_lat_min,ind_lat_max):    
                if emis_tmp[ilon,ilat] != 0.0 :     # Just to avoid a useless computation 
                  area_temp     = area_common_array[c,r,3+count]     # Area ratio (area of emission grid (ilon,ilat) shared with a wrf grid)/(area of emission grid)
                  emis_wrf_temp = emis_wrf_temp+area_temp*emis_tmp[ilon,ilat]
                #area_all    = area_all + area_temp
                count += 1
            # End of for ilon and ilat
            #print 'area_all = ',area_all
            #print c,'/',ng_we_wrf,' ',r,'/',ng_sn_wrf,emis_wrf
        #
        #---- Using area_common_dict ------------------------------------------------
        #
        if 1==1:
          emis_wrf_temp = 0.0
          for idict in np.arange(area_common_dict[c][r]['total_count']):
            ilon = area_common_dict[c][r][idict]['ilon']
            ilat = area_common_dict[c][r][idict]['ilat']
            emis_wrf_temp += area_common_dict[c][r][idict]['area']*emis_tmp[ilon,ilat]
        #
        #---- End of Using area_common_dict ------------------------------------------
        # 
        emis_wrf[c,r] = emis_wrf_temp
      # End of loop r
    # End of loop c
    #
    #-----------------------------------------------------------
    #
    keyname = var.split('_')[1]+'-ECLIPSE-VOC'
    if rank == 0: 
      print(keyname)
    emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
    emis_dict[keyname]['dimensions']={}
    emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
    emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
    # emis_dict[keyname]['dimensions']['time']       =<# of points in time series> e.g.:12
    emis_dict[keyname]['west_east']={}
    emis_dict[keyname]['west_east']['dtype']='i4'
    emis_dict[keyname]['west_east']['dims' ]=['west_east']
    emis_dict[keyname]['west_east']['units']=''
    emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
    emis_dict[keyname]['south_north']={}
    emis_dict[keyname]['south_north']['dtype']='i4'
    emis_dict[keyname]['south_north']['dims' ]=['south_north']
    emis_dict[keyname]['south_north']['units']=''
    emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
    emis_dict[keyname]['longitude']={}
    emis_dict[keyname]['longitude']['dtype']='f4'
    emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
    emis_dict[keyname]['longitude']['units']='degrees_east'
    emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
    emis_dict[keyname]['latitude']={}
    emis_dict[keyname]['latitude']['dtype']='f4'
    emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
    emis_dict[keyname]['latitude']['units']='degrees_east'
    emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
    # emis_dict[keyname]['time']={}
    # emis_dict[keyname]['time']['dtype']='i4'
    # emis_dict[keyname]['time']['dims' ]=['time']
    # emis_dict[keyname]['time']['data' ]=np.arange(12) : [0,1,2,3,4,...,11]
    emis_dict[keyname]['voc']={}
    emis_dict[keyname]['voc']['dtype']='f4'
    # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] or ['west_east','south_north']
    emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
    emis_dict[keyname]['voc']['units']='kg/sec/m2'
    emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
  
  return emis_dict

#-------------------------------------------------------------------
#
#  Read HTAP Anthoropogenic Other emission data (except for AIR/SHIPS) 
#  and interpolate it on to WRF grid
#
def AddDict_HTAP_Anth_Other_OnWRF( emis_dict, fnames_voc,\
                    area_common_array, area_common_dict, emis_cell_area,\
                    XLON, XLAT,spec):
  Mmol      = {'CO':28,'CH4':16,'SO2':64,'NH3':17,'NO':30,'NO2':46}
  ind_ratio = {'CO' :[0.55,0.45],'CH4':[0.92,0.08],'BC'  :[0.5 ,0.5 ],'OM':[0.5,0.5],'OC':[0.5,0.5],\
               'SO2':[0.71,0.29],'NH3':[0.08,0.92],'PM25':[0.43,0.57],'NO':[0.73,0.27],'NO2':[0.73,0.27]}
  if spec in ['VOC','PM10']:
    return emis_dict

  if 'SHIPS' in fnames_voc[0] or 'AIR' in fnames_voc[0]:
    sec = 'shp'
    if rank == 0:
      print(fnames_voc)

    if len(fnames_voc) != 1:
      print('HTAP  files should consist of only 1 file')
      print(fnames_voc)
      sys.exit(1)
    #
    #
    # New Grid Dimensions
    ng_we_wrf = XLON.shape[0]
    ng_sn_wrf = XLON.shape[1]
    #
    fname_voc = fnames_voc[0]
    if rank == 0:
      print('fname_voc = ',fname_voc)
    ### Test
    ## if imonth > 0:
    ##   continue
    ## print fname_voc
    ## print os.path.basename(fname_voc)
    nc    = Dataset(fname_voc,'r',format='NETCDF4')
    #
    #====================================================================
    #--------------------------------------------------------------------
    #
    #  Get a list of variable names 
    #  vars = ['lat','lon','emi_nmvoc']
    #  nvars=3 
    #  dimension = [1800,3600]
    #  lon.shape[0] = 3600
    #  lat.shape[0] = 1800
    #  
    #  Output: nvars : # of variables 
    #          vars  : list of variable names
    #
    vars  = list(nc.variables.keys())
    nvars = len(vars)
    vars  = list(vars)
    # print vars
    #
    #=====================================================================
    #---------------------------------------------------------------------
    #
    #  assign variables : 'lon' and 'lat' 
    lon   = np.array(nc.variables["lon"])   #    0 < lon < 360
    lat   = np.array(nc.variables["lat"])   #  -90 < lat <  90
    #
    for iv, var in enumerate(vars):
      # print iv, var
      #
      #  assign variable : 'emi_nmvoc'
      emis_tmp = np.array(nc.variables[var]) 
      #
      # iv = 0 : 'lat'
      # iv = 1 : 'lon'
      if iv <= 1:
        continue
      #
      nanind = np.where(np.isnan(emis_tmp))              # Indices of NaN in emis_tmp
      #emis_vname.append(var)                             # Create a list of variable name 
      emis_tmp[nanind] = np.zeros_like(emis_tmp)[nanind] # Replace NaN with 0
      #emis_nmvoc[:,:,iv-2] = np.transpose(emis_tmp[:,:]) # Set data to 'emis_nmvoc' [lon,lat,nvar]
      #
      #----------------------------------------------------------------------
      #
      #  Change the order of indexing
      #
      emis_tmp   = np.transpose(emis_tmp)     # [lat,lon] ---> [lon,lat]
      #  
      #======================================================================
      #----------------------------------------------------------------------
      #
      #  Expand longitude range from 0 < lon < 360 to -180 < lon < 360
      #
      #  Output: emis_nmvoc[nlon+nlon/2,nlat,nvars-2]
      #          emis_lon[nlon+nlon/2,nlat]
      #          emis_lat[nlon+nlon/2,nlat]
      #
      #
      emis_tmp   = np.concatenate((emis_tmp[emis_tmp.shape[0]//2:,:],emis_tmp[:,:]),axis=0)
      lon_temp   = np.concatenate((lon[lon.shape[0]//2:]-360.0,lon))
      emis_lon1  = lon_temp
      emis_lat1  = lat
      emis_lon, emis_lat = np.meshgrid(lon_temp,lat,indexing='ij')
      #
      #===========================================================================================
      #-------------------------------------------------------------------------------------------
      #  variable : 'emis_awb'
      #  unit : [kg/m2/sec]
      #  emis_cell_area : area [m2] of each cell
      # 
      if spec in ['CO','NH3','NOx','SO2']:
        #  unit : [kg/m2/sec] -> [g/m2/sec]
        #
        units = 'g/m2/sec'
        emis_tmp *= 1.e3
      if spec in ['OC','PM25','BC']:
        #  unit : [kg/m2/sec] -> [ug/m2/sec]
        #
        units = 'ug/m2/sec'
        emis_tmp *= 1.e9

      # New Grid Dimensions
      ng_we_wrf = XLON.shape[0]
      ng_sn_wrf = XLON.shape[1]
      #
      emis_wrf  = np.zeros(shape=(ng_we_wrf,ng_sn_wrf))
      # 
      for c in np.arange(ng_we_wrf):                          # Loop over west_east index of wrf grid (wrfinput_d01)
        for r in np.arange(ng_sn_wrf):                        # Loop over south_north index of wrf grid (wrfinput_d01)
          if 1==0:
            #... Common to WRF grid ........................
            #    Index ranges of 'west-east' and 'south-north' variables of emission grid. 
            #    
            ind_lon_min = int(area_common_array[c,r,0])
            ind_lon_max = int(area_common_array[c,r,1])
            ind_lat_min = int(area_common_array[c,r,2])
            ind_lat_max = int(area_common_array[c,r,3])
            #
            count            = 1           # index counter for 3rd index of area_common_array[c,r,3+count]
            emis_wrf_temp    = 0.0         # temp. variable of emission on a wrf grid
            ## area_all    = 0.0         # checking parameter : should be 1.0 after the loop
            #
            # Loops over "ilon" 'west-east (longitude)' and "ilat" 'south-north (latitude)' 
            # index of emission grid 
            # Only those containing a wrf cell (XLON[c,r],XLAT[c,r]) defined by 4 points [X,Y]
            #
            if np.amax(emis_tmp[ind_lon_min:ind_lon_max,ind_lat_min:ind_lat_max]) != 0.0:
              for ilon in np.arange(ind_lon_min,ind_lon_max):      
                for ilat in np.arange(ind_lat_min,ind_lat_max):    
                  if emis_tmp[ilon,ilat] != 0.0 :     # Just to avoid a useless computation 
                    area_temp     = area_common_array[c,r,3+count]     # Area ratio (area of emission grid (ilon,ilat) shared with a wrf grid)/(area of emission grid)
                    emis_wrf_temp = emis_wrf_temp+area_temp*emis_tmp[ilon,ilat]
                  #area_all    = area_all + area_temp
                  count += 1
              # End of for ilon and ilat
              #print 'area_all = ',area_all
              #print c,'/',ng_we_wrf,' ',r,'/',ng_sn_wrf,emis_wrf
        #
        #---- Using area_common_dict ------------------------------------------------
        #
        if 1==1:
          emis_wrf_temp = 0.0
          for idict in np.arange(area_common_dict[c][r]['total_count']):
            ilon = area_common_dict[c][r][idict]['ilon']
            ilat = area_common_dict[c][r][idict]['ilat']
            emis_wrf_temp += area_common_dict[c][r][idict]['area']*emis_tmp[ilon,ilat]
        #
        #---- End of Using area_common_dict ------------------------------------------
        # 
          emis_wrf[c,r] = emis_wrf_temp
        # End of loop r
      # End of loop c
      #
      #-----------------------------------------------------------
      #
    # END of loop over var : for iv, var in enumerate(vars):
    nc.close()

    sec_arr = [sec]
    spec_arr= [spec]
    coef_arr= [1.0]
    if sec == 'ind':
      sec_arr = ['ind1','ind2']
    if spec == 'NOx':
      spec_arr = ['NO','NO2']
      coef_arr = [0.9,0.1]

    for isec, sec2 in enumerate(sec_arr):
      for ispec, spec2 in enumerate(spec_arr):
        Mm = 1.0
        # Molecular mass (g/mol)
        if spec2 in Mmol:
          Mm = Mmol[spec2]
        keyname = sec2+'-HTAP-'+spec2
        if emis_dict.get(keyname) == None:
          emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
          emis_dict[keyname]['dimensions']={}
          emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
          emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
          #emis_dict[keyname]['dimensions']['time']       = 12        # of points in time series> e.g.:12
          emis_dict[keyname]['west_east']={}
          emis_dict[keyname]['west_east']['dtype']='i4'
          emis_dict[keyname]['west_east']['dims' ]=['west_east']
          emis_dict[keyname]['west_east']['units']=''
          emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
          emis_dict[keyname]['south_north']={}
          emis_dict[keyname]['south_north']['dtype']='i4'
          emis_dict[keyname]['south_north']['dims' ]=['south_north']
          emis_dict[keyname]['south_north']['units']=''
          emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
          emis_dict[keyname]['longitude']={}
          emis_dict[keyname]['longitude']['dtype']='f4'
          emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
          emis_dict[keyname]['longitude']['units']='degrees_east'
          emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
          emis_dict[keyname]['latitude']={}
          emis_dict[keyname]['latitude']['dtype']='f4'
          emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
          emis_dict[keyname]['latitude']['units']='degrees_east'
          emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
          #emis_dict[keyname]['time']={}
          #emis_dict[keyname]['time']['dtype']='i4'
          #emis_dict[keyname]['time']['dims' ]=['time']
          #emis_dict[keyname]['time']['data' ]=np.arange(12) # [0,1,2,3,4,...,11]
          emis_dict[keyname]['voc']={}
          emis_dict[keyname]['voc']['dtype']='f4'
          # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] 
          emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
          if spec2 in ['OC','PM25','BC']:
            units = 'ug/m2/sec'
          else:
            units = 'mol/m2/sec'
          emis_dict[keyname]['voc']['units']=units
          if 'ind' in sec2:
            # unit [g] --> [mol] for chemical species
            emis_dict[keyname]['voc']['data' ] = emis_wrf[:,:]/Mm*ind_ratio[spec2][isec]*coef_arr[ispec] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>]
          else:
            emis_dict[keyname]['voc']['data' ] = emis_wrf[:,:]/Mm*coef_arr[ispec] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>]
        else:
          if 'ind' in sec2:
            # unit [g] --> [mol] for chemical species
            emis_dict[keyname]['voc']['data' ]+= emis_wrf[:,:]/Mm*ind_ratio[spec2][isec]*coef_arr[ispec] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>]
          else:
            emis_dict[keyname]['voc']['data' ]+= emis_wrf[:,:]/Mm*coef_arr[ispec] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>]

  else:
    #
    if rank == 0:
      print(fnames_voc)
    if len(fnames_voc) != 12:
      print('HTAP  files should consist of 12 monthly files')
      print(fnames_voc)
      sys.exit(1)
    #
    #
    # New Grid Dimensions
    ng_we_wrf = XLON.shape[0]
    ng_sn_wrf = XLON.shape[1]
    #
    emis_wrf_total  = np.zeros(shape=(ng_we_wrf,ng_sn_wrf,12))
    # 
    new_fnames_voc = []
    for imonth in np.arange(12)+1:
      if rank == 0:
         print(imonth)
      month_str = '_'+str(imonth)+'.0.1'
      for fname_voc in fnames_voc:
        if month_str in fname_voc: 
          new_fnames_voc.append(fname_voc)
          continue
        # END of if month_str in fname_voc:
      # END of for fname_voc in fnames_voc:
    if rank == 0:
      print(new_fnames_voc)
    if 'AGRICULTURE' in new_fnames_voc[0]:
      sec = 'awb'
    if 'RESIDENTIAL' in new_fnames_voc[0]:
      sec = 'dom'
    if 'ENERGY'      in new_fnames_voc[0]:
      sec = 'ene'
    if 'INDUSTRY'    in new_fnames_voc[0]:
      sec = 'ind'
    if 'TRANSPORT'   in new_fnames_voc[0]:
      sec = 'tra' 
    #
    for imonth, fname_voc in enumerate(new_fnames_voc):
      ### Test
      ## if imonth > 0:
      ##   continue
      ## print fname_voc
      ## print os.path.basename(fname_voc)
      nc    = Dataset(fname_voc,'r',format='NETCDF4')
      #
      #====================================================================
      #--------------------------------------------------------------------
      #
      #  Get a list of variable names 
      #  vars = ['lat','lon','emi_nmvoc']
      #  nvars=3 
      #  dimension = [1800,3600]
      #  lon.shape[0] = 3600
      #  lat.shape[0] = 1800
      #  
      #  Output: nvars : # of variables 
      #          vars  : list of variable names
      #
      vars  = list(nc.variables.keys())
      nvars = len(vars)
      vars  = list(vars)
      # print vars
      #
      #=====================================================================
      #---------------------------------------------------------------------
      #
      #  assign variables : 'lon' and 'lat' 
      lon   = np.array(nc.variables["lon"])   #    0 < lon < 360
      lat   = np.array(nc.variables["lat"])   #  -90 < lat <  90
      #
      for iv, var in enumerate(vars):
        # print iv, var
        #
        #  assign variable : 'emi_nmvoc'
        emis_tmp = np.array(nc.variables[var]) 
        #
        # iv = 0 : 'lat'
        # iv = 1 : 'lon'
        if iv <= 1:
          continue
        #
        nanind = np.where(np.isnan(emis_tmp))              # Indices of NaN in emis_tmp
        #emis_vname.append(var)                             # Create a list of variable name 
        emis_tmp[nanind] = np.zeros_like(emis_tmp)[nanind] # Replace NaN with 0
        #emis_nmvoc[:,:,iv-2] = np.transpose(emis_tmp[:,:]) # Set data to 'emis_nmvoc' [lon,lat,nvar]
        #
        #----------------------------------------------------------------------
        #
        #  Change the order of indexing
        #
        emis_tmp   = np.transpose(emis_tmp)     # [lat,lon] ---> [lon,lat]
        #  
        #======================================================================
        #----------------------------------------------------------------------
        #
        #  Expand longitude range from 0 < lon < 360 to -180 < lon < 360
        #
        #  Output: emis_nmvoc[nlon+nlon/2,nlat,nvars-2]
        #          emis_lon[nlon+nlon/2,nlat]
        #          emis_lat[nlon+nlon/2,nlat]
        #
        #
        emis_tmp   = np.concatenate((emis_tmp[emis_tmp.shape[0]//2:,:],emis_tmp[:,:]),axis=0)
        lon_temp   = np.concatenate((lon[lon.shape[0]//2:]-360.0,lon))
        emis_lon1  = lon_temp
        emis_lat1  = lat
        emis_lon, emis_lat = np.meshgrid(lon_temp,lat,indexing='ij')
        #
        #===========================================================================================
        #-------------------------------------------------------------------------------------------
        #  variable : 'emis_awb'
        #  unit : [kg/m2/sec]
        #  emis_cell_area : area [m2] of each cell
        # 
        if spec in ['CO','NH3','NOx','SO2']:
          #  unit : [kg/m2/sec] -> [g/m2/sec]
          #
          units = 'g/m2/sec'
          emis_tmp *= 1.e3
        if spec in ['OC','PM25','BC']:
          #  unit : [kg/m2/sec] -> [ug/m2/sec]
          #
          units = 'ug/m2/sec'
          emis_tmp *= 1.e9

        # New Grid Dimensions
        ng_we_wrf = XLON.shape[0]
        ng_sn_wrf = XLON.shape[1]
        #
        emis_wrf  = np.zeros(shape=(ng_we_wrf,ng_sn_wrf))
        # 
        for c in np.arange(ng_we_wrf):                          # Loop over west_east index of wrf grid (wrfinput_d01)
          for r in np.arange(ng_sn_wrf):                        # Loop over south_north index of wrf grid (wrfinput_d01)
            if 1==0:
              #... Common to WRF grid ........................
              #    Index ranges of 'west-east' and 'south-north' variables of emission grid. 
              #    
              ind_lon_min = int(area_common_array[c,r,0])
              ind_lon_max = int(area_common_array[c,r,1])
              ind_lat_min = int(area_common_array[c,r,2])
              ind_lat_max = int(area_common_array[c,r,3])
              #
              count            = 1           # index counter for 3rd index of area_common_array[c,r,3+count]
              emis_wrf_temp    = 0.0         # temp. variable of emission on a wrf grid
              ## area_all    = 0.0         # checking parameter : should be 1.0 after the loop
              #
              # Loops over "ilon" 'west-east (longitude)' and "ilat" 'south-north (latitude)' 
              # index of emission grid 
              # Only those containing a wrf cell (XLON[c,r],XLAT[c,r]) defined by 4 points [X,Y]
              #
              if np.amax(emis_tmp[ind_lon_min:ind_lon_max,ind_lat_min:ind_lat_max]) != 0.0:
                for ilon in np.arange(ind_lon_min,ind_lon_max):      
                  for ilat in np.arange(ind_lat_min,ind_lat_max):    
                    if emis_tmp[ilon,ilat] != 0.0 :     # Just to avoid a useless computation 
                      area_temp     = area_common_array[c,r,3+count]     # Area ratio (area of emission grid (ilon,ilat) shared with a wrf grid)/(area of emission grid)
                      emis_wrf_temp = emis_wrf_temp+area_temp*emis_tmp[ilon,ilat]
                    #area_all    = area_all + area_temp
                    count += 1
                # End of for ilon and ilat
                #print 'area_all = ',area_all
                #print c,'/',ng_we_wrf,' ',r,'/',ng_sn_wrf,emis_wrf
            #
            #---- Using area_common_dict ------------------------------------------------
            #
            if 1==1:
              emis_wrf_temp = 0.0
              for idict in np.arange(area_common_dict[c][r]['total_count']):
                ilon = area_common_dict[c][r][idict]['ilon']
                ilat = area_common_dict[c][r][idict]['ilat']
                emis_wrf_temp += area_common_dict[c][r][idict]['area']*emis_tmp[ilon,ilat]
            #
            #---- End of Using area_common_dict ------------------------------------------
            # 
            emis_wrf[c,r] = emis_wrf_temp
          # End of loop r
        # End of loop c
        #
        #-----------------------------------------------------------
        #
        emis_wrf_total[:,:,imonth] = emis_wrf[:,:]
        #
        #-----------------------------------------------------------
        #
      # END of loop over var : for iv, var in enumerate(vars):
      nc.close()
    # END of loop for monthly input files

    sec_arr = [sec]
    spec_arr= [spec]
    coef_arr= [1.0]
    if sec == 'ind':
      sec_arr = ['ind1','ind2']
    if spec == 'NOx':
      spec_arr = ['NO','NO2']
      coef_arr = [0.9,0.1]

    for isec, sec2 in enumerate(sec_arr):
      for ispec, spec2 in enumerate(spec_arr):
        Mm = 1.0
        # Molecular mass (g/mol)
        if spec2 in Mmol:
          Mm = Mmol[spec2]
        keyname = sec2+'-HTAP-'+spec2
        emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
        emis_dict[keyname]['dimensions']={}
        emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
        emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
        emis_dict[keyname]['dimensions']['time']       = 12        # of points in time series> e.g.:12
        emis_dict[keyname]['west_east']={}
        emis_dict[keyname]['west_east']['dtype']='i4'
        emis_dict[keyname]['west_east']['dims' ]=['west_east']
        emis_dict[keyname]['west_east']['units']=''
        emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
        emis_dict[keyname]['south_north']={}
        emis_dict[keyname]['south_north']['dtype']='i4'
        emis_dict[keyname]['south_north']['dims' ]=['south_north']
        emis_dict[keyname]['south_north']['units']=''
        emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
        emis_dict[keyname]['longitude']={}
        emis_dict[keyname]['longitude']['dtype']='f4'
        emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
        emis_dict[keyname]['longitude']['units']='degrees_east'
        emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
        emis_dict[keyname]['latitude']={}
        emis_dict[keyname]['latitude']['dtype']='f4'
        emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
        emis_dict[keyname]['latitude']['units']='degrees_east'
        emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
        emis_dict[keyname]['time']={}
        emis_dict[keyname]['time']['dtype']='i4'
        emis_dict[keyname]['time']['dims' ]=['time']
        emis_dict[keyname]['time']['data' ]=np.arange(12) # [0,1,2,3,4,...,11]
        emis_dict[keyname]['voc']={}
        emis_dict[keyname]['voc']['dtype']='f4'
        emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] 
        # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
        if spec2 in ['OC','PM25','BC']:
          units = 'ug/m2/sec'
        else:
          units = 'mol/m2/sec'
        emis_dict[keyname]['voc']['units']=units
        if 'ind' in sec2:
          # unit [g] --> [mol] for chemical species
          emis_dict[keyname]['voc']['data' ]= emis_wrf_total[:,:,:]/Mm*ind_ratio[spec2][isec]*coef_arr[ispec] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
        else:
          emis_dict[keyname]['voc']['data' ]= emis_wrf_total[:,:,:]/Mm*coef_arr[ispec] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
  
  return emis_dict

#-------------------------------------------------------------------
#
#  Read HTAP Anthoropogenic NMVOC emission data (except for AIR/SHIPS) 
#  and interpolate it on to WRF grid
#
def AddDict_HTAP_Anth_VOC_OnWRF( emis_dict, fnames_voc,\
                    area_common_array, area_common_dict, emis_cell_area,\
                    XLON, XLAT):
  if rank == 0:
    print(fnames_voc)

  if 'SHIPS' in fnames_voc[0] or 'AIR' in fnames_voc[0]:
    #
    #
    if len(fnames_voc) != 1:
      if rank == 0:
        print('HTAP NMVOC (AIR,SHIP) files should consist of 1 file')
        print(fnames_voc)
      sys.exit(1)
    #
    # New Grid Dimensions
    ng_we_wrf = XLON.shape[0]
    ng_sn_wrf = XLON.shape[1]
    #
    if 'SHIPS' in fnames_voc[0]:
      sec = 'shp'
    if 'AIR' in fnames_voc[0]:
      sec = 'shp'
    #
    fname_voc = fnames_voc[0]
    nc    = Dataset(fname_voc,'r',format='NETCDF4')
    #
    #====================================================================
    #--------------------------------------------------------------------
    #
    #  Get a list of variable names 
    #  vars = ['lat','lon','emi_nmvoc']
    #  nvars=3 
    #  dimension = [1800,3600]
    #  lon.shape[0] = 3600
    #  lat.shape[0] = 1800
    #  
    #  Output: nvars : # of variables 
    #          vars  : list of variable names
    #
    vars  = list(nc.variables.keys())
    nvars = len(vars)
    vars  = list(vars)
    # print vars
    #
    #=====================================================================
    #---------------------------------------------------------------------
    #
    #  assign variables : 'lon' and 'lat' 
    lon   = np.array(nc.variables["lon"])   #    0 < lon < 360
    lat   = np.array(nc.variables["lat"])   #  -90 < lat <  90
    #
    for iv, var in enumerate(vars):
      # print iv, var
      #
      #  assign variable : 'emi_nmvoc'
      emis_tmp = np.array(nc.variables[var]) 
      #
      # iv = 0 : 'lat'
      # iv = 1 : 'lon'
      if iv <= 1:
        continue
      #
      nanind = np.where(np.isnan(emis_tmp))              # Indices of NaN in emis_tmp
      #emis_vname.append(var)                             # Create a list of variable name 
      emis_tmp[nanind] = np.zeros_like(emis_tmp)[nanind] # Replace NaN with 0
      #emis_nmvoc[:,:,iv-2] = np.transpose(emis_tmp[:,:]) # Set data to 'emis_nmvoc' [lon,lat,nvar]
      #
      #----------------------------------------------------------------------
      #
      #  Change the order of indexing
      #
      emis_tmp   = np.transpose(emis_tmp)     # [lat,lon] ---> [lon,lat]
      #  
      #======================================================================
      #----------------------------------------------------------------------
      #
      #  Expand longitude range from 0 < lon < 360 to -180 < lon < 360
      #
      #  Output: emis_nmvoc[nlon+nlon/2,nlat,nvars-2]
      #          emis_lon[nlon+nlon/2,nlat]
      #          emis_lat[nlon+nlon/2,nlat]
      #
      emis_tmp   = np.concatenate((emis_tmp[emis_tmp.shape[0]//2:,:],emis_tmp[:,:]),axis=0)
      lon_temp   = np.concatenate((lon[lon.shape[0]//2:]-360.0,lon))
      emis_lon1  = lon_temp
      emis_lat1  = lat
      emis_lon, emis_lat = np.meshgrid(lon_temp,lat,indexing='ij')
      #
      #===========================================================================================
      #-------------------------------------------------------------------------------------------
      #  variable : 'emis_awb'
      #  unit : [kg/m2/sec]
      #  emis_cell_area : area [m2] of each cell
      # 
      # New Grid Dimensions
      ng_we_wrf = XLON.shape[0]
      ng_sn_wrf = XLON.shape[1]
      #
      emis_wrf  = np.zeros(shape=(ng_we_wrf,ng_sn_wrf))
      # 
      for c in np.arange(ng_we_wrf):                          # Loop over west_east index of wrf grid (wrfinput_d01)
        for r in np.arange(ng_sn_wrf):                        # Loop over south_north index of wrf grid (wrfinput_d01)
          if 1==0:
            #... Common to WRF grid ........................
            #    Index ranges of 'west-east' and 'south-north' variables of emission grid. 
            #    
            ind_lon_min = int(area_common_array[c,r,0])
            ind_lon_max = int(area_common_array[c,r,1])
            ind_lat_min = int(area_common_array[c,r,2])
            ind_lat_max = int(area_common_array[c,r,3])
            #
            count            = 1           # index counter for 3rd index of area_common_array[c,r,3+count]
            emis_wrf_temp    = 0.0         # temp. variable of emission on a wrf grid
            ## area_all    = 0.0         # checking parameter : should be 1.0 after the loop
            #
            # Loops over "ilon" 'west-east (longitude)' and "ilat" 'south-north (latitude)' 
            # index of emission grid 
            # Only those containing a wrf cell (XLON[c,r],XLAT[c,r]) defined by 4 points [X,Y]
            #
            if np.amax(emis_tmp[ind_lon_min:ind_lon_max,ind_lat_min:ind_lat_max]) != 0.0:
              for ilon in np.arange(ind_lon_min,ind_lon_max):      
                for ilat in np.arange(ind_lat_min,ind_lat_max):    
                  if emis_tmp[ilon,ilat] != 0.0 :     # Just to avoid a useless computation 
                    area_temp     = area_common_array[c,r,3+count]     # Area ratio (area of emission grid (ilon,ilat) shared with a wrf grid)/(area of emission grid)
                    emis_wrf_temp = emis_wrf_temp+area_temp*emis_tmp[ilon,ilat]
                  #area_all    = area_all + area_temp
                  count += 1
              # End of for ilon and ilat
              #print 'area_all = ',area_all
              #print c,'/',ng_we_wrf,' ',r,'/',ng_sn_wrf,emis_wrf
          #
          #---- Using area_common_dict ------------------------------------------------
          #
          if 1==1:
            emis_wrf_temp = 0.0
            for idict in np.arange(area_common_dict[c][r]['total_count']):
              ilon = area_common_dict[c][r][idict]['ilon']
              ilat = area_common_dict[c][r][idict]['ilat']
              emis_wrf_temp += area_common_dict[c][r][idict]['area']*emis_tmp[ilon,ilat]
          #
          #---- End of Using area_common_dict ------------------------------------------
          # 
          emis_wrf[c,r] = emis_wrf_temp
        # End of loop r
      # End of loop c
      #
      #-----------------------------------------------------------
      #
    # END of loop over var : for iv, var in enumerate(vars):
    nc.close()
    # END of loop for monthly input files

    keyname = sec+'-HTAP-VOC'
    if rank == 0:
      print('emis_dict.get(keyname) = ',emis_dict.get(keyname))
    if emis_dict.get(keyname) == None:
      emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
      emis_dict[keyname]['dimensions']={}
      emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
      emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
      #emis_dict[keyname]['dimensions']['time']       = 12        # of points in time series> e.g.:12
      emis_dict[keyname]['west_east']={}
      emis_dict[keyname]['west_east']['dtype']='i4'
      emis_dict[keyname]['west_east']['dims' ]=['west_east']
      emis_dict[keyname]['west_east']['units']=''
      emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
      emis_dict[keyname]['south_north']={}
      emis_dict[keyname]['south_north']['dtype']='i4'
      emis_dict[keyname]['south_north']['dims' ]=['south_north']
      emis_dict[keyname]['south_north']['units']=''
      emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
      emis_dict[keyname]['longitude']={}
      emis_dict[keyname]['longitude']['dtype']='f4'
      emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
      emis_dict[keyname]['longitude']['units']='degrees_east'
      emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
      emis_dict[keyname]['latitude']={}
      emis_dict[keyname]['latitude']['dtype']='f4'
      emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
      emis_dict[keyname]['latitude']['units']='degrees_east'
      emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
      #emis_dict[keyname]['time']={}
      #emis_dict[keyname]['time']['dtype']='i4'
      #emis_dict[keyname]['time']['dims' ]=['time']
      #emis_dict[keyname]['time']['data' ]=np.arange(12) # [0,1,2,3,4,...,11]
      emis_dict[keyname]['voc']={}
      emis_dict[keyname]['voc']['dtype']='f4'
      #emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] 
      emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
      # 'kg/m2/sec' --> 'kt/km2/year'
      #emis_dict[keyname]['voc']['units']='kg/m2/sec'
      emis_dict[keyname]['voc']['units']='kt/km2/year'
      emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:]*60.0*60.0*24.0*365.0 # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>]
    else:
      emis_dict[keyname]['voc']['data' ]+= emis_wrf[:,:]*60.0*60.0*24.0*365.0 # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>]

  else:
    if len(fnames_voc) != 12:
      print('HTAP NMVOC files should consist of 12 monthly files')
      print(fnames_voc)
      sys.exit(1)
    #
    #
    # New Grid Dimensions
    ng_we_wrf = XLON.shape[0]
    ng_sn_wrf = XLON.shape[1]
    #
    emis_wrf_total  = np.zeros(shape=(ng_we_wrf,ng_sn_wrf,12))
    # 
    new_fnames_voc = []
    for imonth in np.arange(12)+1:
      if rank == 0:
        print(imonth)
      month_str = '_'+str(imonth)+'.0.1'
      for fname_voc in fnames_voc:
        if month_str in fname_voc: 
          new_fnames_voc.append(fname_voc)
          continue
        # END of if month_str in fname_voc:
      # END of for fname_voc in fnames_voc:
    if rank == 0:
      print(new_fnames_voc)
    if 'AGRICULTURE' in new_fnames_voc[0]:
      sec = 'awb'
    if 'RESIDENTIAL' in new_fnames_voc[0]:
      sec = 'dom'
    if 'ENERGY'      in new_fnames_voc[0]:
      sec = 'ene'
    if 'INDUSTRY'    in new_fnames_voc[0]:
      sec = 'ind'
    if 'TRANSPORT'   in new_fnames_voc[0]:
      sec = 'tra' 
    #
    for imonth, fname_voc in enumerate(new_fnames_voc):
      ### Test
      ## if imonth > 0:
      ##   continue
      ## print fname_voc
      ## print os.path.basename(fname_voc)
      nc    = Dataset(fname_voc,'r',format='NETCDF4')
      #
      #====================================================================
      #--------------------------------------------------------------------
      #
      #  Get a list of variable names 
      #  vars = ['lat','lon','emi_nmvoc']
      #  nvars=3 
      #  dimension = [1800,3600]
      #  lon.shape[0] = 3600
      #  lat.shape[0] = 1800
      #  
      #  Output: nvars : # of variables 
      #          vars  : list of variable names
      #
      vars  = list(nc.variables.keys())
      nvars = len(vars)
      vars  = list(vars)
      # print vars
      #
      #=====================================================================
      #---------------------------------------------------------------------
      #
      #  assign variables : 'lon' and 'lat' 
      lon   = np.array(nc.variables["lon"])   #    0 < lon < 360
      lat   = np.array(nc.variables["lat"])   #  -90 < lat <  90
      #
      for iv, var in enumerate(vars):
        # print iv, var
        #
        #  assign variable : 'emi_nmvoc'
        emis_tmp = np.array(nc.variables[var]) 
        #
        # iv = 0 : 'lat'
        # iv = 1 : 'lon'
        if iv <= 1:
          continue
        #
        nanind = np.where(np.isnan(emis_tmp))              # Indices of NaN in emis_tmp
        #emis_vname.append(var)                             # Create a list of variable name 
        emis_tmp[nanind] = np.zeros_like(emis_tmp)[nanind] # Replace NaN with 0
        #emis_nmvoc[:,:,iv-2] = np.transpose(emis_tmp[:,:]) # Set data to 'emis_nmvoc' [lon,lat,nvar]
        #
        #----------------------------------------------------------------------
        #
        #  Change the order of indexing
        #
        emis_tmp   = np.transpose(emis_tmp)     # [lat,lon] ---> [lon,lat]
        #  
        #======================================================================
        #----------------------------------------------------------------------
        #
        #  Expand longitude range from 0 < lon < 360 to -180 < lon < 360
        #
        #  Output: emis_nmvoc[nlon+nlon/2,nlat,nvars-2]
        #          emis_lon[nlon+nlon/2,nlat]
        #          emis_lat[nlon+nlon/2,nlat]
        #
        #
        emis_tmp   = np.concatenate((emis_tmp[emis_tmp.shape[0]//2:,:],emis_tmp[:,:]),axis=0)
        lon_temp   = np.concatenate((lon[lon.shape[0]//2:]-360.0,lon))
        emis_lon1  = lon_temp
        emis_lat1  = lat
        emis_lon, emis_lat = np.meshgrid(lon_temp,lat,indexing='ij')
        #
        #===========================================================================================
        #-------------------------------------------------------------------------------------------
        #  variable : 'emis_awb'
        #  unit : [kg/m2/sec]
        #  emis_cell_area : area [m2] of each cell
        # 
        # New Grid Dimensions
        ng_we_wrf = XLON.shape[0]
        ng_sn_wrf = XLON.shape[1]
        #
        emis_wrf  = np.zeros(shape=(ng_we_wrf,ng_sn_wrf))
        # 
        for c in np.arange(ng_we_wrf):                          # Loop over west_east index of wrf grid (wrfinput_d01)
          for r in np.arange(ng_sn_wrf):                        # Loop over south_north index of wrf grid (wrfinput_d01)
            if 1==0:
              #... Common to WRF grid ........................
              #    Index ranges of 'west-east' and 'south-north' variables of emission grid. 
              #    
              ind_lon_min = int(area_common_array[c,r,0])
              ind_lon_max = int(area_common_array[c,r,1])
              ind_lat_min = int(area_common_array[c,r,2])
              ind_lat_max = int(area_common_array[c,r,3])
              #
              count            = 1           # index counter for 3rd index of area_common_array[c,r,3+count]
              emis_wrf_temp    = 0.0         # temp. variable of emission on a wrf grid
              ## area_all    = 0.0         # checking parameter : should be 1.0 after the loop
              #
              # Loops over "ilon" 'west-east (longitude)' and "ilat" 'south-north (latitude)' 
              # index of emission grid 
              # Only those containing a wrf cell (XLON[c,r],XLAT[c,r]) defined by 4 points [X,Y]
              #
              if np.amax(emis_tmp[ind_lon_min:ind_lon_max,ind_lat_min:ind_lat_max]) != 0.0:
                for ilon in np.arange(ind_lon_min,ind_lon_max):      
                  for ilat in np.arange(ind_lat_min,ind_lat_max):    
                    if emis_tmp[ilon,ilat] != 0.0 :     # Just to avoid a useless computation 
                      area_temp     = area_common_array[c,r,3+count]     # Area ratio (area of emission grid (ilon,ilat) shared with a wrf grid)/(area of emission grid)
                      emis_wrf_temp = emis_wrf_temp+area_temp*emis_tmp[ilon,ilat]
                    #area_all    = area_all + area_temp
                    count += 1
                # End of for ilon and ilat
                #print 'area_all = ',area_all
                #print c,'/',ng_we_wrf,' ',r,'/',ng_sn_wrf,emis_wrf
            #
            #---- Using area_common_dict ------------------------------------------------
            #
            if 1==1:
              emis_wrf_temp = 0.0
              for idict in np.arange(area_common_dict[c][r]['total_count']):
                ilon = area_common_dict[c][r][idict]['ilon']
                ilat = area_common_dict[c][r][idict]['ilat']
                emis_wrf_temp += area_common_dict[c][r][idict]['area']*emis_tmp[ilon,ilat]
            #
            #---- End of Using area_common_dict ------------------------------------------
            # 
            emis_wrf[c,r] = emis_wrf_temp
          # End of loop r
        # End of loop c
        #
        #-----------------------------------------------------------
        #
        emis_wrf_total[:,:,imonth] = emis_wrf[:,:]
        #
        #-----------------------------------------------------------
        #
      # END of loop over var : for iv, var in enumerate(vars):
      nc.close()
    # END of loop for monthly input files

    keyname = sec+'-HTAP-VOC'
    emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
    emis_dict[keyname]['dimensions']={}
    emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
    emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
    emis_dict[keyname]['dimensions']['time']       = 12        # of points in time series> e.g.:12
    emis_dict[keyname]['west_east']={}
    emis_dict[keyname]['west_east']['dtype']='i4'
    emis_dict[keyname]['west_east']['dims' ]=['west_east']
    emis_dict[keyname]['west_east']['units']=''
    emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
    emis_dict[keyname]['south_north']={}
    emis_dict[keyname]['south_north']['dtype']='i4'
    emis_dict[keyname]['south_north']['dims' ]=['south_north']
    emis_dict[keyname]['south_north']['units']=''
    emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
    emis_dict[keyname]['longitude']={}
    emis_dict[keyname]['longitude']['dtype']='f4'
    emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
    emis_dict[keyname]['longitude']['units']='degrees_east'
    emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
    emis_dict[keyname]['latitude']={}
    emis_dict[keyname]['latitude']['dtype']='f4'
    emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
    emis_dict[keyname]['latitude']['units']='degrees_east'
    emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
    emis_dict[keyname]['time']={}
    emis_dict[keyname]['time']['dtype']='i4'
    emis_dict[keyname]['time']['dims' ]=['time']
    emis_dict[keyname]['time']['data' ]=np.arange(12) # [0,1,2,3,4,...,11]
    emis_dict[keyname]['voc']={}
    emis_dict[keyname]['voc']['dtype']='f4'
    emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] 
    # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
    emis_dict[keyname]['voc']['units']='kg/m2/sec'
    emis_dict[keyname]['voc']['data' ]= emis_wrf_total[:,:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
  
  return emis_dict

#----------------------------------------------------------
#  
#  **** Imporved version for a polar stereographic projection*****
#  WRF grid points and WRF Cell corner grid points 
#
#  INPUT  : wrfinput_d01
#
#  OUTPUT : XLON, XLAT, XLONa, XLATa (N.B.[we-index,sn-index])
#
#-----------------------------------------------------------
def WRF_Grids2(fname):
  #
  #========================================================
  #
  #   INPUT from 'wrfinput_d01'
  #
  #--------------------------------------------------------
  nc = Dataset(fname,'r',format='NETCDF4')       # wrfinput_d01, 'fname' is defined above
  #
  clat     = nc.getncattr("CEN_LAT")
  clon     = nc.getncattr("CEN_LON")
  map_proj = nc.getncattr("MAP_PROJ")
  #
  #   INPUT : Lat and Lon from wrfinput_d01
  #-------------------------------------------------------------
  #
  XLAT  = np.array(nc.variables['XLAT'])
  XLON  = np.array(nc.variables['XLONG'])
  XLATu = np.array(nc.variables['XLAT_U'])
  XLONu = np.array(nc.variables['XLONG_U'])
  XLATv = np.array(nc.variables['XLAT_V'])
  XLONv = np.array(nc.variables['XLONG_V'])
  #
  #   Close the netCDF file
  #---------------------------------------------------------------
  nc.close()
  #
  #   XLAT and XLON [pres-index,lat-index,lon-index]
  #---------------------------------------------------------------
  #
  XLAT  = XLAT[0,:,:]
  XLON  = XLON[0,:,:]
  XLATu = XLATu[0,:,:]
  XLONu = XLONu[0,:,:]
  XLATv = XLATv[0,:,:]
  XLONv = XLONv[0,:,:]
  #
  #   Index Order : [sn-index,we-index] --> [we-index,sn-index]
  #----------------------------------------------------------------
  #
  XLAT  = np.transpose(XLAT)
  XLON  = np.transpose(XLON)
  XLATu = np.transpose(XLATu)
  XLONu = np.transpose(XLONu)
  XLATv = np.transpose(XLATv)
  XLONv = np.transpose(XLONv)
  #
  #   Map to projection if map_proj=2 (polar stereographic)
  #-----------------------------------------------------------------
  #
  if map_proj == 2: 
    m = Basemap(projection='stere',boundinglat=0, lon_0=clon, lat_0=clat, \
                width=1000000, height=1000000, resolution='l')
    XLON_m,  XLAT_m  = m(XLON ,XLAT )
    XLONu_m, XLATu_m = m(XLONu,XLATu)
    XLONv_m, XLATv_m = m(XLONv,XLATv)
  else:
    XLON_m,  XLAT_m  = XLON  ,XLAT 
    XLONu_m, XLATu_m = XLONu ,XLATu
    XLONv_m, XLATv_m = XLONv ,XLATv
    
  #
  #   WRF Grid Dimensions : # of grid points 
  #-------------------------------------------------------------
  ng_we_wrf = XLON.shape[0]   # west-east cell center grid points
  ng_sn_wrf = XLON.shape[1]   # south-north cell center grid points
  #
  #   Lat, Lon of wrf grid cell corner (XLONa,XLATa)
  #   Area calculation of (XLON,XLAT) grids
  #------------------------------------------------------------- 
  #
  XLONa   = np.zeros(shape=(ng_we_wrf+1,ng_sn_wrf+1))
  XLATa   = np.zeros(shape=(ng_we_wrf+1,ng_sn_wrf+1))
  XLONa_m = np.zeros(shape=(ng_we_wrf+1,ng_sn_wrf+1))
  XLATa_m = np.zeros(shape=(ng_we_wrf+1,ng_sn_wrf+1))
  #
  #   Cell corner grid points 
  #-------------------------------------------------------------
  for i in range(ng_we_wrf):
   print('Defining boundary coordinates of wrfinput ', i)
   for j in range(ng_sn_wrf):
     X = np.array([XLON_m[i,j],XLONu_m[i,j],XLONv_m[i,j],XLONu_m[i+1,j],XLONv_m[i,j+1]])
     Y = np.array([XLAT_m[i,j],XLATu_m[i,j],XLATv_m[i,j],XLATu_m[i+1,j],XLATv_m[i,j+1]])
     if map_proj == 2:
       if np.amax(X) >= 150. and np.amin(X) <= -150. and np.amax(X)-np.amin(X) > 180.0:
         ind = np.where(X < 0)
         if len(ind) > 0:
           X[ind] = X[ind] + 360.0
     XLONa_m[i  ,j  ] = X[1]-X[0]+X[2]
     XLONa_m[i+1,j  ] = X[2]-X[0]+X[3]
     XLONa_m[i  ,j+1] = X[4]-X[0]+X[1]
     XLONa_m[i+1,j+1] = X[3]-X[0]+X[4]
     XLATa_m[i  ,j  ] = Y[1]-Y[0]+Y[2]
     XLATa_m[i+1,j  ] = Y[2]-Y[0]+Y[3]
     XLATa_m[i  ,j+1] = Y[4]-Y[0]+Y[1]
     XLATa_m[i+1,j+1] = Y[3]-Y[0]+Y[4]
 
  #
  #  Convert back to Lat/Lon
  #-----------------------------------------------------------------
  if map_proj==2:
    XLONa, XLATa = m(XLONa_m, XLATa_m, inverse=True)
  else:
    XLONa, XLATa = XLONa_m, XLATa_m

  #
  #  Check if the value is within a valid range
  #------------------------------------------------------------------
  #
  for i in range(ng_we_wrf):
   print('Defining boundary coordinates of wrfinput ', i)
   for j in range(ng_sn_wrf):
     #
     # XLATa must be between -90 =< XLAT =< 90
     #
     XLATa[i  ,j  ] = np.amin([XLATa[i  ,j  ],90.0])
     XLATa[i+1,j  ] = np.amin([XLATa[i+1,j  ],90.0])
     XLATa[i  ,j+1] = np.amin([XLATa[i  ,j+1],90.0])
     XLATa[i+1,j+1] = np.amin([XLATa[i+1,j+1],90.0])
     #
     XLATa[i  ,j  ] = np.amax([XLATa[i  ,j  ],-90.0])
     XLATa[i+1,j  ] = np.amax([XLATa[i+1,j  ],-90.0])
     XLATa[i  ,j+1] = np.amax([XLATa[i  ,j+1],-90.0])
     XLATa[i+1,j+1] = np.amax([XLATa[i+1,j+1],-90.0])
     #
     # XLONa must be between -180.0 =< XLONa =< 360.0
     #
     if XLONa[i  ,j  ] < -180.0:
        XLONa[i  ,j  ] = XLONa[i  ,j  ]+360.0
     if XLONa[i+1,j  ] < -180.0:
        XLONa[i+1,j  ] = XLONa[i+1,j  ]+360.0
     if XLONa[i  ,j+1] < -180.0:
        XLONa[i  ,j+1] = XLONa[i  ,j+1]+360.0
     if XLONa[i+1,j+1] < -180.0:
        XLONa[i+1,j+1] = XLONa[i+1,j+1]+360.0
     #
     if XLONa[i  ,j  ] > 360.0:
        XLONa[i  ,j  ] = XLONa[i  ,j  ]-360.0
     if XLONa[i+1,j  ] > 360.0:
        XLONa[i+1,j  ] = XLONa[i+1,j  ]-360.0
     if XLONa[i  ,j+1] > 360.0:
        XLONa[i  ,j+1] = XLONa[i  ,j+1]-360.0
     if XLONa[i+1,j+1] > 360.0:
        XLONa[i+1,j+1] = XLONa[i+1,j+1]-360.0

  return XLON, XLAT, XLONa, XLATa

#----------------------------------------------------------
#
#  WRF grid points and WRF Cell corner grid points 
#
#  INPUT  : wrfinput_d01
#
#  OUTPUT : XLON, XLAT, XLONa, XLATa (N.B.[we-index,sn-index])
#
#-----------------------------------------------------------
def WRF_Grids(fname):
  #
  #========================================================
  #
  #   INPUT from 'wrfinput_d01'
  #
  #--------------------------------------------------------
  nc = Dataset(fname,'r',format='NETCDF4')       # wrfinput_d01, 'fname' is defined above
  #
  ##### if 'west_east' in nc.dimensions.keys() and 'south_north' in nc.dimensions.keys():
  #####   if (nc.dimensions['west_east'].size == 2) and \
  #####      (nc.dimensions['south_north'].size == 2):
  #####     SCM = True
  #####   else:
  #####     SCM = False
  SCM = False
  
  print('SCM is ', SCM)
  ##### vars = nc.variables.keys()
  ##### vars = set(map(str,nc.variables))
  ##### nvars= len(vars)
  ##### atts = set(map(str,nc.ncattrs()))
  ##### natts= len(atts)
  ##### #
  ##### #   Variable Name List : vars_name
  ##### #----------------------------------------------------------
  ##### vars_name = []
  ##### for ij, var in enumerate(vars):
  #####    print ij, var
  #####    vars_name.append(var)
  ##### #
  ##### #   Attribute Name List : atts_name
  ##### #-----------------------------------------------------------
  ##### atts_name  = []
  ##### atts_value = []
  ##### for ij, att in enumerate(atts):
  #####    temp = getattr(nc,att)
  #####    atts_name.append(att)
  #####    atts_value.append(str(temp)) 
  #####    print att, ' = ', str(getattr(nc,att))
  ##### #
  ##### #   Attribute Name List Kept for output
  ##### #------------------------------------------------------------
  ##### glob_att = ['DX','DY','CEN_LAT','CEN_LON','TRUELAT1','TRUELAT2','MOAD_CEN_LAT','STAND_LON',
  #####             'POLE_LAT','POLE_LON','GMT','JULYR','JULDAY','MAP_PROJ','MMINLU','NUM_LAND_CAT',
  #####             'ISWATER','ISLAKE','ISICE','ISURBAN','ISOILWATER','WEST-EAST_GRID_DIMENSION',
  #####             'SOUTH-NORHT_GRID_DIMENSION','BOTTOM-TOP_GRID_DIMENSION']
  ##### #
  ##### #   Attribute Name List Actually exists
  ##### #------------------------------------------------------------
  ##### value_glob_att = []
  ##### name_glob_att  = glob_att[:]
  ##### for ij, gatt in enumerate(glob_att):
  #####    print gatt
  #####    try: 
  #####      ind = atts_name.index(gatt)
  #####      value_glob_att.append(atts_value[ind])
  #####    except ValueError:
  #####      value_glob_att.append(' ')
  #
  #   INPUT : Lat and Lon from wrfinput_d01
  #-------------------------------------------------------------
  #
  XLAT  = np.array(nc.variables['XLAT'])
  XLON  = np.array(nc.variables['XLONG'])
  XLATu = np.array(nc.variables['XLAT_U'])
  XLONu = np.array(nc.variables['XLONG_U'])
  XLATv = np.array(nc.variables['XLAT_V'])
  XLONv = np.array(nc.variables['XLONG_V'])
  if rank == 0:
    print(XLAT)
    print(XLON)
    print(XLATu)
    print(XLONu)
    print(XLATv)
    print(XLONv)

  #
  #   Close the netCDF file
  #---------------------------------------------------------------
  nc.close()
  #
  #   XLAT and XLON [pres-index,lat-index,lon-index]
  #---------------------------------------------------------------
  #
  XLAT  = XLAT[0,:,:]
  XLON  = XLON[0,:,:]
  XLATu = XLATu[0,:,:]
  XLONu = XLONu[0,:,:]
  XLATv = XLATv[0,:,:]
  XLONv = XLONv[0,:,:]
  #
  #   Index Order : [sn-index,we-index] --> [we-index,sn-index]
  #----------------------------------------------------------------
  #
  XLAT  = np.transpose(XLAT)
  XLON  = np.transpose(XLON)
  XLATu = np.transpose(XLATu)
  XLONu = np.transpose(XLONu)
  XLATv = np.transpose(XLATv)
  XLONv = np.transpose(XLONv)
  #
  #   WRF Grid Dimensions : # of grid points 
  #-------------------------------------------------------------
  ng_we_wrf = XLON.shape[0]   # west-east cell center grid points
  ng_sn_wrf = XLON.shape[1]   # south-north cell center grid points
  #
  #   Lat, Lon of wrf grid cell corner (XLONa,XLATa)
  #   Area calculation of (XLON,XLAT) grids
  #------------------------------------------------------------- 
  #
  XLONa   = np.zeros(shape=(ng_we_wrf+1,ng_sn_wrf+1))
  XLATa   = np.zeros(shape=(ng_we_wrf+1,ng_sn_wrf+1))
  #
  #   Cell corner grid points 
  #-------------------------------------------------------------
  for i in range(ng_we_wrf):
   if rank == 0:
     print('Defining boundary coordinates of wrfinput ', i)
   for j in range(ng_sn_wrf):
     X = np.array([XLON[i,j],XLONu[i,j],XLONv[i,j],XLONu[i+1,j],XLONv[i,j+1]])
     Y = np.array([XLAT[i,j],XLATu[i,j],XLATv[i,j],XLATu[i+1,j],XLATv[i,j+1]])
     if np.amax(X) >= 150. and np.amin(X) <= -150. and np.amax(X)-np.amin(X) > 180.0:
       ind = np.where(X < 0)
       if len(ind) > 0:
         X[ind] = X[ind] + 360.0
     XLONa[i  ,j  ] = X[1]-X[0]+X[2]
     XLONa[i+1,j  ] = X[2]-X[0]+X[3]
     XLONa[i  ,j+1] = X[4]-X[0]+X[1]
     XLONa[i+1,j+1] = X[3]-X[0]+X[4]
     XLATa[i  ,j  ] = Y[1]-Y[0]+Y[2]
     XLATa[i+1,j  ] = Y[2]-Y[0]+Y[3]
     XLATa[i  ,j+1] = Y[4]-Y[0]+Y[1]
     XLATa[i+1,j+1] = Y[3]-Y[0]+Y[4]
     #
     # XLATa must be between -90 =< XLAT =< 90
     #
     XLATa[i  ,j  ] = np.amin([XLATa[i  ,j  ],90.0])
     XLATa[i+1,j  ] = np.amin([XLATa[i+1,j  ],90.0])
     XLATa[i  ,j+1] = np.amin([XLATa[i  ,j+1],90.0])
     XLATa[i+1,j+1] = np.amin([XLATa[i+1,j+1],90.0])
     #
     XLATa[i  ,j  ] = np.amax([XLATa[i  ,j  ],-90.0])
     XLATa[i+1,j  ] = np.amax([XLATa[i+1,j  ],-90.0])
     XLATa[i  ,j+1] = np.amax([XLATa[i  ,j+1],-90.0])
     XLATa[i+1,j+1] = np.amax([XLATa[i+1,j+1],-90.0])
     #
     # XLONa must be between -180.0 =< XLONa =< 360.0
     #
     if XLONa[i  ,j  ] < -180.0:
        XLONa[i  ,j  ] = XLONa[i  ,j  ]+360.0
     if XLONa[i+1,j  ] < -180.0:
        XLONa[i+1,j  ] = XLONa[i+1,j  ]+360.0
     if XLONa[i  ,j+1] < -180.0:
        XLONa[i  ,j+1] = XLONa[i  ,j+1]+360.0
     if XLONa[i+1,j+1] < -180.0:
        XLONa[i+1,j+1] = XLONa[i+1,j+1]+360.0
     #
     if XLONa[i  ,j  ] > 360.0:
        XLONa[i  ,j  ] = XLONa[i  ,j  ]-360.0
     if XLONa[i+1,j  ] > 360.0:
        XLONa[i+1,j  ] = XLONa[i+1,j  ]-360.0
     if XLONa[i  ,j+1] > 360.0:
        XLONa[i  ,j+1] = XLONa[i  ,j+1]-360.0
     if XLONa[i+1,j+1] > 360.0:
        XLONa[i+1,j+1] = XLONa[i+1,j+1]-360.0
  
  #print XLONa.shape
  #print XLATa.shape

  #plt.figure
  #for ib in np.arange(0,ng_sn_wrf,10):
  #  a = 0  
  #  b1 = ib
  #  b2 = ib+ng_sn_wrf
  #  for i in range(ng_we_wrf+1)[a:]:
  #   plt.plot(XLONa[i,b1:b2],XLATa[i,b1:b2])
  #   print XLONa[i,b1:b2].shape
  #  for j in range(ng_sn_wrf+1)[b1:b2]:
  #   plt.plot(XLONa[a:,j],XLATa[a:,j])
  #  plt.show()
  #raw_input()

  return XLON, XLAT, XLONa, XLATa

#-------------------------------------------------------------------------
#
#  calculate common area ratio between WRF grids and emission data grids
#
#--------------------------------------------------------------------------

def create_commonarea(res,\
                      XLONa,XLATa,\
                      lon_bound_glob_2D,lat_bound_glob_2D,\
                      domain=1,map_proj=0):
  #  Ratio of common area w.r.t. wrf grids (XLONa,XLATa)
  #
  if res == 0.5:
    fn_commonarea = './common_area05_d'+str(domain).zfill(2)+'.npy'
    fn_commonarea_dict = './common_area05_dict_d'+str(domain).zfill(2)+'.npy'
  elif res == 1.0:
    fn_commonarea = './common_area10_d'+str(domain).zfill(2)+'.npy'
    fn_commonarea_dict = './common_area10_dict_d'+str(domain).zfill(2)+'.npy'
  elif res == 0.1:
    fn_commonarea = './common_area01_d'+str(domain).zfill(2)+'.npy'
    fn_commonarea_dict = './common_area01_dict_d'+str(domain).zfill(2)+'.npy'
  elif res == 0.25:
    fn_commonarea = './common_area025_d'+str(domain).zfill(2)+'.npy'
    fn_commonarea_dict = './common_area025_dict_d'+str(domain).zfill(2)+'.npy'
  else:
    sys.exit("resolution (res) should be 0.1, 0.25, 0.5 or 1.0")
  #
  print(not os.path.isfile(fn_commonarea), not os.path.isfile(fn_commonarea_dict))  
  print(not os.path.isfile(fn_commonarea) or not os.path.isfile(fn_commonarea_dict))  
  if not os.path.isfile(fn_commonarea) or not os.path.isfile(fn_commonarea_dict):  
    #.... Calculate common area of wrf grids and emission grids ......
    #.... Unless a file already exists, calculate all common areas ...
    ng_we_wrf = XLONa.shape[0]-1
    ng_sn_wrf = XLONa.shape[1]-1
    #
    area_common_array = np.zeros((ng_we_wrf,ng_sn_wrf,1000))
    #
    area_common_dict = {}
    #
    if map_proj == 2:
      m = Basemap(projection='stere',boundinglat=0, lon_0=cen_lon, lat_0=cen_lat, \
                  width=1000000, height=1000000, resolution='l')
      XLONa_m, XLATa_m = m(XLONa,XLATa)
      lon_m,   lat_m   = m(lon_bound_glob_2D,lat_bound_glob_2D)
    else:
      XLONa_m = XLONa
      XLATa_m = XLATa
      lon_m   = lon_bound_glob_2D
      lat_m   = lat_bound_glob_2D


    #
    for c in np.arange(ng_we_wrf):
      area_common_dict[c]={}
      
      ### if rank == 0:
      ###   print(c, ng_we_wrf, ' rank = ', rank, ' fn_commonarea = ',fn_commonarea)
        
      for r in np.arange(ng_sn_wrf):
        ### if rank == 0:
        ###   print(c,r,ng_we_wrf,ng_sn_wrf, ' rank = ', rank, ' fn_commonarea = ',fn_commonarea)
        area_common_dict[c][r]={}
        #... WRF grid from wrfinput_d01 ....
        X1 = XLONa[c  ,r  ]
        Y1 = XLATa[c  ,r  ]
        X2 = XLONa[c+1,r  ]
        Y2 = XLATa[c+1,r  ]
        X3 = XLONa[c+1,r+1]
        Y3 = XLATa[c+1,r+1]
        X4 = XLONa[c  ,r+1]
        Y4 = XLATa[c  ,r+1]
        #
        X1_m = XLONa_m[c  ,r  ]
        Y1_m = XLATa_m[c  ,r  ]
        X2_m = XLONa_m[c+1,r  ]
        Y2_m = XLATa_m[c+1,r  ]
        X3_m = XLONa_m[c+1,r+1]
        Y3_m = XLATa_m[c+1,r+1]
        X4_m = XLONa_m[c  ,r+1]
        Y4_m = XLATa_m[c  ,r+1]
        #
        # X  = np.array([X1,X2,X3,X4,X1])
        # Y  = np.array([Y1,Y2,Y3,Y4,Y1])
        X    = np.array([X1,X2,X3,X4])
        Y    = np.array([Y1,Y2,Y3,Y4])
        X_m  = np.array([X1_m,X2_m,X3_m,X4_m])
        Y_m  = np.array([Y1_m,Y2_m,Y3_m,Y4_m])
        # X, Y = poly2cw(X,Y,1) 
        Xmax = np.amax(X)
        Xmin = np.amin(X)
        Xdif = Xmax-Xmin
        Ymax = np.amax(Y)
        Ymin = np.amin(Y)
        if Xdif > 180:
          index = np.where(X < 0.0)
          X[index]  = X[index]+360.0
          Xmax = np.amax(X)
          Xmin = np.amin(X)
        #... min. and max. indices on emission grids ...
        #... Common to WRF grid ........................
        ind_lon_min = int((Xmin+180.0)/res)
        ind_lon_max = int((Xmax+180.0)/res)+1
        ind_lat_min = int((Ymin+ 90.0)/res)
        ind_lat_max = int((Ymax+ 90.0)/res)+1
        #
        area_common_array[c,r,0] = ind_lon_min
        area_common_array[c,r,1] = ind_lon_max
        area_common_array[c,r,2] = ind_lat_min
        area_common_array[c,r,3] = ind_lat_max
        area_count               = 1
        #
        if 1==0:
          if Ymin > 85.0 or Ymax < -85.0:
            continue
          area_all                 = 0.0
          
          for ilon in np.arange(ind_lon_min,ind_lon_max):
            for ilat in np.arange(ind_lat_min,ind_lat_max):
              XX          = np.array([lon_bound_glob_2D[ilon  ,ilat  ],lon_bound_glob_2D[ilon+1,ilat  ],\
                                      lon_bound_glob_2D[ilon+1,ilat+1],lon_bound_glob_2D[ilon  ,ilat+1]])
              YY          = np.array([lat_bound_glob_2D[ilon  ,ilat  ],lat_bound_glob_2D[ilon+1,ilat  ],\
                                      lat_bound_glob_2D[ilon+1,ilat+1],lat_bound_glob_2D[ilon  ,ilat+1]])
              area_common3_temp = area_common3(X,Y,XX,YY)
              area_common_array[c,r,3+area_count] = area_common3_temp
              #
              area_all    = area_all + area_common_array[c,r,3+area_count]
              area_count += 1
              #
              if area_count > 996:
                print('area_count is greater than 996')
                print('res = ',res)
                print((ind_lon_max-ind_lon_min)*(ind_lat_max-ind_lat_min))
                print(ind_lat_min, ind_lat_max, ind_lon_min, ind_lon_max)
                print(X)
                print(Y)
                sys.exit(1) 
          # End of for ilon and ilat
          # print 'area_all = ', area_all
          # print c,'/',ng_we_wrf,' ',r,'/',ng_sn_wrf
          #plt.show()
          #raw_input('check here5')


        if 1==1:
          area_all   = 0.0
          dict_count = 0
          for ilon in np.arange(ind_lon_min,ind_lon_max):
            for ilat in np.arange(ind_lat_min,ind_lat_max):
              ### XX          = np.array([lon_bound_glob_2D[ilon  ,ilat  ],lon_bound_glob_2D[ilon+1,ilat  ],\
              ###                         lon_bound_glob_2D[ilon+1,ilat+1],lon_bound_glob_2D[ilon  ,ilat+1]])
              ### YY          = np.array([lat_bound_glob_2D[ilon  ,ilat  ],lat_bound_glob_2D[ilon+1,ilat  ],\
              ###                         lat_bound_glob_2D[ilon+1,ilat+1],lat_bound_glob_2D[ilon  ,ilat+1]])
              XX          = np.array([lon_m[ilon  ,ilat  ],lon_m[ilon+1,ilat  ],\
                                      lon_m[ilon+1,ilat+1],lon_m[ilon  ,ilat+1]])
              YY          = np.array([lat_m[ilon  ,ilat  ],lat_m[ilon+1,ilat  ],\
                                      lat_m[ilon+1,ilat+1],lat_m[ilon  ,ilat+1]])
             
              area_common3_temp = area_common3(X_m,Y_m,XX,YY)
     
              area_all    = area_all + area_common3_temp
              #print('c,r, ilon, ilat, dict_count, area_common3_temp = ',datetime.now(),':',\
              #      c,r,ilon,ilat,dict_count, \
              #      '(',(1+ind_lon_max-ind_lon_min)*(1+ind_lat_max-ind_lat_min),')', \
              #      area_common3_temp, area_all)
              #
              if area_common3_temp != 0.0: 
                area_common_dict[c][r][dict_count]={}
                area_common_dict[c][r][dict_count]['area']=area_common3_temp
                area_common_dict[c][r][dict_count]['ilon']=ilon
                area_common_dict[c][r][dict_count]['ilat']=ilat
                dict_count += 1
                area_common_dict[c][r]['total_count'] = dict_count
                ### if rank == 0:
                ###   if dict_count % 10 == 0:
                ###     print("dict_count = ",dict_count, " rank = ",rank)
              if area_all >= 1.0:
                continue

              #
          # End of for ilon and ilat
          # print 'area_all = ', area_all
          # print c,'/',ng_we_wrf,' ',r,'/',ng_sn_wrf
          #plt.show()
          #raw_input('check here5')
        # END of if 1==1:
      # End of loop r
    # End of loop c
    np.save(fn_commonarea,area_common_array)
    with open(fn_commonarea_dict,'wb') as f:
      pickle.dump(area_common_dict,f)
  else:
    area_common_array = np.load(fn_commonarea)
    with open(fn_commonarea_dict,'rb') as f:
      area_common_dict = pickle.load(f)
  # End of "if not os.path.isfile(fn_commonarea) ..........

  return area_common_array, area_common_dict

#--------------------------------------------------------------------------
#
#   Get start_time and end_time from namelist
#
#--------------------------------------------------------------------------

def get_Start_datetime_and_End_datetime(namelist):
  
  with open(namelist,"r",encoding="utf-8") as f:
    for line in f:
      line_split = line.lower().split('=')
      line_split = [x.strip() for x in line_split]
      
      if 'start_year' in line_split:
        start_year = int(line_split[1].split(',')[0])
      if 'start_month' in line_split:
        start_month = int(line_split[1].split(',')[0])
      if 'start_day' in line_split:
        start_day = int(line_split[1].split(',')[0])
      if 'start_hour' in line_split:
        start_hour = int(line_split[1].split(',')[0])
      if 'start_minute' in line_split:
        start_minute = int(line_split[1].split(',')[0])
      if 'start_second' in line_split:
        start_second = int(line_split[1].split(',')[0])
      if 'end_year' in line_split:
        end_year = int(line_split[1].split(',')[0])
      if 'end_month' in line_split:
        end_month = int(line_split[1].split(',')[0])
      if 'end_day' in line_split:
        end_day = int(line_split[1].split(',')[0])
      if 'end_hour' in line_split:
        end_hour = int(line_split[1].split(',')[0])
      if 'end_minute' in line_split:
        end_minute = int(line_split[1].split(',')[0])
      if 'end_second' in line_split:
        end_second = int(line_split[1].split(',')[0])
      if 'max_dom' in line_split:
        max_dom    = int(line_split[1].split(',')[0])
  start_dt = datetime(start_year, start_month, start_day, start_hour, start_minute, start_second)
  end_dt = datetime(end_year, end_month, end_day, end_hour, end_minute, end_second)
  
  return start_dt, end_dt, max_dom
#
#----------------------------------------------------------------------
#
def Speciate_RCP60_SHP_OnWRF(emis_dict,chem='SAPRC'):
  #
  # Input  unit : [kg/m2/sec]
  # output unit : [mol/m2/sec] or [ug/m2/sec] (BC and OC)
  #
  days       = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
  #------------------------------------------------------------------------------------------------
  #
  # PART A : change the unit of 'shp_CO','shp_CH4','shp_BC','shp_OC','shp_SO2','shp_NH3','shp_NO'
  #
  #------------------------------------------------------------------------------------------------
  RCP_keynames = ['shp-RCP60-CO','shp-RCP60-CH4','shp-RCP60-BC','shp-RCP60-OC','shp-RCP60-SO2','shp-RCP60-NH3','shp-RCP60-NO']
  RCP_Mmolaire = {'CO':28,'CH4':16,'SO2':64,'NH3':17,'NO':30}
  sector = 'shp'
  #
  for ikey, RCP_keyname in enumerate(RCP_keynames):
    spec = RCP_keyname.split('-RCP60-')[1]
    data = np.zeros_like(emis_dict[RCP_keyname]['voc']['data'])
    keyname = RCP_keyname+'_new'
    for imonth in np.arange(12):
      if spec in RCP_Mmolaire:
        # Unit : [kg/sec/m2] --> [mol/sec/m2](%/100 of mass)
        # print 'check ----1 ', RCP_keyname, emis_dict[RCP_keyname]['voc']['units']
        data[:,:,imonth] = emis_dict[RCP_keyname]['voc']['data'][:,:,imonth] \
                   *1.e3 \
                   /RCP_Mmolaire[spec]
        units = 'mol/sec/m2'
      else:
        # Unit : [kg/m2/sec] --> [ug/m2/sec]
        # print 'check ----2 ', RCP_keyname, emis_dict[RCP_keyname]['voc']['units']
        data[:,:,imonth] = emis_dict[RCP_keyname]['voc']['data'][:,:,imonth] \
                   *1.e9 
        units = 'ug/m2/sec'
    # END of for imonth in np.arange(12)
    emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
    emis_dict[keyname]['dimensions']={}
    emis_dict[keyname]['dimensions']['south_north']= emis_dict[RCP_keyname]['dimensions']['south_north'] # of grid points in sn direction 
    emis_dict[keyname]['dimensions']['west_east']  = emis_dict[RCP_keyname]['dimensions']['west_east'] # of grid points in we direction
    emis_dict[keyname]['dimensions']['time']       = 12 # of points in time series> e.g.:12
    emis_dict[keyname]['west_east']={}
    emis_dict[keyname]['west_east']['dtype']='i4'
    emis_dict[keyname]['west_east']['dims' ]=['west_east']
    emis_dict[keyname]['west_east']['units']=''
    emis_dict[keyname]['west_east']['data' ]=emis_dict[RCP_keyname]['west_east']['data'] # : [0,1,2,3,4,5,...,ng_we_wrf-1]
    emis_dict[keyname]['south_north']={}
    emis_dict[keyname]['south_north']['dtype']='i4'
    emis_dict[keyname]['south_north']['dims' ]=['south_north']
    emis_dict[keyname]['south_north']['units']=''
    emis_dict[keyname]['south_north']['data' ]=emis_dict[RCP_keyname]['south_north']['data'] # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
    emis_dict[keyname]['longitude']={}
    emis_dict[keyname]['longitude']['dtype']='f4'
    emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
    emis_dict[keyname]['longitude']['units']='degrees_east'
    emis_dict[keyname]['longitude']['data' ]=emis_dict[RCP_keyname]['longitude']['data'] # <WRF longitude grid>
    emis_dict[keyname]['latitude']={}
    emis_dict[keyname]['latitude']['dtype']='f4'
    emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
    emis_dict[keyname]['latitude']['units']='degrees_east'
    emis_dict[keyname]['latitude']['data' ]=emis_dict[RCP_keyname]['latitude']['data']  # <WRF latitude grid>
    emis_dict[keyname]['time']={}
    emis_dict[keyname]['time']['dtype']='i4'
    emis_dict[keyname]['time']['dims' ]=['time']
    emis_dict[keyname]['time']['data' ]=np.arange(12) #: [0,1,2,3,4,...,11]
    emis_dict[keyname]['voc']={}
    emis_dict[keyname]['voc']['dtype']='f4'
    # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] or ['west_east','south_north']
    emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time']
    emis_dict[keyname]['voc']['units']= units
    emis_dict[keyname]['voc']['data' ]= data[:,:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]

    emis_dict[RCP_keyname] = emis_dict[keyname]
    del emis_dict[keyname]
  # END of for ikey, RCP_keyname

  

  #------------------------------------------------------
  #
  # PART B : Speciate for 'shp_NMVOC'     
  #
  #------------------------------------------------------

  table, table_2007, total_voc_2007 = Get_Anthropogenic_Emissions_From_IIASA()
  #
  RCP_keynames = ['shp-RCP60-NMVOC']
  sector_index = {'shp':7}
  if chem == 'CBMZ':
    spec_list    = ['C2H5OH','CH3OH','HC5','ETH','TOL','OL2','KET','XYL','HCHO','OLT','ALD','ISO','OLI']
  elif chem == 'SAPRC':
    spec_list  = ['C2H6','C3H8','C2H2','ALK3','ALK4','ALK5','ETHENE','C3H6','OLE2','ARO1','ARO2','ECHO','CCHO','ACET','MEK','TERP','MEOH','PROD2']
    
  # LOOP over sectors {'ene':0, 'dom':1, 'ind1':2, 'ind2':3, 'flr':4, 'slv':5, 'tra':6, 'shp':7, 'wst':8}
  sector = 'shp'
  #
  RCP_keyname = 'shp-RCP60-NMVOC'
  if rank == 0:
    print(RCP_keyname in emis_dict)
  #
  # LOOP over CBMZ :<spec> = ['C2H5OH','CH3OH','HC5','ETH','TOL','OL2','KET','XYL','HCHO','OLT','ALD','ISO','OLI']
  #           SAPRC:<spec> = ['C2H6','C3H8','C2H2','ALK3','ALK4','ALK5','ETHENE','C3H6','OLE2','ARO1','ARO2','ECHO','CCHO','ACET','MEK','TERP','MEOH','PROD2']
  index = sector_index['shp']
  for spec in spec_list:
    keyname = sector+'-RCP60-'+spec
    data = np.zeros_like(emis_dict[RCP_keyname]['voc']['data']) 
    for key, value in table.items():
      if spec in value[chem][chem+' name']:
        for imonth in np.arange(12):
          #  Unit : [kg/sec/m2] --> [mol/sec/m2](%/100 of mass) 
          data[:,:,imonth] += emis_dict[RCP_keyname]['voc']['data'][:,:,imonth] \
                       *1.e3 \
                       *value[chem][spec] \
                       *value['voc'][index]/total_voc_2007[index] \
                       /value['Molecular Mass']
        # END of for imonth in np.arange(12)
      # END of if spec in value[chem][chem+' name']
    # for key, value in table.iteritems():

    emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
    emis_dict[keyname]['dimensions']={}
    emis_dict[keyname]['dimensions']['south_north']= emis_dict[RCP_keyname]['dimensions']['south_north'] # of grid points in sn direction 
    emis_dict[keyname]['dimensions']['west_east']  = emis_dict[RCP_keyname]['dimensions']['west_east'] # of grid points in we direction
    emis_dict[keyname]['dimensions']['time']       = 12 # of points in time series> e.g.:12
    emis_dict[keyname]['west_east']={}
    emis_dict[keyname]['west_east']['dtype']='i4'
    emis_dict[keyname]['west_east']['dims' ]=['west_east']
    emis_dict[keyname]['west_east']['units']=''
    emis_dict[keyname]['west_east']['data' ]=emis_dict[RCP_keyname]['west_east']['data'] # : [0,1,2,3,4,5,...,ng_we_wrf-1]
    emis_dict[keyname]['south_north']={}
    emis_dict[keyname]['south_north']['dtype']='i4'
    emis_dict[keyname]['south_north']['dims' ]=['south_north']
    emis_dict[keyname]['south_north']['units']=''
    emis_dict[keyname]['south_north']['data' ]=emis_dict[RCP_keyname]['south_north']['data'] # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
    emis_dict[keyname]['longitude']={}
    emis_dict[keyname]['longitude']['dtype']='f4'
    emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
    emis_dict[keyname]['longitude']['units']='degrees_east'
    emis_dict[keyname]['longitude']['data' ]=emis_dict[RCP_keyname]['longitude']['data'] # <WRF longitude grid>
    emis_dict[keyname]['latitude']={}
    emis_dict[keyname]['latitude']['dtype']='f4'
    emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
    emis_dict[keyname]['latitude']['units']='degrees_east'
    emis_dict[keyname]['latitude']['data' ]=emis_dict[RCP_keyname]['latitude']['data']  # <WRF latitude grid>
    emis_dict[keyname]['time']={}
    emis_dict[keyname]['time']['dtype']='i4'
    emis_dict[keyname]['time']['dims' ]=['time']
    emis_dict[keyname]['time']['data' ]=np.arange(12) # : [0,1,2,3,4,...,11]
    emis_dict[keyname]['voc']={}
    emis_dict[keyname]['voc']['dtype']='f4'
    # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] or ['west_east','south_north']
    emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time']
    emis_dict[keyname]['voc']['units']='mol/sec/m2'
    emis_dict[keyname]['voc']['data' ]= data[:,:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
  # END of for spec in spec_list:
  del emis_dict[RCP_keyname]
  # END of for sector, index in sector_index.iteritems():
  return emis_dict

#
#----------------------------------------------------------------------
#
def Speciate_RCP60_SOIL_OnWRF(emis_dict):
  days       = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
  #------------------------------------------------------------------------------------------------
  #
  # PART A : change the unit of 'shp_CO','shp_CH4','shp_BC','shp_OC','shp_SO2','shp_NH3','shp_NO'
  #
  #------------------------------------------------------------------------------------------------
  RCP_keynames = ['shp_CO','shp_CH4','shp_BC','shp_OC','shp_SO2','shp_NH3','shp_NO']
  RCP_Mmolaire = {'CO':28,'CH4':16,'SO2':64,'NH3':17,'NO':30}
  sector = 'shp'
  #
  for ikey, RCP_keyname in enumerate(RCP_keynames):
    spec = RCP_keyname.split('_')[1]
    data = np.zeros_like(emis_dict[RCP_keyname]['voc']['data'])
    keyname = RCP_keyname+'_new'
    for imonth in np.arange(12):
      if spec in RCP_Mmolaire:
        # Unit : [kg/sec/m2] --> [mol/month/km2](%/100 of mass) 
        data[:,:,imonth] = emis_dict[RCP_keyname]['voc']['data'][:,:,imonth] \
                   *1.e9*3600.0*24.0*days[imonth] \
                   /RCP_Mmolaire[spec]
        units = 'mol/month/km2'
      else:
        # Unit : [kg/m2/sec] --> [ug/m2/sec]
        data[:,:,imonth] = emis_dict[RCP_keyname]['voc']['data'][:,:,imonth] \
                   *1.e9 
        units = 'ug/m2/sec'
    # END of for imonth in np.arange(12)
    emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
    emis_dict[keyname]['dimensions']={}
    emis_dict[keyname]['dimensions']['south_north']= emis_dict[RCP_keyname]['dimensions']['south_north'] # of grid points in sn direction 
    emis_dict[keyname]['dimensions']['west_east']  = emis_dict[RCP_keyname]['dimensions']['west_east'] # of grid points in we direction
    emis_dict[keyname]['dimensions']['time']       = 12 # of points in time series> e.g.:12
    emis_dict[keyname]['west_east']={}
    emis_dict[keyname]['west_east']['dtype']='i4'
    emis_dict[keyname]['west_east']['dims' ]=['west_east']
    emis_dict[keyname]['west_east']['units']=''
    emis_dict[keyname]['west_east']['data' ]=emis_dict[RCP_keyname]['west_east']['data'] # : [0,1,2,3,4,5,...,ng_we_wrf-1]
    emis_dict[keyname]['south_north']={}
    emis_dict[keyname]['south_north']['dtype']='i4'
    emis_dict[keyname]['south_north']['dims' ]=['south_north']
    emis_dict[keyname]['south_north']['units']=''
    emis_dict[keyname]['south_north']['data' ]=emis_dict[RCP_keyname]['south_north']['data'] # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
    emis_dict[keyname]['longitude']={}
    emis_dict[keyname]['longitude']['dtype']='f4'
    emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
    emis_dict[keyname]['longitude']['units']='degrees_east'
    emis_dict[keyname]['longitude']['data' ]=emis_dict[RCP_keyname]['longitude']['data'] # <WRF longitude grid>
    emis_dict[keyname]['latitude']={}
    emis_dict[keyname]['latitude']['dtype']='f4'
    emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
    emis_dict[keyname]['latitude']['units']='degrees_east'
    emis_dict[keyname]['latitude']['data' ]=emis_dict[RCP_keyname]['latitude']['data']  # <WRF latitude grid>
    emis_dict[keyname]['time']={}
    emis_dict[keyname]['time']['dtype']='i4'
    emis_dict[keyname]['time']['dims' ]=['time']
    emis_dict[keyname]['time']['data' ]=np.arange(12) #: [0,1,2,3,4,...,11]
    emis_dict[keyname]['voc']={}
    emis_dict[keyname]['voc']['dtype']='f4'
    # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] or ['west_east','south_north']
    emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time']
    emis_dict[keyname]['voc']['units']= units
    emis_dict[keyname]['voc']['data' ]= data[:,:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]

    emis_dict[RCP_keyname] = emis_dict[keyname]
    del emis_dict[keyname]
  # END of for ikey, RCP_keyname
  return emis_dict
#----------------------------------------------------------------------
#
def Speciate_SAPRC_Anth_VOC_OnWRF(emis_dict):
  # 
  # Input unit : [kg/m2/sec]
  #
  table, table_2007, total_voc_2007 = Get_Anthropogenic_Emissions_From_IIASA()
  #
  # In registry.chem
  # package   esaprcnov       emiss_opt==13                  -             emis_ant:e_so2,e_c2h6,e_c3h8,e_c2h2,
  #           e_alk3,e_alk4,e_alk5,e_ethene,e_c3h6,e_ole1,e_ole2,e_aro1,e_aro2,e_hcho,e_ccho,e_rcho,e_acet,e_mek,e_isoprene,
  #           e_terp,e_sesq,e_co,e_no,e_no2,e_phen,e_cres,e_meoh,e_gly,e_mgly,e_bacl,e_isoprod,e_methacro,e_mvk,e_prod2,e_ch4,
  #           e_bald,e_hcooh,e_cco_oh,e_rco_oh,e_dms_oc,e_nh3,e_pm25i,e_pm25j,e_eci,e_ecj,e_orgi,e_orgj,e_so4i,e_so4j,e_no3i,
  #           e_no3j,e_orgi_a,e_orgj_a,e_orgi_bb,e_orgj_bb
  #           .............................................................................................................
  #
  #           From "Development of a Condensed SAPRC-07 chemical mechanism", William P.L.Carter 2010


  #           ___Active Inorgnic Species_________
  #           
  #           co              : Carbon monoxide
  #           no              : Nitrogen monoxide, Nitric oxide
  #           no2             : Nitrogen dioxide
  #           so2             : Sulfur Dioxide (Active Inorganic Species)
  #           ___Explicit and Lumped model species (VOC)______
  #           c2h6            : Ethane  (VOC : ~ALK1)
  #           c3h8            : Propane (VOC : ~ALK2)
  #           c2h2            : Acetylene (no emissions here)
  #           alk3            : Alkanes and other non-aromatic compounds that react only with OH,
  #                             and have kOH between 2.5 x 10^3 and 5 x 10^3 ppm-1 min-1. For CS07 this
  #                             is used for all alkanes with kOH less than 5 x 10^3 ppm-1 min-1.            
  #           alk4            : Alkanes and other non-aromatic compounds that react only with OH,
  #                             and have kOH between 5 x 10^3 and 1 x 10^4 ppm-1 min-1. For CS07 this is
  #                             used for all alkanes with kOH greater than 5 x 10^3 ppm-1 min-1.
  #           alk5            : Alkanes and other non-aromatic compounds that react only with OH,
  #                                  and have kOH greater than 1 x 10^4 ppm-1 min-1.
  #           ole1            : Alkenes (other than ethene) with kOH < 7x10^4 ppm-1 min-1.
  #           ole2            : Alkenes with kOH > 7x10^4 ppm-1 min-1.
  #           aro1            : Aromatics with kOH < 2x10^4 ppm-1 min-1
  #           aro2            : Aromatics with kOH > 2x10^4 ppm-1 min-1
  #           terp            : Terpenes

  #           ___Primary Organics represented explicitly_____________________
  #           ch4             : Methane,
  #           ethene (C2H4)   : ETHENE (Ethylene) emissions
  #           c3h6            : Propene  emissions
  #           isoprene        : Isoprene emissions
  #
  #           ___Explicit and Lumped Molecule Reactive Organic Product Species____
  #           HCHO            : Formaldehyde
  #           ACET            : Acetone
  #           MEK             : Ketones and other non-aldehyde oxygenated products that react with OH
  #                             radicals faster than 5 x 10^-13 but slower than 5 x 10^-12 cm3 molec^-2 sec^-1
  #           sesq            : SESQUITERPEN emissions (C15H24) 
  #           meoh            : Methanol
  #           gly             : Glyoxal
  #           cres            : Cresols
  #           bald            : Aromatic aldehydes (e.g. benzaldehyde)
  #           phen            : Phenol
  #           hcooh           : Formic Acid
  #           cco_oh          : Acetic Acid: Also used for peroxyacetic acid
  #
  #           ___Lumped Products__________
  #           bacl            : Biacetyl
  #           mvk             : Methyl Vinyl Ketone
  #           mgly            : Methyl Glyoxal
  #           CCHO            : Acetaldehyde
  #           RCHO            : Lumped C3+ Aldehyde
  #           prod2           : Ketones and other non-aldehyde oxygenated products which react 
  #                             with OH radicals faster than 5 x 10^-12 cm3 molec-2 sec-1.

  # 
  #           isoprod         : Lumped other isoprene products : represents reactive isoprene products 
  #                             other than methacrolein and MVK, and also to represent other unsaturated
  #                             ketones or aldehydes
  #           methacro        : Methacrolein
  #           rco_oh          : Higher organic acids and peroxy acids (mechanism based on propionic acid)
  #           dms_oc
  #           nh3             : Ammoniac
  #           pm25i,pm25j     : PM 2.5 is portioned into the accumulation and nuclei mode and released 
  #                             into the atmosphere as PM25J and PM25I respectively.
  #           eci,ecj         : Elemental Carbon (*)
  #           orgi,orgj       : Organic Carbon (*)
  #           so4i,so4j       : Sulfate (*)
  #           no3i,no3j       : Nitrate (*)
  #           orgi_a,orgj_a   : Organic Carbon Aerosol (*) 
  #           orgi_bb,orgj_bb : Organic Carbon Biomass Burning (*)
  #           
  #           (*) suffix : I:nuclei/Aitken mode J:accumulation mode
  #
  input_keynames = ['dom_VOC','ene_VOC','ind_VOC','slv_VOC','tra_VOC','shp_VOC','wst_VOC']
  sector_index     = {'ene':0, 'dom':1, 'ind1':2, 'ind2':3, 'slv':5, 'tra':6, 'shp':7,'wst':8}
  SAPRC_spec       = ['C2H6','C3H8','C2H2','ALK3','ALK4','ALK5','ETHENE','C3H6','OLE2','ARO1','ARO2','ECHO','CCHO','ACET','MEK','TERP','MEOH','PROD2']

  # LOOP over sectors {'ene':0, 'dom':1, 'ind1':2, 'ind2':3, 'flr':4, 'slv':5, 'tra':6, 'shp':7, 'wst':8}
  for sector, index in sector_index.items():
    if 'ind' not in sector:
      sect = sector
      coef = 1.0
    else:
      sect = 'ind'
      if index == 2:
        coef   = 0.1444
      if index == 3:
        coef   = 0.8556
    #
    #input_VOC_keyname = sect+'_VOC'

    for keyname_temp in emis_dict.keys():
      if sect in keyname_temp and 'VOC' in keyname_temp:
        input_VOC_keyname = keyname_temp
        break

    if rank == 0:
      print(input_VOC_keyname)

    #
    # if it does not exist, skip it.
    #
    if emis_dict.get(input_VOC_keyname) == None:
      continue
    #
    # LOOP over SAPRC_spec = ['C2H6','C3H8','C2H2','ALK3','ALK4','ALK5','ETHENE','C3H6','OLE2','ARO1','ARO2','ECHO','CCHO','ACET','MEK','TERP','MEOH','PROD2']
    # LOOP over CBMZ_spec = ['C2H5OH','CH3OH','HC5','ETH','TOL','OL2','KET','XYL','HCHO','OLT','ALD','ISO','OLI']
    for spec in SAPRC_spec:
      
      keyname = sector+'-'+input_VOC_keyname.split('-')[1]+'-'+spec
      ## if rank == 0:
      ##   print('keyname = ',keyname)
      #if emis_dict.get(input_VOC_keyname) == None:
      #  continue
      data = np.zeros_like(emis_dict[input_VOC_keyname]['voc']['data']) 
      ## if rank == 0:
      ##   print('data.ndim = ', data.ndim)
      units = emis_dict[input_VOC_keyname]['voc']['units']
      if data.ndim == 2 and 'kg' in units and '/sec' in units and '/m2' in units:    # ECLIPSE and HTAP(AIR,SHIPS)
        for key, value in table.items():
          if spec in value['SAPRC']['SAPRC name']:
            #  Unit : [kg/sec/m2] --> [mol/sec/m2](%/100 of mass) 
            data[:,:] += emis_dict[input_VOC_keyname]['voc']['data'][:,:]*coef*1.e3 \
                       *value['SAPRC'][spec] \
                       *value['voc'][index]/total_voc_2007[index] \
                       /value['Molecular Mass']
          # END of if spec in value['SAPRC']['SAPRC name']
        # END of for key, value in table.iteritems():
      elif data.ndim == 3 and 'kg' in units and '/sec' in units and '/m2' in units:  # HTAP (Except for AIR,SHIPS)
        ## if rank == 0:
        ##   print(data.ndim, data.shape[2])
        if data.shape[2] == 12:
          for imonth in np.arange(12):
            for key, value in table.items():
              ## if rank == 0:
              ##   print('key, value, month, spec and SAPRC name = ',key, imonth, spec, value['SAPRC']['SAPRC name'])
              if spec in value['SAPRC']['SAPRC name']:
                #  HTAP unit : [kg/sec/m2] --> [mol/sec/m2] 
                #   
                data[:,:,imonth] += emis_dict[input_VOC_keyname]['voc']['data'][:,:,imonth]*coef*1.e3 \
                           *value['SAPRC'][spec] \
                           *value['voc'][index]/total_voc_2007[index] \
                           /value['Molecular Mass']
              # END of if spec in value['SAPRC']['SAPRC name']
            # END of for key, value in table.iteritems():
          # END of loop on month
        # END of if data.shape[2] == 12:

      # END of elif data.ndim == 3:
      #
      #----------------------------------------------------------------------------------------------
      #
      emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
      emis_dict[keyname]['dimensions']={}
      emis_dict[keyname]['dimensions']['south_north']= emis_dict[input_VOC_keyname]['dimensions']['south_north'] # of grid points in sn direction 
      emis_dict[keyname]['dimensions']['west_east']  = emis_dict[input_VOC_keyname]['dimensions']['west_east'] # of grid points in we direction
      if data.ndim == 3:
        emis_dict[keyname]['dimensions']['time']       = 12
      emis_dict[keyname]['west_east']={}
      emis_dict[keyname]['west_east']['dtype']='i4'
      emis_dict[keyname]['west_east']['dims' ]=['west_east']
      emis_dict[keyname]['west_east']['units']=''
      emis_dict[keyname]['west_east']['data' ]=emis_dict[input_VOC_keyname]['west_east']['data'] # : [0,1,2,3,4,5,...,ng_we_wrf-1]
      emis_dict[keyname]['south_north']={}
      emis_dict[keyname]['south_north']['dtype']='i4'
      emis_dict[keyname]['south_north']['dims' ]=['south_north']
      emis_dict[keyname]['south_north']['units']=''
      emis_dict[keyname]['south_north']['data' ]=emis_dict[input_VOC_keyname]['south_north']['data'] # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
      emis_dict[keyname]['longitude']={}
      emis_dict[keyname]['longitude']['dtype']='f4'
      emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
      emis_dict[keyname]['longitude']['units']='degrees_east'
      emis_dict[keyname]['longitude']['data' ]=emis_dict[input_VOC_keyname]['longitude']['data'] # <WRF longitude grid>
      emis_dict[keyname]['latitude']={}
      emis_dict[keyname]['latitude']['dtype']='f4'
      emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
      emis_dict[keyname]['latitude']['units']='degrees_east'
      emis_dict[keyname]['latitude']['data' ]=emis_dict[input_VOC_keyname]['latitude']['data']  # <WRF latitude grid>
      if data.ndim == 3:
        emis_dict[keyname]['time']={}
        emis_dict[keyname]['time']['dtype']='i4'
        emis_dict[keyname]['time']['dims' ]=['time']
        emis_dict[keyname]['time']['data' ]=np.arange(12) # [0,1,2,3,4,...,11]
      emis_dict[keyname]['voc']={}
      emis_dict[keyname]['voc']['dtype']='f4'
      if data.ndim == 3:  #HTAP
        emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] or ['west_east','south_north']
        emis_dict[keyname]['voc']['units']='mol/sec/m2'
        emis_dict[keyname]['voc']['data' ]= data[:,:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
      elif data.ndim == 2:  #ECLIPSE and HTAP (AIR, SHIPS)
        emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
        emis_dict[keyname]['voc']['units']='mol/sec/m2'
        emis_dict[keyname]['voc']['data' ]= data[:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>]
    # END of for spec in SAPRC_spec:
    # emis_dict[input_VOC_keyname] = None
  # END of for sector, index in sector_index.iteritems():
  #
  for keyname in input_keynames: # ['dom_VOC','ene_VOC','ind_VOC','slv_VOC','tra_VOC','wst_VOC']
    if emis_dict.get(keyname) != None:
      del emis_dict[keyname]
  #
  return emis_dict

#
#----------------------------------------------------------------------
#
def Speciate_CBMZ_Anth_VOC_OnWRF(emis_dict):
  #
  table, table_2007, total_voc_2007 = Get_Anthropogenic_Emissions_From_IIASA()
  #
  input_keynames = ['dom_VOC','ene_VOC','ind_VOC','slv_VOC','tra_VOC','wst_VOC']
  sector_index     = {'ene':0, 'dom':1, 'ind1':2, 'ind2':3, 'slv':5, 'tra':6, 'wst':8}
  CBMZ_spec        = ['C2H5OH','CH3OH','HC5','ETH','TOL','OL2','KET','XYL','HCHO','OLT','ALD','ISO','OLI']
  SAPRC_spec       = ['C2H6','C3H8','C2H2','ALK3','ALK4','ALK5','ETHENE','C3H6','OLE1','OLE2','ARO1','ARO2','CCHO','HCHO','ACET','MEK','TERP','MEOH','PROD2']

  # LOOP over sectors {'ene':0, 'dom':1, 'ind1':2, 'ind2':3, 'flr':4, 'slv':5, 'tra':6, 'shp':7, 'wst':8}
  for sector, index in sector_index.items():
    if 'ind' not in sector:
      sect = sector
      coef = 1.0
    else:
      sect = 'ind'
      if index == 2:
        coef   = 0.1444
      if index == 3:
        coef   = 0.8556
    #
    #input_VOC_keyname = sect+'_VOC'
    for keyname_temp in emis_dict.keys():
      if sect in keyname_temp and 'VOC' in keyname_temp:
        input_VOC_keyname = keyname_temp
        break
    if input_VOC_keyname not in emis_dict:
      continue
    #
    # LOOP over CBMZ_spec = ['C2H5OH','CH3OH','HC5','ETH','TOL','OL2','KET','XYL','HCHO','OLT','ALD','ISO','OLI']
    for spec in CBMZ_spec:
      #keyname = sector+'_'+spec
      keyname = sector+'-'+input_VOC_keyname.split('-')[1]+'-'+spec
      if rank == 0:
        print(keyname)
      data = np.zeros_like(emis_dict[input_VOC_keyname]['voc']['data']) 
      if rank == 0:
        print('data.ndim = ', data.ndim)
      units = emis_dict[input_VOC_keyname]['voc']['units']
      if data.ndim == 2 and 'kg' in units and '/sec' in units and '/m2' in units:    # ECLIPSE and HTAP(AIR,SHIPS)
        for key, value in table.items():
          if spec in value['CBMZ']['CBMZ name']:
            #  Unit : [kt/year/km2] --> [mol/year/km2](%/100 of mass) 
            data[:,:] += emis_dict[input_VOC_keyname]['voc']['data'][:,:]*coef*1.e9 \
                       *value['CBMZ'][spec] \
                       *value['voc'][index]/total_voc_2007[index] \
                       /value['Molecular Mass']
          # END of if spec in value['CBMZ']['CBMZ name']
        # END of for key, value in table.iteritems():
      elif data.ndim == 3 and 'kg' in units and '/sec' in units and '/m2' in units:  # HTAP (Except for AIR,SHIPS)
        if rank == 0:
          print(data.ndim, data.shape[2])
        if data.shape[2] == 12:
          for imonth in np.arange(12):
            for key, value in table.items():
              ## if rank == 0:
              ##   print('key, value, month, spec and CBMZ anme = ',key, imonth, spec, value['CBMZ']['CBMZ name'])
              if spec in value['CBMZ']['CBMZ name']:
                #  HTAP unit : [kg/sec/m2] --> [mol/sec/m2] 
                #   
                data[:,:,imonth] += emis_dict[input_VOC_keyname]['voc']['data'][:,:,imonth]*coef*1.e3 \
                           *value['CBMZ'][spec] \
                           *value['voc'][index]/total_voc_2007[index] \
                           /value['Molecular Mass']
              # END of if spec in value['CBMZ']['CBMZ name']
            # END of for key, value in table.iteritems():
          # END of loop on month
        # END of if data.shape[2] == 12:

      # END of elif data.ndim == 3:

      emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
      emis_dict[keyname]['dimensions']={}
      emis_dict[keyname]['dimensions']['south_north']= emis_dict[input_VOC_keyname]['dimensions']['south_north'] # of grid points in sn direction 
      emis_dict[keyname]['dimensions']['west_east']  = emis_dict[input_VOC_keyname]['dimensions']['west_east'] # of grid points in we direction
      if data.ndim == 3:
        emis_dict[keyname]['dimensions']['time']       = 12
      emis_dict[keyname]['west_east']={}
      emis_dict[keyname]['west_east']['dtype']='i4'
      emis_dict[keyname]['west_east']['dims' ]=['west_east']
      emis_dict[keyname]['west_east']['units']=''
      emis_dict[keyname]['west_east']['data' ]=emis_dict[input_VOC_keyname]['west_east']['data'] # : [0,1,2,3,4,5,...,ng_we_wrf-1]
      emis_dict[keyname]['south_north']={}
      emis_dict[keyname]['south_north']['dtype']='i4'
      emis_dict[keyname]['south_north']['dims' ]=['south_north']
      emis_dict[keyname]['south_north']['units']=''
      emis_dict[keyname]['south_north']['data' ]=emis_dict[input_VOC_keyname]['south_north']['data'] # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
      emis_dict[keyname]['longitude']={}
      emis_dict[keyname]['longitude']['dtype']='f4'
      emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
      emis_dict[keyname]['longitude']['units']='degrees_east'
      emis_dict[keyname]['longitude']['data' ]=emis_dict[input_VOC_keyname]['longitude']['data'] # <WRF longitude grid>
      emis_dict[keyname]['latitude']={}
      emis_dict[keyname]['latitude']['dtype']='f4'
      emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
      emis_dict[keyname]['latitude']['units']='degrees_east'
      emis_dict[keyname]['latitude']['data' ]=emis_dict[input_VOC_keyname]['latitude']['data']  # <WRF latitude grid>
      if data.ndim == 3:
        emis_dict[keyname]['time']={}
        emis_dict[keyname]['time']['dtype']='i4'
        emis_dict[keyname]['time']['dims' ]=['time']
        emis_dict[keyname]['time']['data' ]=np.arange(12) # [0,1,2,3,4,...,11]
      emis_dict[keyname]['voc']={}
      emis_dict[keyname]['voc']['dtype']='f4'
      if data.ndim == 3:  #HTAP
        emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] or ['west_east','south_north']
        emis_dict[keyname]['voc']['units']='mol/sec/m2'
        emis_dict[keyname]['voc']['data' ]= data[:,:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
      elif data.ndim == 2:  #ECLIPSE and HTAP (AIR, SHIPS)
        emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
        emis_dict[keyname]['voc']['units']='mol/sec/m2'
        emis_dict[keyname]['voc']['data' ]= data[:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>]
    # END of for spec in CBMZ_spec:
    # emis_dict[input_VOC_keyname] = None
  # END of for sector, index in sector_index.iteritems():
  #
  for keyname in input_keynames: # ['dom_VOC','ene_VOC','ind_VOC','slv_VOC','tra_VOC','wst_VOC']
    if keyname in emis_dict:
      del emis_dict[keyname]
  #
  return emis_dict

#
#------------------------------------------------------------------------------------------------------
#
def Get_Anthropogenic_Emissions_From_IIASA():
  #
  total_voc_2007 = np.array([3940.284,34468.024,2347.,103725.,149622.,276421.229,58168.,22653.,20661.])
  #
  table_2007 =np.array([\
           [0       ,6552    ,54      ,55044   ,0       ,41527   ,0       ,0       ,630   ],\
           [186     ,1314    ,291     ,2936    ,41243   ,19314   ,4667    ,228     ,53    ],\
           [242     ,3778    ,95      ,1195    ,29762   ,0       ,1282    ,221     ,5956  ],\
           [0       ,0       ,0       ,1486    ,0       ,28719   ,0       ,0       ,163   ],\
           [140     ,1530    ,139     ,1594    ,20396   ,3777    ,430     ,152     ,5642  ],\
           [94      ,689     ,89      ,3392    ,165     ,10766   ,5431    ,1792    ,364   ],\
           [49      ,7186    ,107     ,3570    ,24      ,0       ,6107    ,3734    ,1013  ],\
           [18      ,6       ,15      ,1560    ,0       ,18078   ,578     ,48      ,3     ],\
           [125     ,713     ,320     ,1593    ,15446   ,434     ,2866    ,135     ,46    ],\
           [57      ,1241    ,112     ,929     ,7613    ,48      ,6095    ,318     ,34    ],\
           [460     ,102     ,41      ,1916    ,58      ,12319   ,1658    ,540     ,167   ],\
           [124     ,105     ,51      ,3226    ,8004    ,2612    ,2613    ,84      ,240   ],\
           [88      ,8213    ,339     ,1584    ,541     ,0.043   ,2218    ,2853    ,985   ],\
           [2002    ,673     ,527     ,292     ,39      ,24      ,3815    ,2511    ,3761  ],\
           [0       ,0       ,0       ,575     ,0       ,12335   ,0       ,0       ,132   ],\
           [17      ,447     ,12      ,210     ,9084    ,973     ,1998    ,120     ,17    ],\
           [0       ,0       ,0       ,636     ,0       ,11677   ,176     ,8       ,30    ],\
           [0       ,0       ,0       ,2086    ,0       ,9983    ,0       ,0       ,148   ],\
           [0.107   ,16      ,0       ,681     ,20      ,8284    ,562     ,1345    ,0     ],\
           [0       ,0       ,0       ,193     ,0       ,10132   ,0       ,0       ,47    ],\
           [59      ,1322    ,29      ,3690    ,14      ,0.009   ,2547    ,1092    ,58    ],\
           [0.107   ,0.003   ,0       ,464     ,4       ,5408    ,1906    ,453     ,0     ],\
           [134     ,35      ,25      ,1538    ,17      ,4717    ,1278    ,348     ,270   ],\
           [0       ,4       ,0       ,561     ,0       ,7717    ,0       ,0       ,36    ],\
           [0       ,0       ,0       ,1166    ,0       ,7010    ,0       ,0       ,50    ],\
           [18      ,281     ,1       ,284     ,7857    ,1472    ,620     ,102     ,0     ],\
           [0       ,0       ,0       ,673     ,0       ,5806    ,0       ,0       ,0     ],\
           [0.214   , 27     ,0       ,182     ,6907    ,1277    ,274     ,37      ,0     ],\
           [3       ,79      ,18      ,819     ,12      ,3314    ,1282    ,417     ,130   ],\
           [102     ,58      ,13      ,660     ,27      ,3091    ,1432    ,468     ,95    ],\
           [0       ,0       ,0       ,119     ,0       ,5661    ,0       , 0      ,272   ],\
           [0.107   ,23      ,0       ,425     ,51      ,4994    ,140     ,338     ,0     ],\
           [0.107   ,0.003   ,0       ,354     ,0       ,4317    ,0       ,639     ,0     ],\
           [0       ,0       ,0       ,213     ,0       ,4163    ,0       ,0       ,15    ],\
           [0       ,59      ,0       ,575     ,178     ,0       ,1718    ,1344    ,11    ],\
           [15      ,7       ,57      ,625     ,11      ,0.177   ,2109    ,876     ,0     ],\
           [0.321   ,0.009   ,0       ,681     ,0       ,0       ,1940    ,1281    ,0     ],\
           [0       ,0       ,0       ,60      ,0       ,3535    ,0       ,0       ,86    ],\
           [0       ,0       ,0       ,96      ,0       ,3360    ,0       ,0       ,0     ],\
           [3       ,5       ,7       ,876     ,1389    ,1236    ,0       ,6       ,114   ],\
           [0       ,0       ,0       ,13      ,0       ,3364    ,0       ,0       ,0     ],\
           [0.214   ,0.006   ,0       ,164     ,0       ,1873    ,721     ,255     ,0     ],\
           [0       ,0       ,0       ,3061    ,0       ,0       ,0       ,0       ,0     ],\
           [0       ,0       ,0       ,93      ,0       ,2879    ,0       ,0       ,0     ],\
           [0       ,0       ,0       ,225     ,0       ,2657    ,0       ,0       ,0     ],\
           [0.107   ,0.003   ,0       ,155     ,0       ,1874    ,438     ,215     ,0     ],\
           [0       ,0       ,0       ,206     ,0       ,2514    ,0       ,0       ,0     ],\
           [1       ,0       ,0       ,402     ,5       ,0       ,1267    ,693     ,16    ],\
           [2       ,3       ,5       ,599     ,755     ,964     ,0       ,0       ,77    ],\
           [0       ,0       ,0       ,48      ,0       ,2216    ,0       ,0       ,0     ]]) 
  #
  # SAPRC_spec = ['C2H6','C3H8','C2H2','ALK3','ALK4','ALK5','ETHENE','C3H6','OLE1','OLE2','ARO1','ARO2','CCHO','HCHO','ACET','MEK','TERP','MEOH','PROD2']
  # CBMZ_spec = ['C2H5OH','CH3OH','HC5','ETH','TOL','OL2','KET','XYL','HCHO','OLT','ALD','ISO','OLI']
  #                                                                                                                        CBMZ_spec = ['C2H5OH','CH3OH','HC5','ETH','TOL','OL2','KET','XYL','HCHO','OLT','ALD','ISO','OLI']
  table = {'ethanol'                   :{'Molecular Mass': 46.1,'voc':table_2007[ 0],'SAPRC':{'SAPRC name':'E_ALK3'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':1,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_C2H5OH'   ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':1,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'butane'                    :{'Molecular Mass': 58.1,'voc':table_2007[ 1],'SAPRC':{'SAPRC name':'E_ALK3'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':1,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_HC5'      ,'HC5': 4,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'ethane'                    :{'Molecular Mass': 30.1,'voc':table_2007[ 2],'SAPRC':{'SAPRC name':'E_C2H6'   ,'C2H6':1,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_ETH'      ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':1,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'methanol'                  :{'Molecular Mass': 32.0,'voc':table_2007[ 3],'SAPRC':{'SAPRC name':'E_MEOH'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':1,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_CH3OH'    ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':1,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'propane'                   :{'Molecular Mass': 44.1,'voc':table_2007[ 4],'SAPRC':{'SAPRC name':'E_C3H8'   ,'C2H6':0,'C3H8':1,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_HC5'      ,'HC5': 3,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'toluene'                   :{'Molecular Mass': 92.1,'voc':table_2007[ 5],'SAPRC':{'SAPRC name':'E_ARO1'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':1    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_TOL'      ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':1,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'ethylene'                  :{'Molecular Mass': 28.1,'voc':table_2007[ 6],'SAPRC':{'SAPRC name':'E_ETHENE' ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':1,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_OL2'      ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':1,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'acetone'                   :{'Molecular Mass': 58.1,'voc':table_2007[ 7],'SAPRC':{'SAPRC name':'E_ACET'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':1,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_KET'      ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':1,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'pentane'                   :{'Molecular Mass': 72.2,'voc':table_2007[ 8],'SAPRC':{'SAPRC name':'E_ALK4'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':1,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_HC5'      ,'HC5': 5,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           '2-methylbutane'            :{'Molecular Mass': 72.2,'voc':table_2007[ 9],'SAPRC':{'SAPRC name':'E_ALK4'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':1,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_HC5'      ,'HC5': 5,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'm-xylene'                  :{'Molecular Mass':106.2,'voc':table_2007[10],'SAPRC':{'SAPRC name':'E_ARO2'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':1,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_XYL'      ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':1,'HCHO':0,'ALD':0,'OLI':0}},\
           'hexane'                    :{'Molecular Mass': 86.2,'voc':table_2007[11],'SAPRC':{'SAPRC name':'E_ALK4'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':1,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_HC5'      ,'HC5': 6,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'benzene'                   :{'Molecular Mass': 78.1,'voc':table_2007[12],'SAPRC':{'SAPRC name':'E_ARO1'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0.295,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_TOL'      ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':1,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'formaldehyde'              :{'Molecular Mass': 30.0,'voc':table_2007[13],'SAPRC':{'SAPRC name':'E_HCHO'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':1,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_HCHO'     ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':1,'ALD':0,'OLI':0}},\
           'trichloroethene'           :{'Molecular Mass':131.4,'voc':table_2007[14],'SAPRC':{'SAPRC name':'E_ALK3'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':1,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':''           ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           '2-methylpropane'           :{'Molecular Mass': 58.1,'voc':table_2007[15],'SAPRC':{'SAPRC name':'E_ALK3'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':1,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_HC5'      ,'HC5': 4,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           '2-butanone'                :{'Molecular Mass': 72.1,'voc':table_2007[16],'SAPRC':{'SAPRC name':'E_MEK'    ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':1,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_KET,E_HC5','HC5': 1,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':1,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'dichloromethane'           :{'Molecular Mass': 84.9,'voc':table_2007[17],'SAPRC':{'SAPRC name':'E_C2H6'   ,'C2H6':1,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':''           ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'decane'                    :{'Molecular Mass':142.3,'voc':table_2007[18],'SAPRC':{'SAPRC name':'E_ALK5'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':1,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_HC5'      ,'HC5':10,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'butyl acetate'             :{'Molecular Mass':116.2,'voc':table_2007[19],'SAPRC':{'SAPRC name':'E_ALK4'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':1,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':''           ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'propylene'                 :{'Molecular Mass': 42.1,'voc':table_2007[20],'SAPRC':{'SAPRC name':'E_C3H6'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':1,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_OLT,E_HC5','HC5': 1,'ISO':0,'OLT':1,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           '1,2,4-trimethylbenzene'    :{'Molecular Mass':120.2,'voc':table_2007[21],'SAPRC':{'SAPRC name':'E_ARO2'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':1,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_KET,E_HC5','HC5': 1,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':1,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'ethylbenzene'              :{'Molecular Mass':106.2,'voc':table_2007[22],'SAPRC':{'SAPRC name':'E_ARO1'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':1    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_TOL,E_HC5','HC5': 1,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':1,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           '2-propanol'                :{'Molecular Mass': 60.1,'voc':table_2007[23],'SAPRC':{'SAPRC name':'E_ALK4'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':1,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':''           ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'ethyl acetate'             :{'Molecular Mass': 88.1,'voc':table_2007[24],'SAPRC':{'SAPRC name':'E_C3H8'   ,'C2H6':0,'C3H8':1,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':''           ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'heptane'                   :{'Molecular Mass':100.2,'voc':table_2007[25],'SAPRC':{'SAPRC name':'E_ALK4'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':1,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_HC5'      ,'HC5': 7,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           '4-methyl-2-pentanone'      :{'Molecular Mass':100.2,'voc':table_2007[26],'SAPRC':{'SAPRC name':'E_PROD2'  ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':1}        ,'CBMZ':{'CBMZ name':'E_KET,E_HC5','HC5': 3,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':1,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'octane'                    :{'Molecular Mass':114.2,'voc':table_2007[27],'SAPRC':{'SAPRC name':'E_ALK5'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':1,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_HC5'      ,'HC5': 8,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'p-xylene'                  :{'Molecular Mass':106.2,'voc':table_2007[28],'SAPRC':{'SAPRC name':'E_ARO2'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':1,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_XYL'      ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':1,'HCHO':0,'ALD':0,'OLI':0}},\
           'o-xylene'                  :{'Molecular Mass':106.2,'voc':table_2007[29],'SAPRC':{'SAPRC name':'E_ARO2'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':1,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_XYL'      ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':1,'HCHO':0,'ALD':0,'OLI':0}},\
           'tetrachloroethene'         :{'Molecular Mass':165.8,'voc':table_2007[30],'SAPRC':{'SAPRC name':'E_C2H6'   ,'C2H6':1,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':''           ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'nonane'                    :{'Molecular Mass':128.3,'voc':table_2007[31],'SAPRC':{'SAPRC name':'E_ALK5'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':1,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_HC5'      ,'HC5': 9,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'undecane'                  :{'Molecular Mass':156.3,'voc':table_2007[32],'SAPRC':{'SAPRC name':'E_ALK5'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':1,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_HC5'      ,'HC5':11,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           '1-butanol'                 :{'Molecular Mass': 74.1,'voc':table_2007[33],'SAPRC':{'SAPRC name':'E_ALK5'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':1,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':''           ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           '2-methylpropene'           :{'Molecular Mass': 56.1,'voc':table_2007[34],'SAPRC':{'SAPRC name':'E_OLE2'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':1,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_OLT,E_HC5','HC5': 2,'ISO':0,'OLT':1,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'acetylene'                 :{'Molecular Mass': 26.0,'voc':table_2007[35],'SAPRC':{'SAPRC name':'E_C2H2'   ,'C2H6':0,'C3H8':0,'C2H2':1,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':''           ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'acetaldehyde'              :{'Molecular Mass': 44.1,'voc':table_2007[36],'SAPRC':{'SAPRC name':'E_CCHO'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':1,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_ALD'      ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':1,'OLI':0}},\
           '1-propanol'                :{'Molecular Mass': 60.1,'voc':table_2007[37],'SAPRC':{'SAPRC name':'E_ALK4'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':1,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':''           ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           '2-butoxyethanol'           :{'Molecular Mass':118.2,'voc':table_2007[38],'SAPRC':{'SAPRC name':'E_ALK5'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':1,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':''           ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           '2-methylpentane'           :{'Molecular Mass': 86.2,'voc':table_2007[39],'SAPRC':{'SAPRC name':'E_ALK4'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':1,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':1,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_HC5'      ,'HC5': 6,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'dipentene'                 :{'Molecular Mass':136.2,'voc':table_2007[40],'SAPRC':{'SAPRC name':'E_TERP'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':1,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_ISO'      ,'HC5': 0,'ISO':2,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           '1,3,5-trimethylbenzene'    :{'Molecular Mass':120.2,'voc':table_2007[41],'SAPRC':{'SAPRC name':'E_ARO2'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':1,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_XYL,E_HC5','HC5': 1,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':1,'HCHO':0,'ALD':0,'OLI':0}},\
           'methyl acetate'            :{'Molecular Mass': 74.1,'voc':table_2007[42],'SAPRC':{'SAPRC name':'E_C3H8'   ,'C2H6':0,'C3H8':1,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':''           ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           '1-methoxy 2-propanol'      :{'Molecular Mass': 90.0,'voc':table_2007[43],'SAPRC':{'SAPRC name':'E_ALK5'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':1,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':''           ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           'methylethylbenzene'        :{'Molecular Mass':120.2,'voc':table_2007[44],'SAPRC':{'SAPRC name':'E_ARO1'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':1    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_TOL,E_HC5','HC5': 2,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':1,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           '1,2,3-trimethylbenzene'    :{'Molecular Mass':120.2,'voc':table_2007[45],'SAPRC':{'SAPRC name':'E_ARO2'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':1,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_XYL,E_HC5','HC5': 1,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':1,'HCHO':0,'ALD':0,'OLI':0}},\
           '4-methyldecane'            :{'Molecular Mass':156.3,'voc':table_2007[46],'SAPRC':{'SAPRC name':'E_ALK5'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':1,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_HC5'      ,'HC5':11,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           '1,3-butadiene'             :{'Molecular Mass': 54.1,'voc':table_2007[47],'SAPRC':{'SAPRC name':'E_OLE2'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':1,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_OLT'      ,'HC5': 0,'ISO':0,'OLT':2,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           '3-methylpentane'           :{'Molecular Mass': 86.2,'voc':table_2007[48],'SAPRC':{'SAPRC name':'E_ALK4'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':1,'ALK5':0,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':'E_HC5'      ,'HC5': 6,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}},\
           '1-methoxy-2-propyl acetate':{'Molecular Mass':132.0,'voc':table_2007[49],'SAPRC':{'SAPRC name':'E_ALK5'   ,'C2H6':0,'C3H8':0,'C2H2':0,'ALK3':0,'ALK4':0,'ALK5':1,'ETHENE':0,'C3H6':0,'OLE2':0,'ARO1':0    ,'ARO2':0,'HCHO':0,'CCHO':0,'ACET':0,'MEK':0,'TERP':0,'MEOH':0,'PROD2':0}        ,'CBMZ':{'CBMZ name':''           ,'HC5': 0,'ISO':0,'OLT':0,'C2H5OH':0,'CH3OH':0,'ETH':0,'TOL':0,'OL2':0,'KET':0,'XYL':0,'HCHO':0,'ALD':0,'OLI':0}}}

  return  table, table_2007, total_voc_2007 

#____________________________________________________________________
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
#    MAIN PROGRAM BEGINS HERE
#
#
#    THIS PROGRAM CREATES WRFCHEMI input files for a WRF-Chem run
#    FOR SAPRC or CBMZ mechanism.
#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
data_description = ''
REAS_ANTH_VOC         = False
REAS_ANTH_Other       = False
HTAP_ANTH_VOC         = False 
HTAP_ANTH_Other       = False 
HTAP_SHIP_AIR_VOC     = False 
HTAP_SHIP_AIR_Other   = False
ECLIPSE_ANTH_VOC      = True
ECLIPSE_ANTH_Other    = True
ECLIPSEinUse          = True 
HUANG_BC_EMISSION     = False
ECLIPSE_RCP60_SHP     = False
POLMIP_DAILY_SOIL_NO  = True 
POLMIP_DAILY_VOLC_SO2 = True  
LANA_MONTHLY_DMS      = True
GFED_FIRE             = False

wrfdir   = '/directory/where/you/have/your/namelist/and/wrfinput/'

out_dir  = '/directory/where/wrfchemi/files/will/be/output/'

if not os.path.exists(out_dir):
  os.makedirs(out_dir)

### #--------------------------------------------------------------------
### #
### #          MIX Asian emission inventory 
### #          - /data/onishi/MIX_Asia/SAPRC99/<year>/MICS_Asia_SAPRC99_<spec>_<year>_0.25x0.25.nc
### #            <year>     : '2008' or '2010'
### #            <sec>      : ['POWER_PLANTS_NON-POINT','POWER_PLANTS_NON-POINT_JPN','INDUSTRY',
### #                          'ROAD_TRANSPORT','AVIATION','INTNNV','OTHER_TRANSPORT',          
### #                          'DOMESTIC','FUGITIVE_COAL','FUGITIVE_OIL','FUGITIVE_GAS',              
### #                          'EXTRACTION','SOLVENTS','FERTILIZER',                
### #                          'MANURE_MANAGEMENT','ENTERIC_FERMENTATION',      
### #                          'RICE_CULTIVATION','SOIL','SOIL_DIRECT','SOIL_INDIRECT',             
### #                          'WASTE','MISC']                    
### #
### #                 'POWER_PLANTS_NON-POINT'    : (ene )Power and heat plants as non-point sources except for Japan
### #                 'POWER_PLANTS_NON-POINT_JPN : (ene )Power and heat plants as non-point sources for Japan
### #                 'INDUSTRY'                  : (ind )Industry (emissions both from fuel combustion and industrial processes)
### #                 'ROAD_TRANSPORT'            : (tra )Road transport (cars, buses, trucks, motor cycles, and other on-road vehicles)
### #                 'AVIATION'                  : (shp )Domestic and international aviation (0-1km)
### #                 'INTNNV'                    : (shp )International navigation
### #                 'OTHER_TRANSPORT'           : (shp )Domestic navigation, railway, and other off-road transports
### #                 'DOMESTIC'                  : (dom )Residential, commerce and public services, agricultural equipment, fishing, and others.
### #                 'FUGITIVE_COAL'             : (flr?)Fugitive emissions from production, processing, and distribution of coal (For CH4)
### #                 'FUGITIVE_OIL'              : (flr?)Fugitive emissions from production, processing, and distribution of oil (For CH4)
### #                 'FUGITIVE_GAS'              : (flr?)Fugitive emissions from production, processing, and distribution of gas (For CH4)
### #                 'EXTRACTION'                : (flr )Extraction and handling of fossil fuels (For NMVOC)
### #                 'SOLVENTS'                  : (slv )Solvent use (including paint use)
### #                 'FERTILIZER'                : (??? )Fertilizer application
### #                 'MANURE_MANAGEMENT'         : (??? )Manure management of livestock
### #                 'ENTERIC_FERMENTATION'      : (??? )Enteric fermentation of livestock (For CH4)
### #                 'RICE_CULTIVATION'          : (??? )Rice cultivation (For CH4)
### #                 'SOIL'                      : (soi )Soil NOx emissions
### #                 'SOIL_DIRECT'               : (soi )Direct soil N2O emissions
### #                 'SOIL_INDIRECT'             : (soi )Indirect soil N2O emissions
### #                 'WASTE'                     : (wst )Waste treatment (both solid and water waste)
### #                 'MISC'                      : (??? )Human respiration and perspiration, latrines, and others (For NH3)
### #            resolution : 0.25[deg] x 0.25[deg]
### #            latitutde  : 
### #            unit       : [ton/month]
### #           
### #            Output :  
### #            "emis_dict": dictionary : emission on WRF grid
### #            keynames   : 'awb_VOC','dom_VOC','ene_VOC','ind_VOC','slv_VOC','tra_VOC','wst_VOC','all_VOC'
### #            keynames   : <sec>_VOC
### #                 <sec>   : 'agr','awb','dom','ene','flr','ind','tra','wst','all'
### #            dimension  : ['west_east','south_north','time'=12]
### #            unit       : [mol/sec/m2] or [ug/sec/m2] (for 'BC','OC','PM10','PM2.5')
### if REAS_ANTH_VOC:
###   data_description = data_description + '//REAS Anthropogenic NMVOC '
### try:
###   Grid025 = Grid025 or REAS_ANTH_VOC
### except:
###   Grid025 = REAS_ANTH_VOC 
### #
#--------------------------------------------------------------------
#
#          REAS species for NMVOC
#          - /data/onishi/REAS/<spec>/<year>/REASv2.1_NMV_20_<sec>_<year>_0.25x0.25
#            <year>     : '2008' 
#            <sec>      : ['POWER_PLANTS_NON-POINT','POWER_PLANTS_NON-POINT_JPN','INDUSTRY',
#                          'ROAD_TRANSPORT','AVIATION','INTNNV','OTHER_TRANSPORT',          
#                          'DOMESTIC','FUGITIVE_COAL','FUGITIVE_OIL','FUGITIVE_GAS',              
#                          'EXTRACTION','SOLVENTS','FERTILIZER',                
#                          'MANURE_MANAGEMENT','ENTERIC_FERMENTATION',      
#                          'RICE_CULTIVATION','SOIL','SOIL_DIRECT','SOIL_INDIRECT',             
#                          'WASTE','MISC']                    
#
#                 'POWER_PLANTS_NON-POINT'    : (ene )Power and heat plants as non-point sources except for Japan
#                 'POWER_PLANTS_NON-POINT_JPN : (ene )Power and heat plants as non-point sources for Japan
#                 'INDUSTRY'                  : (ind )Industry (emissions both from fuel combustion and industrial processes)
#                 'ROAD_TRANSPORT'            : (tra )Road transport (cars, buses, trucks, motor cycles, and other on-road vehicles)
#                 'AVIATION'                  : (shp )Domestic and international aviation (0-1km)
#                 'INTNNV'                    : (shp )International navigation
#                 'OTHER_TRANSPORT'           : (shp )Domestic navigation, railway, and other off-road transports
#                 'DOMESTIC'                  : (dom )Residential, commerce and public services, agricultural equipment, fishing, and others.
#                 'FUGITIVE_COAL'             : (flr?)Fugitive emissions from production, processing, and distribution of coal (For CH4)
#                 'FUGITIVE_OIL'              : (flr?)Fugitive emissions from production, processing, and distribution of oil (For CH4)
#                 'FUGITIVE_GAS'              : (flr?)Fugitive emissions from production, processing, and distribution of gas (For CH4)
#                 'EXTRACTION'                : (flr )Extraction and handling of fossil fuels (For NMVOC)
#                 'SOLVENTS'                  : (slv )Solvent use (including paint use)
#                 'FERTILIZER'                : (??? )Fertilizer application
#                 'MANURE_MANAGEMENT'         : (??? )Manure management of livestock
#                 'ENTERIC_FERMENTATION'      : (??? )Enteric fermentation of livestock (For CH4)
#                 'RICE_CULTIVATION'          : (??? )Rice cultivation (For CH4)
#                 'SOIL'                      : (soi )Soil NOx emissions
#                 'SOIL_DIRECT'               : (soi )Direct soil N2O emissions
#                 'SOIL_INDIRECT'             : (soi )Indirect soil N2O emissions
#                 'WASTE'                     : (wst )Waste treatment (both solid and water waste)
#                 'MISC'                      : (??? )Human respiration and perspiration, latrines, and others (For NH3)
#            resolution : 0.25[deg] x 0.25[deg]
#            latitutde  : 
#            unit       : [ton/month]
#           
#            Output :  
#            "emis_dict": dictionary : emission on WRF grid
#            keynames   : 'awb_VOC','dom_VOC','ene_VOC','ind_VOC','slv_VOC','tra_VOC','wst_VOC','all_VOC'
#            keynames   : <sec>_VOC
#                 <sec>   : 'agr','awb','dom','ene','flr','ind','tra','wst','all'
#            dimension  : ['west_east','south_north','time'=12]
#            unit       : [mol/sec/m2] or [ug/sec/m2] (for 'BC','OC','PM10','PM2.5')
if REAS_ANTH_VOC:
  data_description = data_description + '//REAS Anthropogenic NMVOC '
try:
  Grid025 = Grid025 or REAS_ANTH_VOC
except:
  Grid025 = REAS_ANTH_VOC 
#
#--------------------------------------------------------------------
#
#          REAS species except for NMVOC
#          - /data/onishi/REAS/<spec>/<year>/REASv2.1_<spec>_<sec>_<year>_0.25x0.25
#            <year>     : '2008' 
#            <spec>     : ['BC_','CH4','CO_','CO2','N2O','NH3','OC_','PM10_','PM2.5']
#            <sec>      : ['POWER_PLANTS_NON-POINT','POWER_PLANTS_NON-POINT_JPN','INDUSTRY',
#                          'ROAD_TRANSPORT','AVIATION','INTNNV','OTHER_TRANSPORT',          
#                          'DOMESTIC','FUGITIVE_COAL','FUGITIVE_OIL','FUGITIVE_GAS',              
#                          'EXTRACTION','SOLVENTS','FERTILIZER',                
#                          'MANURE_MANAGEMENT','ENTERIC_FERMENTATION',      
#                          'RICE_CULTIVATION','SOIL','SOIL_DIRECT','SOIL_INDIRECT',             
#                          'WASTE','MISC']                    
#
#                 'POWER_PLANTS_NON-POINT'    : (ene )Power and heat plants as non-point sources except for Japan
#                 'POWER_PLANTS_NON-POINT_JPN : (ene )Power and heat plants as non-point sources for Japan
#                 'INDUSTRY'                  : (ind )Industry (emissions both from fuel combustion and industrial processes)
#                 'ROAD_TRANSPORT'            : (tra )Road transport (cars, buses, trucks, motor cycles, and other on-road vehicles)
#                 'AVIATION'                  : (shp )Domestic and international aviation (0-1km)
#                 'INTNNV'                    : (shp )International navigation
#                 'OTHER_TRANSPORT'           : (shp )Domestic navigation, railway, and other off-road transports
#                 'DOMESTIC'                  : (dom )Residential, commerce and public services, agricultural equipment, fishing, and others.
#                 'FUGITIVE_COAL'             : (flr?)Fugitive emissions from production, processing, and distribution of coal (For CH4)
#                 'FUGITIVE_OIL'              : (flr?)Fugitive emissions from production, processing, and distribution of oil (For CH4)
#                 'FUGITIVE_GAS'              : (flr?)Fugitive emissions from production, processing, and distribution of gas (For CH4)
#                 'EXTRACTION'                : (flr )Extraction and handling of fossil fuels (For NMVOC)
#                 'SOLVENTS'                  : (slv )Solvent use (including paint use)
#                 'FERTILIZER'                : (??? )Fertilizer application
#                 'MANURE_MANAGEMENT'         : (??? )Manure management of livestock
#                 'ENTERIC_FERMENTATION'      : (??? )Enteric fermentation of livestock (For CH4)
#                 'RICE_CULTIVATION'          : (??? )Rice cultivation (For CH4)
#                 'SOIL'                      : (soi )Soil NOx emissions
#                 'SOIL_DIRECT'               : (soi )Direct soil N2O emissions
#                 'SOIL_INDIRECT'             : (soi )Indirect soil N2O emissions
#                 'WASTE'                     : (wst )Waste treatment (both solid and water waste)
#                 'MISC'                      : (??? )Human respiration and perspiration, latrines, and others (For NH3)
#            resolution : 0.25[deg] x 0.25[deg]
#            latitutde  : 
#            unit       : [ton/month]
#           
#            Output :  
#            "emis_dict": dictionary : emission on WRF grid
#            keynames   : 'awb_<spec>','dom_<spec>','ene_<spec>','ind_<spec>','slv_<spec>','tra_<spec>','wst_<spec>','all_<spec>'
#            keynames   : <sec>_<spec>
#                 <sec>   : 'agr','awb','dom','ene','flr','ind','tra','wst','all'
#                 <spec>  : ['BC','CH4','CO','CO2','NH3','OC','PM25']
#            dimension  : ['west_east','south_north','time'=12]
#            unit       : [mol/sec/m2] or [ug/sec/m2] (for 'BC','OC','PM10','PM2.5')
if REAS_ANTH_Other:
  data_description = data_description + '//REAS Anthropogenic gas and aerosols '
try:
  Grid025 = Grid025 or REAS_ANTH_Other
except:
  Grid025 = REAS_ANTH_Other
#
#--------------------------------------------------------------------
#
#          - /data/raut/EMISSIONS/HTAP_v2/<year>/edgar_HTAP_NMVOC_emi_<sec>_<year>_<month>.0.1x0.1.nc
#            <year>     : '2008' or '2010'
#            <sec>      : ['ENERGY','INDUSTRY','RESIDENTIAL','TRANSPORT','AGRICULTURE']
#           
#            Output :  
#            keynames   : 'awb_VOC','dom_VOC','ene_VOC','ind_VOC','slv_VOC','tra_VOC','wst_VOC','all_VOC'
#            dimension  : ['west_east','south_north','time'=12]
#            unit       : [mol/sec/m2]   
if HTAP_ANTH_VOC:
  data_description = data_description + '//HTAP Anthropogenic VOC '
try:
  Grid01 = Grid01 or HTAP_ANTH_VOC
except:
  Grid01 = HTAP_ANTH_VOC
#
#--------------------------------------------------------------------
#
#          - /data/raut/EMISSIONS/HTAP_v2/<year>/edgar_HTAP_<spec>_emi_<sec>_<year>_<month>.0.1x0.1.nc
#            <year>     : '2008' or '2010'
#            <spec>     : ['BC','CO','NH3','NOx','OC','PM10','PM25','SO2']
#            <sec>      : ['ENERGY','INDUSTRY','RESIDENTIAL','TRANSPORT','AGRICULTURE']
#            
#            keynames   : 'awb_<spec>','dom_<spec>','ene_<spec>','ind_<spec>','slv_<spec>','tra_<spec>','wst_<spec>','all_<spec>'
#            dimension  : ['west_east','south_north','time'=12]
#            unit       : [mol/m2/sec] or [ug/m2/sec]('BC','OC','PM25')
if HTAP_ANTH_Other:
  data_description = data_description + '//HTAP Anthropogenic gas and aerosols '
try:
  Grid01 = Grid01 or HTAP_ANTH_Other
except:
  Grid01 = HTAP_ANTH_Other
#
#--------------------------------------------------------------------
#
#          - /data/raut/EMISSIONS/HTAP_v2/<year>/edgar_HTAP_NMVOC_emi_<sec>_<year>.0.1x0.1.nc
#            <year>     : '2008' or '2010'
#            <sec>      : ['AIR','SHIPS']
#            
#            keynames   : 'awb_VOC','dom_VOC','ene_VOC','ind_VOC','slv_VOC','tra_VOC','wst_VOC','all_VOC'
#            dimension  : ['west_east','south_north']
#            unit       : [mol/m2/sec]   
if HTAP_SHIP_AIR_VOC:
  data_description = data_description + '//HTAP Anthropogenic VOC (air and ship) '
try:
  Grid01 = Grid01 or HTAP_SHIP_AIR_VOC
except:
  Grid01 = HTAP_SHIP_AIR_VOC
#
#--------------------------------------------------------------------
#
#          - /data/raut/EMISSIONS/HTAP_v2/<year>/edgar_HTAP_<spec>_emi_<sec>_<year>.0.1x0.1.nc
#            <year>     : '2008' or '2010'
#            <spec>     : ['BC','CO','NOx','OC','PM10','PM25','SO2']
#            <sec>      : ['AIR','SHIPS']
#            
#            keynames   : 'awb_VOC','dom_VOC','ene_VOC','ind_VOC','slv_VOC','tra_VOC','wst_VOC','all_VOC'
#            dimension  : ['west_east','south_north']
#            new unit   : [mol/sec/m2] or [ug/m2/sec]('BC','OC','PM25')
if HTAP_SHIP_AIR_Other:
  data_description = data_description + '//HTAP Anthropogenic gas and aerosols (air and ship) '
try:
  Grid01 = Grid01 or HTAP_SHIP_AIR_Other
except:
  Grid01 = HTAP_SHIP_AIR_Other
#
#--------------------------------------------------------------------
#
#          - /data/onishi/ECLIPSE_V6b/ETP_base_CLE_V6_VOC_<year>.nc
#            keynames   : 'awb_VOC','dom_VOC','ene_VOC','ind_VOC','slv_VOC','tra_VOC','wst_VOC','all_VOC'
#            dimension  : ['west_east','south_north']
#            new unit   : [mol/sec/m2]   
if ECLIPSE_ANTH_VOC:
  data_description = data_description + '//ECLIPSE v6b Anthropogenic VOC '
try:
  Grid05 = Grid05 or ECLIPSE_ANTH_VOC
except:
  Grid05 = ECLIPSE_ANTH_VOC
#
#-----------------------------------------------------------------------------------------------
#          - /data/onishi/ECLIPSE_V6b/ETP_base_CLE_V6_<spec>_<year>.nc
#            keynames   : <sec>_<spec>
#                 <sec>   : 'agr','awb','dom','ene','flr','ind','tra','wst','all'
#                 <spec>  : ['CO','CH4','BC','OM','SO2','NH3','PM25','NOx']
#            dimension  : ['west_east','south_north']
#            new unit   : [mol/sec/m2] or [ug/sec/m2]('OM','BC','PM25')
if ECLIPSE_ANTH_Other:
  data_description = data_description + '//ECLIPSE v6b Anthropogenic gas and aerosols '
try:
  Grid05 = Grid05 or ECLIPSE_ANTH_Other
except:
  Grid05 = ECLIPSE_ANTH_Other
#
#-----------------------------------------------------------------------------------------------
#          - /data/onishi/BC_RUS/Huang/RUS_BC_2010_Huang.nc
#            keynames   : <sec>_BC
#                 <sec>   : 'dom','ene','flr','ind','tra','all'
#            dimension  : ['west_east','south_north']
#            new unit   : [ug/m2/sec]  
if HUANG_BC_EMISSION:
  data_description = data_description + '//HUAGN BC EMISSION '
try:
  Grid01 = Grid01 or HUANG_BC_EMISSION 
except:
  Grid01 = HUANG_BC_EMISSION
#
#-----------------------------------------------------------------------------------------------
#          - /data/quennehen/ECLIPSE/RCP60_AIR,SHP_2005,2010/
#              1. IPCC_emissions_RCP60_NMVOC_ships_2005_0.5x0.5_v1_01_03_2010.nc
#              2. IPCC_emissions_RCP60_NMVOC_ships_2010_0.5x0.5_v1_01_03_2010.nc
#              3. IPCC_emissions_RCP60_<spec>_ships_2005_0.5x0.5_v1_01_03_2010.nc
#              4. IPCC_emissions_RCP60_<spec>_ships_2010_0.5x0.5_v1_01_03_2010.nc
#            input unit : [kg/m2/sec]
#            keynames   : 'shp_NMVOC','shp_CO','shp_CH4','shp_BC','shp_OC','shp_SO2','shp_NH3','shp_NO'
#            dimension  : ['west_east','south_north',12]
#            new unit   : [mol/sec/m2] or [ug/m2/sec] (BC & OC)   
if ECLIPSE_RCP60_SHP:
  data_description = data_description + '//ECLIPSE RCP60 SHIP '
try:
  Grid05 = Grid05 or ECLIPSE_RCP60_SHP
except:
  Grid05 = ECLIPSE_RCP60_SHP
#
#-----------------------------------------------------------------------------------------------
#          - /data/quennehen/EMISSIONS/POLMIP/emissions.NO.surface.1x1.nc'
#            input unit : [molecules/cm2/sec]
#            keynames   : 'soil_NO'
#            dimension  : ['west_east','south_north',367]
#            unit   : [mol/km2/day]   
#            new unit   : [mol/m2/sec]   
if POLMIP_DAILY_SOIL_NO:
  data_description = data_description + '//POLMIP DAILY SOIL NO '
try:
  Grid10 = Grid10 or POLMIP_DAILY_SOIL_NO
except:
  Grid10 = POLMIP_DAILY_SOIL_NO
#
#-----------------------------------------------------------------------------------------------
#          - /data/quennehen/EMISSIONS/POLMIP/emissions.SO2.surface.1x1.nc'
#            input unit : [molecules/cm2/sec]
#            keynames   : 'vol_SO2'
#            dimension  : ['west_east','south_north',367]
#            new unit   : [mol/sec/m2]   
if POLMIP_DAILY_VOLC_SO2:
  data_description = data_description + '//POLMIP DAILY VOLCANIC SO2 '
try:
  Grid10 = Grid10 or POLMIP_DAILY_VOLC_SO2
except:
  Grid10 = POLMIP_DAILY_VOLC_SO2
#
#-----------------------------------------------------------------------------------------------
#          - /data/marelle/EMISSIONS/DMS_LANA/DMSclim_<month>.csv
#            keynames   : 'DMS_OC'
#            dimension  : ['west_east','south_north',12]
#            unit       : [mol/m3]   
if LANA_MONTHLY_DMS:
  data_description = data_description + '//Dimethyl Sulfide from LANA (monthly resolution) '
try:
  Grid10 = Grid10 or LANA_MONTHLY_DMS     
except:
  Grid10 = LANA_MONTHLY_DMS
#
#-----------------------------------------------------------------------------------------------
#          - /data/onishi/GFEDv4/fire_emissions_v4_R1/data/GFED4.1s_<year>.hdf5
#            keynames   : '<spec>_GFED'
#            dimension  : ['west_east','south_north',12]
#            unit       : [ug/month/m2] or [mol/month/m2]   
if GFED_FIRE:       
  data_description = data_description + '//GFED fire emissions (ver. 4.1) '
try:
  Grid025 = Grid025 or GFED_FIRE     
except:
  Grid025 = GFED_FIRE
#
#-----------------------------------------------------------------------------------------------

namelist = wrfdir+'namelist.input'

outdir   = wrfdir.split('/')
if outdir[-1] == '':
  outdir = './'+outdir[-2]+'/'
else:
  outdir = './'+outdir[-1]+'/'

start_dt_namelist, end_dt_namelist, max_dom = get_Start_datetime_and_End_datetime(namelist)
#start_dt_namelist = datetime(2017,7,19,0)
#end_dt_namelist = datetime(2014,1,7,0)
print(start_dt_namelist)
print(end_dt_namelist)


deltaday = 1 

#nday = (end_dt_namelist-start_dt_namelist).days + 1
nday = (end_dt_namelist-start_dt_namelist).days
print(nday/deltaday)

dhour = 1

for iiday in np.arange(nday/deltaday+1):
  if iiday % size != rank:
    continue

  start_dt = start_dt_namelist + timedelta(days=int(deltaday*iiday))
  end_dt   = start_dt_namelist + timedelta(days=int(deltaday*iiday+deltaday))

  print('start_dt',start_dt)
  print('end_dt' ,end_dt)

  for dd in np.arange(max_dom)+1:
    if rank == 0:
      print(('working on domain '+str(dd)))
    
    if rank == 0:
      print('------------------------------------')
      print('')
      print(datetime.now(),' SECTION : at the beginning of loop ')
      print('')
      print('------------------------------------')
  
    if end_dt >= end_dt_namelist:
      end_dt = end_dt_namelist
    #
    year_str  = str(start_dt.year)
    if start_dt.year == 2013:
      year_str  = '2014'
    #
    if start_dt.year == 2018:
      year_str  = '2020'
    #
    if rank == 0:
      print(datetime.now(),'start_dt = ',start_dt)
      print(datetime.now(),'end_dt   = ',end_dt)
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #
    #    WRFCHEMI input variable names
    #   
    #    -----------------------------------------------------
    #    |   CBMZ       |   species   |   Description   |  unit
    #    --------------------------------------------------
    #    |   E_CO       |    CO       |                 | [mol/km2/h]                     
    #    |   E_CH4      |    CH4      |                 | [mol/km2/h]                
    #    |   E_ECJ      |    BC       |                 | [ug/m2/sec]*               
    #    |   E_ORGJ     |    OM, OC   |                 | [ug/m2/sec]*               
    #    |   E_SO2      |    SO2      |                 | [mol/km2/h]                
    #    |   E_NH3      |    NH3      |                 | [mol/km2/h]                
    #    |   E_PM25J    |    PM25     |                 | [ug/m2/sec]*                
    #    |   E_NO       |    NO       |                 | [mol/km2/h]                
    #    |   E_NO2      |    NO2      |                 | [mol/km2/h]                
    #    |   E_C2H5OH   |    C2H5OH   |                 | [mol/km2/h]                
    #    |   E_CH3OH    |    CH3OH    |                 | [mol/km2/h]                
    #    |   E_HC5      |    HC5      |                 | [mol/km2/h]                
    #    |   E_ETH      |    ETH      |                 | [mol/km2/h]                
    #    |   E_TOL      |    TOL      |                 | [mol/km2/h]                
    #    |   E_OL2      |    OL2      |                 | [mol/km2/h]                
    #    |   E_KET      |    KET      |                 | [mol/km2/h]                
    #    |   E_XYL      |    XYL      |                 | [mol/km2/h]                
    #    |   E_HCHO     |    HCHO     |                 | [mol/km2/h]                
    #    |   E_OLT      |    OLT      |                 | [mol/km2/h]                
    #    |   E_ALD      |    ALD      |                 | [mol/km2/h]                
    #    |   E_ISO      |    ISO      |                 | [mol/km2/h]                
    #    ------------------------------------------------------------
    #
    #    ----------------------------------------------------------------------------------------------------
    #    |   SAPRC           | |   species   |   Description                        |  unit       | REAS    | 
    #    ---------------------------------------------------------------------------------------------------- 
    #    |   E_CO            |o|    CO       | Carbon Monoxide                      | [mol/km2/h] |  CO     |                     
    #    |   E_CH4           |o|    CH4      | Methane                              | [mol/km2/h] |  CH4    |                     
    #    |   E_NH3           |o|    NH3      | Ammoniac                             | [mol/km2/h] |  NH3    |                
    #    |   E_SO2           |o|    SO2      | Sulfur Dioxide                       | [mol/km2/h] |  SO2    |                
    #    |   E_NO            |o|    NO       | Nitrogen Monoxide, Nitric oxide      | [mol/km2/h] |         |                
    #    |   E_NO2           |o|    NO2      | Nitrogen dioxide                     | [mol/km2/h] |  NOX    | 
    #    |                   | |             |                                      |             |         | 
    #    |___VOC____________________________________________________________________________________________| 
    #    |____Alkanes____
    #    |   E_C2H6          |o|    C2H6     | Ethane (VOC : ~ALK1)                 | [mol/km2/h] |  NMV 01 |                
    #    |   E_C3H8          |o|    C3H8     | Propane (VOC : ~ALK2)                | [mol/km2/h] |  NMV 02 |                 
    #    |   E_ALK3          |o|    ALK3     | Alkanes (2.5e3 < kOH < 5e3 ppm/min)  | [mol/km2/h] |  Ethanol Butane Trichloroethene 
    #    |   E_ALK4          |o|    ALK4     | Alkanes (5e3 < kOH < 1e4 ppm/min)    | [mol/km2/h] |  Pentane 2-methylbutane hexane butyl acetate 2-propanol 1-propanol 2-methylpentane 3-methylpentane
    #    |   E_ALK5          |o|    ALK5     | Alkanes (1e4 < kOH ppm/min)          | [mol/km2/h] |  Decane Octane Nonane Undecane 1-butanol          
    #    |____Alkenes____
    #    |   E_ETHENE        |o|    ETHENE   | Ethene (Ethylene) : C2H4             | [mol/km2/h] |  Ethylene 06             
    #    |   E_C3H6          |o|    C3H6     | Propene : Alkene                     | [mol/km2/h] |  Propene 07       |                
    #    |   E_OLE1          |x|             | Alkenes (kOH < 7e4 ppm/min)          | [mol/km2/h] |         |                
    #    |                   | |             |     Except for Ethene (Ethylene)     |             |         |                
    #    |   E_OLE2          |o|    OLE2     | Alkenes (kOH > 7e4 ppm/min)          | [mol/km2/h] |         |                
    #    |____Alkynes____
    #    |   E_C2H2          |o|    C2H2     | Acetylene (Alkyne)                   | [mol/km2/h] |  NMV 10 |                
    #    |____Aromatics___
    #    |   E_ARO1          |o|    ARO1     | Aromatics (kOH < 2e4 ppm/min)        | [mol/km2/h] |         |                
    #    |   E_ARO2          |o|    ARO2     | Aromatics (kOH > 2e4 ppm/min)        | [mol/km2/h] |         |                
    #    |____Terpenes ((C5H8)n)____
    #    |   E_TERP          |o|    TERP     | Terpenes                             | [mol/km2/h] |         |                
    #    |   E_PHEN          |x|             | Phenol: C5H8                         |             |         |
    #    |   E_SESQ          |x|             | Sesquiterpen (C15H24)                |             |         |
    #    |____Aldeydes (R-CHO)______
    #    |   E_HCHO          |o|    HCHO     | Formaldehyde                         | [mol/km2/h] |  Formaldehyde 15       |                
    #    |   E_CCHO          |o|    CCHO     | Acetaldehyde                         | [mol/km2/h] |  Other Aldehyde 16       |                
    #    |   E_RCHO          |x|             | Lumped C3+ Aldehyde                  |             |         |
    #    |   E_BALD          |x|             | Aromatic Aldehydes (eg. Benzaldehyde)|             |         |
    #    |____Ketones (RC(=O)R')____
    #    |   E_ACET          |o|    ACET     | Acetone (simplest Ketone)            | [mol/km2/h] |         |                
    #    |   E_MEK           |o|    MEK      | Ketones (non-aldehyde)               | [mol/km2/h] |         |                
    #    |                   | |             |   5e-13 < OH < 5e-12 cm3/molec2/sec  |             |         | 
    #    |   E_PROD2         |o|    PROD2    | Ketones (non-aldehyde)               | [mol/km2/h] |         |                
    #    |                   | |             |   5e-12 < OH  cm3/molec2/sec         |             |         | 
    #    |   E_MVK           |x|             | Methyl Vinyl Ketone                  |             |         |
    #    |                   | |             |   CH3C(O)CH=CH2                      |             |         |
    #    |   E_MGLY          |x|             | Methyl Glyoxal (aldehyde+ketone)     |             |         |
    #    |                   | |             |   CH3C(O)CHO                         |             |         |
    #    |   E_ISOPROD       |x|             | Lumped other isoprene products : represents reactive         |
    #    |                   | |             |   products other than methacrolein and MVK, and also         |
    #    |                   | |             |   to present other unsaturated ketones or aldehydes.         |
    #    |____Alcools (R-OH)____
    #    |   E_MEOH          |o|    MEOH     | Methanol: CH3OH                      | [mol/km2/h] |         |                
    #    |   E_DMS_OC        |o|    DMS_OC   | Dimethyl Sulfide (Oceanic Values)    | [mol/m3]    |         |             
    #    |   E_GLY           |x|             | Glyoxal (Dialdehyde) OCHCHO          |             |         |
    #    |   E_CRES          |x|             | Cresols (hydroxytoluene, methylphenols)            |         |
    #    |____Acide____
    #    |   E_HCOOH         |x|             | Formic Acid (HCOOH)                  |             |         |
    #    |   E_CCO_OH        |x|             | Acetic Acide                         |             |         |
    #    |                   | |             |    Also used for Peroxyacetic acid   |             |         |
    #    |   E_RCO_OH        |x|             | Higher organic acids and peroxy acids                        |
    #    |                   | |             |   (Mechanism based on propionic acide)                       |
    #    |____Aerosols____
    #    |   E_ORGJ          |o|    ORGJ     | Organic Carbon (*)                   | [ug/m2/sec] |         |                
    #    |   E_ORGJ          |o|    OC+OM(?) | Organic Carbon, Organic Matter (*)   | [ug/m2/sec] |         |                
    #    |   E_ECJ           |o|    ECJ      | Elemental Carbon (Black Carbon)(*)   | [ug/m2/sec] |         |                
    #    |   E_PM25I,E_PM25J |x|             | PM 2.5 (*)                                                   |
    #    |   E_ECI           |x|             | Elemental Carbon (nuclei/Aitken mode) (*)                    |
    #    |   E_so4i,E_so4j   |x|             | SO4^2- (*)                           |             |         |
    #    |   E_no3i,E_no3j   |x|             | NO3^1- (*)                           |             |         |
    #    |   E_orgi_a, E_orgj_a  |x|         | Organic Carbon Aerosol (*)           |             |         |
    #    |   E_orgi_bb,E_orgj_bb |x|         | Organic Carbon Biomass Burning (*)   |             |         | 
    #    ---------------------------------------------------------------------------------------------
    #    (*)   suffix J : accumulation mode           (e.g. smoke particles) 
    #          suffix I : nuclei (and/or Aitken) mode (e.g. nano/soot particles)             
    #
    #--------------------------------------------------------------------
    #
    #    INPUT FILES:
    #       
    #       * WRF GRID 
    #          - wrfinput_d01
    #
    #       ----------------------------------------------------------------------------
    #
    #       * HTAP_v2 Anthropogenic Emissions (Except for AIR and SHIPS)
    #          - /data/raut/EMISSIONS/HTAP_v2/<year>/edgar_HTAP_<spec>_emi_<sec>_<year>.0.1x0.1.nc
    #            <spec> : 
    #
    #       ----------------------------------------------------------------------------
    #
    #	* ECLIPSE Anthropogenic Emissions (VOC) 
    #          - /data/quennehen/ECLIPSE/Anth/CP_WEO_2011_UPD_VOC_<year>.nc
    #            variable  : 'emis_agr','emis_awb','emis_dom','emis_ene','emis_flr',
    #                        'emis_ind','emis_slv','emis_tra','emis_wst','emis_all'
    #            resolution: 0.5 deg x 0.5 deg
    #            dimension : [lat,lon] (360,720)
    #            unit      : [kt/year]
    #
    #            Sectors  : awb    : Agriculture (waste burning on field)
    #                       dom    : Residential and commercial
    #                       ene    : Power plants, energy conversion, extraction 
    #                       ind    : Industry (combustion and processing)
    #                          * ind1 : Industrial (Stationary Combustion)
    #                          * ind2 : Production Process 
    #                       slv    : Solvent Use
    #                       flr    : Extraction and distribution of fossil
    #                       tra    : Road transport
    #                       shp    : Other transport
    #                       wst    : Waste treatment and disposal
    #                   
    #       ----------------------------------------------------------------------------
    #
    #       * REAS NMVOC Emissions
    #          - /data/onishi/REAS/NMV/2008/<VOC>/REASv2.1_NMV_20_<sec>_2008_0.25x0.25
    #            <VOC> : Ethane
    #                    Propane
    #                    Butanes
    #                    Pentanes
    #                    OtherAlkanes
    #                    Ethylene
    #                    TerminalAlkenes
    #                    InternalAlkenes
    #                    Acetylene
    #                    Benzene
    #                    Toluene
    #                    Xylenes
    #                    OtherAromatics
    #                    Formaldehyde
    #                    OtherAldehyde
    #                    Ketones
    #                    Halocarbons
    #                    Others
    #                    Total
    #
    #            <sec> : AVIATION                      : Domestic and international aviation (0-1km)
    #                    DOMESTIC                  dom : Residential,commerce and public services, agricultural equipemnt, fishing and others.
    #                    EXTRACTION                ene : Extraction and handling of fossil fuels (for NMVOC)
    #                    INDUSTRY                  ind : Industry (emissions both from fuel combustion and industrial processes)
    #                    INTNNV                    shp : International navigation
    #                    OTHER_TRANSPORT           shp : Domestic navigation, railway, and other off-road ransports
    #                    FERTILIZER                awb : Fertilizer application
    #                    MANURE_MANAGEMENT         awb : Manure management of livestock
    #                    POWER_PLANT_NON-POINT     ene : Power and heat plants as non-point sources except for Japan
    #                    POWER_PLANT_NON-POINT_JPN ene : Power and heat plants as non-point sources for Japan
    #                    ROAD_TRANSPORT            tra : Road transport (cars, buses, trucks, motor cycles and other on-road vehicles)
    #                    SOLVENTS                  slv : Solvent use (including paint use)
    #                    WASTE                     wst : Waste treatment (both solid and water waste)
    #                    __CH4__
    #                    ENTERIC_FERMENTATION      awb : Enteric fermentation of livestock (For CH4)
    #                    FUGITIVE_COAL             ind : Fugitive emissions from production, processing and distribution of coal (For CH4)
    #                    FUGITIVE_OIL              ind : Fugitive emissions from production, processing and distribution of oil (For CH4)
    #                    FUGITIVE_GAS              ind : Fugitive emissions from production, processing and distribution of gas (For CH4)
    #                    RICE_CULTIVATION          awb : Rice cultivation (For CH4)
    #                    __NOx__
    #                    SOIL                      soil: Soil NOx emissions
    #                    SOIL_DIRECT               soil: Direct soil NOx emissions
    #                    SOIL_INDIRECT             soil: Indirect soil NOx emissions
    #                    __NH3__
    #                    MISC                          : Human respiration and perspiration, latrines and others (For NH3)
    #       ----------------------------------------------------------------------------
    #
    #       * RCP60 NMVOC emissions from ship transport
    #          - /data/quennehen/ECLIPSE/RCP60_AIR,SHP_2005,2010/
    #              1. IPCC_emissions_RCP60_NMVOC_ships_2005_0.5x0.5_v1_01_03_2010.nc
    #              2. IPCC_emissions_RCP60_NMVOC_ships_2010_0.5x0.5_v1_01_03_2010.nc
    #            
    #            variable  : 'emiss_shp'
    #            resolution: 0.5deg x 0.5 deg
    #            dimension : [time,lat,lon] (12,360,720)
    #            unit      : [kg/m2/sec]
    #
    #       ----------------------------------------------------------------------------
    #
    #       * RCP60 other emissions from ship transport
    #          - /data/quennehen/ECLIPSE/RCP60_AIR,SHP_2005,2010/
    #              1. IPCC_emissions_RCP60_<spec2>_ships_2005_0.5x0.5_v1_01_03_2010.nc
    #              2. IPCC_emissions_RCP60_<spec2>_ships_2010_0.5x0.5_v1_01_03_2010.nc
    #
    #            <spec2>   : 'CO','CH4','BC','OC','SO2','NH3','NO'
    #            variable  : 'emiss_shp'
    #            resolution: 0.5 deg x 0.5 deg
    #            dimension : [time,lat,lon] (12,360,720)
    #            unit      : [kg/m2/sec]
    #
    #       ----------------------------------------------------------------------------
    #     
    #       * POLMIP Daily soil NOx surface emissions
    #          - /data/quennehen/EMISSIONS/POLMIP/emissions.NO.surface.1x1.nc'
    #
    #            variable  : 'soil'
    #            resolution: 1.0 deg x 1.0 deg
    #            dimension : [time,lat,lon] (367,180,360)
    #            unit      : [molecules/cm2/sec]
    #
    #       -----------------------------------------------------------------------------
    #
    #       * POLMIP Daily Volcanic SO2 emissions
    #         - /data/quennehen/EMISSIONS/POLMIP/emissions.SO2.surface.1x1.nc'
    #
    #            variable  : 'volcano'
    #            resolution: 1.0 deg x 1.0 deg
    #            dimension : [time,lat,lon] (367,180,360)
    #            unit      : [molecules/cm2/sec]
    #           
    #       -----------------------------------------------------------------------------
    #
    #       * Monthly DMS concentration : oceanic values from Lana et al
    #         - /data/marelle/EMISSIONS/DMS_LANA/DMSclim_<month>.csv
    # 
    #            <month>   : 'JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'
    #            variable  : 'DMS_OC'
    #            resolution: 1.0 deg x 1.0 deg
    #            dimension : [time,lat,lon] (12,180,360)
    #            unit      : [mol/cm3]
    #            
    #=========================================================================================
    
    
    #===========================================================================
    #===========================================================================
    #---------------------------------------------------------------------------
    #  
    #   SECTION: GRID GENERATION
    #
    #   Input  : wrfinput_d01
    #
    #   Output : WRF grid (from wrfinput_d01):
    #            XLON            : Longitude of WRF grid (cell center)
    #            XLAT            : Latitude of WRF grid (cell center)
    #            XLONa           : Longitude of WRF grid (cell corners)
    #            XLATa           : Latitude of WRF grid (cell centers)
    #
    #            .................................................................
    #            Emission grid : longitude : -180 to 360 with 0.5 deg resolution
    #                            latitude  :  -90 to  90 with 0.5 deg resolution
    #            Surface             : Surface area (unit: m2) of Emission grid
    #            lon_glob05          : 1D longitude points : (1080,)  : -179.75:359.75:0.5
    #            lat_glob05          : 1D latitude points  : ( 360,)  :  -89.75: 89.75:0.5
    #            lon_bound_glob05    : 1D longitude points : (1081,)  : -180.00:360.00:0.5
    #            lat_bound_glob05    : 1D latitude points  : ( 361,)  :  -90.00: 90.00:0.5
    #            lon_glob05_2D       : 2D longitude points :(1080,360): np.repeat(lon_glob, 360).reshape(1080,360)
    #            lat_glob05_2D       : 2D latitude points  :(1080,360): np.tile(  lat_glob,1080).reshape(1080,360)
    #            lon_bound_glob05_2D : 2D longitude points :(1081,361): np.repeat(lon_glob, 361).reshape(1081,361)
    #            lat_bound_glob05_2D : 2D latitude points  :(1081,361): np.tile(  lat_glob,1081).reshape(1081,361)
    #
    #            .................................................................
    #            Emission grid : longitude : -180 to 360 with 1.0 deg resolution
    #                            latitude  :  -90 to  90 with 1.0 deg resolution
    #            Surface             : Surface area (unit: m2) of Emission grid
    #            lon_glob10          : 1D longitude points : (540,)  : -179.05:359.05:1.0
    #            lat_glob10          : 1D latitude points  : (180,)  :  -89.05: 89.05:1.0
    #            lon_bound_glob10    : 1D longitude points : (541,)  : -180.00:360.00:1.0
    #            lat_bound_glob10    : 1D latitude points  : (181,)  :  -90.00: 90.00:1.0
    #            lon_glob10_2D       : 2D longitude points :(540,180): np.repeat(lon_glob,180).reshape(540,180)
    #            lat_glob10_2d       : 2D latitude points  :(540,180): np.tile(  lat_glob,540).reshape(540,180)
    #            lon_bound_glob10_2D : 2D longitude points :(541,181): np.repeat(lon_glob,181).reshape(541,181)
    #            lat_bound_glob10_2D : 2D latitude points  :(541,181): np.tile(  lat_glob,541).reshape(541,181)
    #
    #            glob_att        : Global attribute from wrfinput_d01
    #
    #-----------------------------------------------------------------------------------------
    if rank == 0:
      print('------------------------------------')
      print('')
      print(datetime.now(),' SECTION : GRID GENERATION ')
      print('')
      print('------------------------------------')
    
    fname    = wrfdir+'wrfinput_d'+str(dd).zfill(2)
    
    #========================================================
    #
    #   Emission Grid (cell corner, 1D)
    #   "lat_bound_glob01" :1801 points : -90: 90:0.1 deg.
    #   "lon_bound_glob01" :5401 points :-180:360:0.1 deg. 
    #   "lat_bound_glob025": 721 points : -90: 90:0.25deg.
    #   "lon_bound_glob025":2161 points :-180:360:0.25deg. 
    #   "lat_bound_glob05" : 361 points : -90: 90:0.5 deg.
    #   "lon_bound_glob05" :1081 points :-180:360:0.5 deg. 
    #   "lat_bound_glob10" : 181 points : -90: 90:1.0 deg.
    #   "lon_bound_glob10" : 541 points :-180:360:1.0 deg. 
    #
    #--------------------------------------------------------
    
    if Grid01:
      lat_bound_glob01 = np.linspace( -90., 90., 1801)
      lon_bound_glob01 = np.linspace(-180.,360., 5401)
    if Grid025:
      lat_bound_glob025= np.linspace( -90., 90., 721)
      lon_bound_glob025= np.linspace(-180.,360.,2161)
    if Grid05:
      lat_bound_glob05 = np.linspace( -90., 90., 361)
      lon_bound_glob05 = np.linspace(-180.,360.,1081)
    if Grid10:
      lat_bound_glob10 = np.linspace( -90., 90., 181)
      lon_bound_glob10 = np.linspace(-180.,360., 541)
    
    #========================================================
    #
    #   Emission Grid (Cell Center, 1D)
    #   "lat_glob01" : 180 points :  -89.95: 89.95:0.1
    #   "lon_glob01" : 540 points : -179.95:359.95:0.1
    #   "lat_glob025": 720 points :  -89.875: 89.875:0.25
    #   "lon_glob025":2160 points : -179.875:359.875:0.25
    #   "lat_glob05" : 360 points :  -89.75: 89.75:0.5
    #   "lon_glob05" :1080 points : -179.75:359.75:0.5
    #   "lat_glob10" : 180 points :  -89.50: 89.50:1.0
    #   "lon_glob10" : 540 points : -179.50:359.50:1.0
    #
    #--------------------------------------------------------
    #
    if Grid01:
      lat_glob01 = np.linspace( -89.95, 89.95,1800)
      lon_glob01 = np.linspace(-179.95,359.95,5400)
    if Grid025:
      lat_glob025 = np.linspace( -89.875, 89.875, 720)
      lon_glob025 = np.linspace(-179.875,359.875,2160)
    if Grid05:
      lat_glob05 = np.linspace( -89.75, 89.75, 360)
      lon_glob05 = np.linspace(-179.75,359.75,1080)
    if Grid10:
      lat_glob10 = np.linspace( -89.50, 89.50, 180)
      lon_glob10 = np.linspace(-179.50,359.50, 540)
    #
    #========================================================
    #
    #   Emission Grid (Cell Center, 2D) 
    #   "lat_glob01_2D" : (5400,1800)~"lat_glob01"
    #   "lon_glob01_2D" : (5400,1800)~"lon_glob01"
    #   "lat_glob025_2D" : (2160,720)~"lat_glob025"
    #   "lon_glob025_2D" : (2160,720)~"lon_glob025"
    #   "lat_glob05_2D" : (1080,360)~"lat_glob05"
    #   "lon_glob05_2D" : (1080,360)~"lon_glob05"
    #   "lat_glob10_2D" : ( 540,180)~"lat_glob10"
    #   "lon_glob10_2D" : ( 540,180)~"lon_glob10"
    #
    #--------------------------------------------------------
    #
    if Grid01:
      lat_glob01_2D = np.tile(  lat_glob01,5400).reshape(5400,1800)
      lon_glob01_2D = np.repeat(lon_glob01,1800).reshape(5400,1800)
    if Grid025:
      lat_glob025_2D = np.tile(  lat_glob025,2160).reshape(2160,720)
      lon_glob025_2D = np.repeat(lon_glob025, 720).reshape(2160,720)
    if Grid05:
      lat_glob05_2D = np.tile(  lat_glob05,1080).reshape(1080,360)
      lon_glob05_2D = np.repeat(lon_glob05, 360).reshape(1080,360)
    if Grid10:
      lat_glob10_2D = np.tile(  lat_glob10, 540).reshape( 540,180)
      lon_glob10_2D = np.repeat(lon_glob10, 180).reshape( 540,180)
    #
    #========================================================
    #
    #   Emission Grid (Cell Corner, 2D) 
    #   "lat_bound_glob01_2D" : (5401,1801)~"lat_bound_glob01"
    #   "lon_bound_glob01_2D" : (5401,1801)~"lon_bound_glob01"
    #   "lat_bound_glob025_2D" : (2161, 721)~"lat_bound_glob025"
    #   "lon_bound_glob025_2D" : (2161, 721)~"lon_bound_glob025"
    #   "lat_bound_glob05_2D" : (1081, 361)~"lat_bound_glob05"
    #   "lon_bound_glob05_2D" : (1081, 361)~"lon_bound_glob05"
    #   "lat_bound_glob10_2D" : ( 541, 181)~"lat_bound_glob10"
    #   "lon_bound_glob10_2D" : ( 541, 181)~"lon_bound_glob10"
    #
    #--------------------------------------------------------
    #
    if Grid01:
      lat_bound_glob01_2D = np.tile(  lat_bound_glob01,5401).reshape(5401,1801)
      lon_bound_glob01_2D = np.repeat(lon_bound_glob01,1801).reshape(5401,1801)
    if Grid025:
      lat_bound_glob025_2D = np.tile(  lat_bound_glob025,2161).reshape(2161,721)
      lon_bound_glob025_2D = np.repeat(lon_bound_glob025, 721).reshape(2161,721)
    if Grid05:
      lat_bound_glob05_2D = np.tile(  lat_bound_glob05,1081).reshape(1081,361)
      lon_bound_glob05_2D = np.repeat(lon_bound_glob05, 361).reshape(1081,361)
    if Grid10:
      lat_bound_glob10_2D = np.tile(  lat_bound_glob10, 541).reshape( 541,181)
      lon_bound_glob10_2D = np.repeat(lon_bound_glob10, 181).reshape( 541,181)
    #
    #========================================================
    #   
    #   WRF Cell Corner Grid points : XLONa, XLATa
    #
    #   INPUT : 'wrfinput_d01'
    #
    #   OUTPUT: XLONa[ng_we_wrf+1,ng_sn_wrf+1]
    #           XLATa[ng_we_wrf+1,ng_sn_wrf+1]
    #
    #--------------------------------------------------------
    #
    fn_XLON  = './XLON_d'+str(dd).zfill(2)+'.npy'
    fn_XLAT  = './XLAT_d'+str(dd).zfill(2)+'.npy'
    fn_XLONa = './XLONa_d'+str(dd).zfill(2)+'.npy'
    fn_XLATa = './XLATa_d'+str(dd).zfill(2)+'.npy'
    if not(os.path.isfile(fn_XLON ) and \
           os.path.isfile(fn_XLAT ) and \
           os.path.isfile(fn_XLONa) and \
           os.path.isfile(fn_XLATa)):
      if rank == 0:
        print(datetime.now(),' getting XLON, XLAT, XLONa, XLATa from ',fname)
      XLON, XLAT, XLONa, XLATa = WRF_Grids2(fname)
      np.save(fn_XLON,XLON)
      np.save(fn_XLAT,XLAT)
      np.save(fn_XLONa,XLONa)
      np.save(fn_XLATa,XLATa)
    else:
      if rank == 0:
        print(datetime.now(),' reading XLON, XLAT, XLONa, XLATa from npy files....')
      XLON  = np.load(fn_XLON )
      XLAT  = np.load(fn_XLAT )
      XLONa = np.load(fn_XLONa)
      XLATa = np.load(fn_XLATa)
    #
    #--------------------------
    #
    ncfile = Dataset(fname,'r')
    map_proj = ncfile.getncattr("MAP_PROJ")
    cen_lat  = ncfile.getncattr("CEN_LAT")
    cen_lon  = ncfile.getncattr("CEN_LON")
    truelat1 = ncfile.getncattr("TRUELAT1")
    truelat2 = ncfile.getncattr("TRUELAT2")
    stand_lon= ncfile.getncattr("STAND_LON")
    pole_lat = ncfile.getncattr("POLE_LAT")
    pole_lon = ncfile.getncattr("POLE_LON")
    ncfile.close()
    #
    # Local time and time zone
    #
    tzw     = tzwhere.tzwhere()
    #local_timedelta_onWRF = np.zeros_like(XLON)
    hour2 = np.zeros_like(XLON)
    start_year = start_dt.year
    start_month= start_dt.month
    start_day  = start_dt.day
    timezone0  = datetime(start_year,start_month,start_day)
    for ilon, lon_temp in np.ndenumerate(XLON):
      lat_temp = XLAT[ilon]
      tdelta, tz_str = lonlat2timezone(lon_temp,lat_temp,timezone0,tzw)
      if tz_str != 'None':
        hour2[ilon] = int(round(tdelta.total_seconds()/3600.0))
      else:
        hour2[ilon] = int(round(lon_temp/15.0))
      ## #debug#
      ## hour2[ilon] = int(round(lon_temp/15.0))
      ## #debug#end
      ## END OF if tz_str != 'None':
      if hour2[ilon] >= 48:
        hour2[ilon] -= 48
      if hour2[ilon] >= 24:
        hour2[ilon] -= 24
      if hour2[ilon] < 0:
        hour2[ilon] += 24
      ## END OF if hour2[ilon] >= 24:
    ## END OF for ilon, lon_temp in np.ndenumerate(XLON):

    ## for ihour2, hour2_temp in np.ndenumerate(hour2):
    ##   print('ihour2 and hour2_temp = ',ihour2, hour2_temp)

    #
    # Number of WRF grid points (we:west-->east, sn:south-->north)
    # 
    ng_we_wrf    = XLONa.shape[0]-1
    ng_sn_wrf    = XLONa.shape[1]-1
    #
    #--------------------------------------------------------------- 
    #  Grid area of [lon_anth,lat_anth] grid 
    #---------------------------------------------------------------
    #
    if Grid01:
      emis_cell_area01    = np.zeros(shape=(len(lon_glob01),len(lat_glob01)))
    if Grid025:
      emis_cell_area025    = np.zeros(shape=(len(lon_glob025),len(lat_glob025)))
    if Grid05:
      emis_cell_area05    = np.zeros(shape=(len(lon_glob05),len(lat_glob05)))
    if Grid10:
      emis_cell_area10    = np.zeros(shape=(len(lon_glob10),len(lat_glob10)))
    #
    #
    if Grid01:
      fn_emis_cell_area = './emis_cell_area01_d'+str(dd).zfill(2)+'.npy'
      if not os.path.isfile(fn_emis_cell_area):
        for i,lat0 in enumerate(lat_glob01):
          if rank == 0:
            print(datetime.now(),' Calculating cell area of emission grid... [unit:m^2] (0.1)', i)
          for j,lon0 in enumerate(lon_glob01):
            emis_cell_area01[j,i] = areaquad(lon_bound_glob01[j  ],lat_bound_glob01[i  ],\
                                             lon_bound_glob01[j+1],lat_bound_glob01[i+1])
        np.save(fn_emis_cell_area,emis_cell_area01)
      else:
        if rank == 0:
          print(datetime.now(),' reading emis_cell_area01 from an npy file')
        emis_cell_area01 = np.load(fn_emis_cell_area)
    #
    if Grid025:
      fn_emis_cell_area = './emis_cell_area025_d'+str(dd).zfill(2)+'.npy'
      if not os.path.isfile(fn_emis_cell_area):
        for i,lat0 in enumerate(lat_glob025):
          if rank == 0:
            print(datetime.now(),' Calculating cell area of emission grid... [unit:m^2] (0.25)', i)
          for j,lon0 in enumerate(lon_glob025):
            emis_cell_area025[j,i] = areaquad(lon_bound_glob025[j  ],lat_bound_glob025[i  ],\
                                              lon_bound_glob025[j+1],lat_bound_glob025[i+1])
        np.save(fn_emis_cell_area,emis_cell_area025)
      else:
        if rank == 0:
          print(datetime.now(),' reading emis_cell_area025 from an npy file')
        emis_cell_area025 = np.load(fn_emis_cell_area)
    #
    if Grid05:
      fn_emis_cell_area = './emis_cell_area05_d'+str(dd).zfill(2)+'.npy'
      if not os.path.isfile(fn_emis_cell_area):
        for i,lat0 in enumerate(lat_glob05):
          if rank == 0:
            print(datetime.now(), ' Calculating cell area of emission grid... [unit:m^2] (0.5)', i)
          for j,lon0 in enumerate(lon_glob05):
            emis_cell_area05[j,i] = areaquad(lon_bound_glob05[j  ],lat_bound_glob05[i  ],\
                                             lon_bound_glob05[j+1],lat_bound_glob05[i+1])
        np.save(fn_emis_cell_area,emis_cell_area05)
      else:
        if rank == 0:
          print(datetime.now(), 'reading emis_cell_area05 from an npy file')
        emis_cell_area05 = np.load(fn_emis_cell_area)
    #
    if Grid10:
      fn_emis_cell_area = './emis_cell_area10_d'+str(dd).zfill(2)+'.npy'
      if not os.path.isfile(fn_emis_cell_area):
        for i,lat0 in enumerate(lat_glob10):
          if rank == 0:
            print(datetime.now(), 'Calculating cell area of emission grid... [unit:m^2] (1.0)', i)
          for j,lon0 in enumerate(lon_glob10):
            emis_cell_area10[j,i] = areaquad(lon_bound_glob10[j  ],lat_bound_glob10[i  ],\
                                             lon_bound_glob10[j+1],lat_bound_glob10[i+1])
        np.save(fn_emis_cell_area,emis_cell_area10)
      else:
        if rank == 0:
          print(datetime.now(), 'reading emis_cell_area10 from an npy file')
        emis_cell_area10 = np.load(fn_emis_cell_area)
    #
    #--------------------------------------------------------------------------------------
    #
    #   END OF SECTION: GRID GENERATION
    #
    #======================================================================================
    
    
    #====================================================================================
    #-------------------------------------------------------------------------------------
    #
    #  SECTION: Ratio of common area w.r.t. wrf grids (XLONa,XLATa)
    #           1. './common_area01_dxx.npy' 
    #           2. './common_area025_dxx.npy' 
    #           3. './common_area05_dxx.npy' 
    #           4. './common_area10_dxx.npy' 
    #
    #-------------------------------------------------------------------------------------
    #
    #  Ratio of common area w.r.t. wrf grids (XLONa,XLATa)
    #
      
    if Grid01:
      area_common_array01, area_common_dict01 = create_commonarea(0.1,XLONa,XLATa,\
                                            lon_bound_glob01_2D,lat_bound_glob01_2D,\
                                            domain=dd,map_proj=map_proj)
    #
    if Grid025:
      area_common_array025, area_common_dict025 = create_commonarea(0.25,XLONa,XLATa,\
                                            lon_bound_glob025_2D,lat_bound_glob025_2D,\
                                            domain=dd,map_proj=map_proj)
    #
    if Grid05:
      area_common_array05, area_common_dict05  = create_commonarea(0.5,XLONa,XLATa,\
                                            lon_bound_glob05_2D,lat_bound_glob05_2D,\
                                            domain=dd,map_proj=map_proj)
    #
    if Grid10:
      area_common_array10, area_common_dict10  = create_commonarea(1.0,XLONa,XLATa,\
                                            lon_bound_glob10_2D,lat_bound_glob10_2D,\
                                            domain=dd,map_proj=map_proj)
    #
    #--------------------------------------------------------------------------
    #
    #  END of SECTION: creation of files 
    #           1. './common_area01_dxx.npy' 
    #           2. './common_area025_dxx.npy' 
    #           3. './common_area05_dxx.npy' 
    #           4. './common_area10_dxx.npy' 
    #
    #--------------------------------------------------------------------------
    #==========================================================================
    
    
    #=========================================================================
    #
    #  DECLARATION OF "emis_dict"
    #
    #    EXAMPLE : "key_name" = "<spec>_<sec>"
    #              <spec> : 'C2H5OH','CH3OH','HC5','ETH','TOL','OL2','KET','XYL','HCHO','OLT','ALD','ISO',etc...
    #              <sec>  : 'ene','dom','ind','ind','flr','slv','tra','shp','wst','awb','agr','oth','vol',etc...
    #    
    #    ATTENTION : the order of dimensions must be transposed when saved in NetCDF format
    #                i.e.) np.transpose(emis_temp[:,:,:])
    #
    #    emis_dict[key_name]['dimensions']={}
    #    emis_dict[key_name]['dimensions']['south_north']=<# of grid points in sn direction> 
    #    emis_dict[key_name]['dimensions']['west_east']  =<# of grid points in we direction>
    #    emis_dict[key_name]['dimensions']['time']       =<# of points in time series> e.g.:12
    #    emis_dict[key_name]['west_east']={}
    #    emis_dict[key_name]['west_east']['dtype']='i4'
    #    emis_dict[key_name]['west_east']['dims' ]=['west_east']
    #    emis_dict[key_name]['west_east']['units']=''
    #    emis_dict[key_name]['west_east']['data' ]=np.arange(ng_we_wrf) : [0,1,2,3,4,5,...,ng_we_wrf-1]
    #    emis_dict[key_name]['south_north']={}
    #    emis_dict[key_name]['south_north']['dtype']='i4'
    #    emis_dict[key_name]['south_north']['dims' ]=['south_north']
    #    emis_dict[key_name]['south_north']['units']=''
    #    emis_dict[key_name]['south_north']['data' ]=np.arange(ng_sn_wrf) : [0,1,2,3,4,5,...,ng_sn_wrf-1]
    #    emis_dict[key_name]['longitude']={}
    #    emis_dict[key_name]['longitude']['dtype']='f4'
    #    emis_dict[key_name]['longitude']['dims' ]=['west_east','south_north']
    #    emis_dict[key_name]['longitude']['units']='degrees_east'
    #    emis_dict[key_name]['longitude']['data' ]=XLON[:<we>,:<sn>] <WRF longitude grid>
    #    emis_dict[key_name]['latitude']={}
    #    emis_dict[key_name]['latitude']['dtype']='f4'
    #    emis_dict[key_name]['latitude']['dims' ]=['west_east','south_north']
    #    emis_dict[key_name]['latitude']['units']='degrees_east'
    #    emis_dict[key_name]['latitude']['data' ]=XLAT[:<we>,:<sn>]) <WRF latitude grid>
    #    emis_dict[key_name]['time']={}
    #    emis_dict[key_name]['time']['dtype']='i4'
    #    emis_dict[key_name]['time']['dims' ]=['time']
    #    emis_dict[key_name]['time']['data' ]=np.arange(12) : [0,1,2,3,4,...,11]
    #    emis_dict[key_name]['voc']={}
    #    emis_dict[key_name]['voc']['dtype']='f4'
    #    emis_dict[key_name]['voc']['dims' ]=['west_east','south_north','time'] or ['west_east','south_north']
    #    emis_dict[key_name]['voc']['units']=voc_units
    #    emis_dict[key_name]['voc']['data' ]=<emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
    #
    emis_dict = {}
    #==========================================================================
    #
    #  SECTION : INTERPOLATION FROM EMISSION DATA TO WRF GRID
    #
    #==========================================================================
    if rank == 0:
      print('------------------------------------')
      print('')
      print(datetime.now(),' SECTION : INTERPOLATION FROM EMISSION DATA TO WRF GRID')
      print('')
      print('------------------------------------')
    #==========================================================================
    #
    #
    #	* REAS Anthropogenic Emissions (NMVOC) 
    #          ____INPUT____
    #          - /data/onishi/REAS/<spec>/<year>/REASv2.1_NMV_<sc>_<sec>_<year>_0.25x0.25
    #            <year>    : '2008'
    #            <sc>      : 
    #                    01 : Ethane
    #                    02 : Propane
    #                    03 : Butanes
    #                    04 : Pentanes
    #                    05 : Other Alkanes
    #                    06 : Ethylene
    #                    07 : Propene
    #                    08 : Terminal Alkenes
    #                    09 : Internal Alkenes
    #                    10 : Acetylene
    #                    11 : Benzene
    #                    12 : Toluene
    #                    13 : Xylenes
    #                    14 : Other Aromatics
    #                    15 : Formaldehyde
    #                    16 : Other Aldehyde
    #                    17 : Ketones
    #                    18 : Halocarbons
    #                    19 : Others
    #                    20 : Total <----- USE THIS 
    #
    #            <sec>      : ['POWER_PLANTS_NON-POINT','POWER_PLANTS_NON-POINT_JPN','INDUSTRY', \
    #                          'ROAD_TRANSPORT','AVIATION','INTNNV','OTHER_TRANSPORT',          \
    #                          'DOMESTIC','FUGITIVE_COAL','FUGITIVE_OIL','FUGITIVE_GAS',        \
    #                          'EXTRACTION','SOLVENTS','FERTILIZER',                            \
    #                          'MANURE_MANAGEMENT','ENTERIC_FERMENTATION',                      \
    #                          'RICE_CULTIVATION','SOIL','SOIL_DIRECT','SOIL_INDIRECT',         \
    #                          'WASTE','MISC']                    
    #                 'POWER_PLANTS_NON-POINT'    : (ene )Power and heat plants as non-point sources except for Japan
    #                 'POWER_PLANTS_NON-POINT_JPN : (ene )Power and heat plants as non-point sources for Japan
    #                 'INDUSTRY'                  : (ind )Industry (emissions both from fuel combustion and industrial processes)
    #                 'ROAD_TRANSPORT'            : (tra )Road transport (cars, buses, trucks, motor cycles, and other on-road vehicles)
    #                 'AVIATION'                  : (shp )Domestic and international aviation (0-1km)
    #                 'INTNNV'                    : (shp )International navigation
    #                 'OTHER_TRANSPORT'           : (shp )Domestic navigation, railway, and other off-road transports
    #                 'DOMESTIC'                  : (dom )Residential, commerce and public services, agricultural equipment, fishing, and others.
    #                 'FUGITIVE_COAL'             : (flr?)Fugitive emissions from production, processing, and distribution of coal (For CH4)
    #                 'FUGITIVE_OIL'              : (flr?)Fugitive emissions from production, processing, and distribution of oil (For CH4)
    #                 'FUGITIVE_GAS'              : (flr?)Fugitive emissions from production, processing, and distribution of gas (For CH4)
    #                 'EXTRACTION'                : (flr )Extraction and handling of fossil fuels (For NMVOC)
    #                 'SOLVENTS'                  : (slv )Solvent use (including paint use)
    #                 'FERTILIZER'                : (??? )Fertilizer application
    #                 'MANURE_MANAGEMENT'         : (??? )Manure management of livestock
    #                 'ENTERIC_FERMENTATION'      : (??? )Enteric fermentation of livestock (For CH4)
    #                 'RICE_CULTIVATION'          : (??? )Rice cultivation (For CH4)
    #                 'SOIL'                      : (soi )Soil NOx emissions
    #                 'SOIL_DIRECT'               : (soi )Direct soil N2O emissions
    #                 'SOIL_INDIRECT'             : (soi )Indirect soil N2O emissions
    #                 'WASTE'                     : (wst )Waste treatment (both solid and water waste)
    #                 'MISC'                      : (??? )Human respiration and perspiration, latrines, and others (For NH3)
    #            resolution : 0.25[deg] x 0.25[deg]
    #            latitutde  : From 11S [deg] to 77N [deg]
    #            longitude  : From 46E [deg] to 180E [deg]
    #            unit       : [ton/month]
    #
    #          ____OUTPUT____
    #
    #            "emis_dict": dictionary : emission on WRF grid
    #            keynames   : 'awb-REAS-VOC','dom-REAS-VOC','ene-REAS-VOC','ind-REAS-VOC','slv-REAS-VOC','tra-REAS-VOC','wst-REAS-VOC','all-REAS-VOC'
    #            unit       : [mol/sec/m2]   
    #            example    :
    #               emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
    #               emis_dict[keyname]['dimensions']={}
    #               emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
    #               emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
    #               emis_dict[keyname]['dimensions']['time']       = 12        # of points in time series> e.g.:12
    #               emis_dict[keyname]['west_east']={}
    #               emis_dict[keyname]['west_east']['dtype']='i4'
    #               emis_dict[keyname]['west_east']['dims' ]=['west_east']
    #               emis_dict[keyname]['west_east']['units']=''
    #               emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
    #               emis_dict[keyname]['south_north']={}
    #               emis_dict[keyname]['south_north']['dtype']='i4'
    #               emis_dict[keyname]['south_north']['dims' ]=['south_north']
    #               emis_dict[keyname]['south_north']['units']=''
    #               emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
    #               emis_dict[keyname]['longitude']={}
    #               emis_dict[keyname]['longitude']['dtype']='f4'
    #               emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['longitude']['units']='degrees_east'
    #               emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
    #               emis_dict[keyname]['latitude']={}
    #               emis_dict[keyname]['latitude']['dtype']='f4'
    #               emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['latitude']['units']='degrees_east'
    #               emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
    #               emis_dict[keyname]['time']={}
    #               emis_dict[keyname]['time']['dtype']='i4'
    #               emis_dict[keyname]['time']['dims' ]=['time']
    #               emis_dict[keyname]['time']['data' ]=np.arange(12) : [0,1,2,3,4,...,11]
    #               emis_dict[keyname]['voc']={}
    #               emis_dict[keyname]['voc']['dtype']='f4'
    #               emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] or ['west_east','south_north']
    #               # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['voc']['units']='mol/sec/m2'
    #               emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
    #                    
    #====================================================================
    #--------------------------------------------------------------------
    #
    if REAS_ANTH_VOC:
      year      = '2008'
      reas_dir  = '/data/onishi/REAS/NMV/2008/'
      # filename = 'REASv2.1_NMV_20_<sec>_<year>_0.25x0.25'
      sectors   = ['POWER_PLANTS_NON-POINT','POWER_PLANTS_NON-POINT_JPN','INDUSTRY', \
                   'ROAD_TRANSPORT','AVIATION','INTNNV','OTHER_TRANSPORT',          \
                   'DOMESTIC','FUGITIVE_COAL','FUGITIVE_OIL','FUGITIVE_GAS',        \
                   'EXTRACTION','SOLVENTS','FERTILIZER',                            \
                   'MANURE_MANAGEMENT','ENTERIC_FERMENTATION',                      \
                   'RICE_CULTIVATION','SOIL','SOIL_DIRECT','SOIL_INDIRECT',         \
                   'WASTE','MISC']                    
      for sec in sectors:
        if sec in ['POWER_PLANTS_NON-POINT','POWER_PLANTS_NON-POINT_JPN']:
          sec2 = 'ene'
        if sec == 'INDUSTRY':
          sec2 = 'ind'
        if sec == 'ROAD_TRANSPORT':
          sec2 = 'tra'
        if sec in ['AVIATION','INTNNV','OTHER_TRANSPORT']:
          sec2 = 'shp'
        if sec == 'DOMESTIC':
          sec2 = 'dom'
        if sec in [ 'FUGITIVE_COAL','FUGITIVE_OIL','FUGITIVE_GAS','EXTRACTION']:
          sec2 = 'flr'
        if sec == 'SOLVENTS':
          sec2 = 'slv'
        if sec in ['FERTILIZER','MANURE_MANAGEMENT','ENTERIC_FERMENTATION','RICE_CULTIVATION']:
          continue
        if sec in ['SOIL','SOIL_DIRECT','SOIL_INDIRECT']:
          sec2 = 'soi'
        if sec == 'WASTE':
          sec2 = 'wst'                    
        reas_file = reas_dir+'/20/REASv2.1_NMV_20_'+sec+'_'+year+'_0.25x0.25'
        if rank == 0:
          print(datetime.now(), 'reas_file = ', reas_file)
        if not os.path.isfile(reas_file):
          continue
        #
        lon, lat, emis_reas = read_REAS(filename=reas_file)
        if np.abs(np.abs(lon[1]-lon[0])-0.25) > 0.001:
          sys.exit("check the resolution of emission data :"+reas_file)
        #
        print('....in AddDict_REAS_Anth_VOC_OnWRF....')
        emis_dict = AddDict_REAS_Anth_VOC_OnWRF(emis_dict,reas_file, \
                            area_common_array025, area_common_dict025, emis_cell_area025, \
                            XLON, XLAT, sec2) 
      # END of loop over sector
      #  
      sorted_key_list = [] 
      for key in emis_dict:
        sorted_key_list.append(key)
      #  plot_emis_dict(key)
      ### for ikey, key in enumerate(sorted(sorted_key_list)):
      ###   if rank == 0:
      ###     print(datetime.now(), ikey, key)
      ## raw_input()
      ## 
      ## for key in emis_dict:
      ##   print key
      ##   data = np.array(emis_dict[key]['voc']['data'])
      ##   if np.amax(data) != 0.0:
      ##     print key
      ##     print data.shape
      ##     plot_emis_dict(key,0)
      ## raw_input('check after Speciate_SAPRC_ECLIPSE_Anth_VOC_OnWRF')
      #
      #.............................
      #
      #  Speciate FROM ['dom-REAS-VOC','ene-REAS-VOC','ind-REAS-VOC','slv-REAS-VOC','tra-REAS-VOC','wst-REAS-VOC']
      #             TO <sec>-REAS-<spec>
      # 
      #           WHERE
      #            <sec> = ['ene','dom','ind1','ind2','slv','tra','wst']
      #   CBMZ :   <spec>= ['C2H5OH','CH3OH','HC5','ETH','TOL','OL2','KET','XYL','HCHO','OLT','ALD','ISO','OLI']
      #   SAPRC:   <spec>= ['C2H6','C3H8','C2H2','ALK3','ALK4','ALK5','ETHENE','C3H6','OLE1','OLE2','ARO1','ARO2','CCHO','HCHO','ACET','MEK','TERP','MEOH','PROD2']
      #
      #           UNIT: [kg/m2/sec] --> [mol/m2/sec]
      #..............................
      #
      # 
      #emis_dict = Speciate_CBMZ_Anth_VOC_OnWRF(emis_dict)
      emis_dict = Speciate_SAPRC_Anth_VOC_OnWRF(emis_dict)
      #
      key_del_list = []
      for key in emis_dict:
        if 'VOC' in key:
          key_del_list.append(key)
      for key in key_del_list:
        #print 'key to be deleted : ', key
        del emis_dict[key]
      ## sorted_key_list = [] 
      ## for key in emis_dict:
      ##   sorted_key_list.append(key)
      ## #  plot_emis_dict(key)
      ## for ikey, key in enumerate(sorted(sorted_key_list)):
      ##   data = np.array(emis_dict[key]['voc']['data'])
      ##   print('sorted key : ', ikey, key, data.shape)
      ## sys.exit()
      #input()
      ## 
      ## for key in emis_dict:
      ##   print key
      ##   data = np.array(emis_dict[key]['voc']['data'])
      ##   if np.amax(data) != 0.0:
      ##     print key
      ##     print data.shape
      ##     plot_emis_dict(key,0)
      ## raw_input('check after Speciate_SAPRC_ECLIPSE_Anth_VOC_OnWRF')
    
    #==========================================================================
    #==========================================================================
    #
    #
    #	* REAS Anthropogenic Emissions (Other) 
    #          ____INPUT____
    #          - /data/onishi/REAS/<spec>/<year>/REASv2.1_<spec>_<sec>_<year>_0.25x0.25
    #            <year>     : '2008' 
    #            <spec>     : ['BC_','CH4','CO_','CO2','N2O','NH3','OC_','PM10_','PM2.5']
    #            <sec>      : ['POWER_PLANTS_NON-POINT','POWER_PLANTS_NON-POINT_JPN','INDUSTRY', \
    #                          'ROAD_TRANSPORT','AVIATION','INTNNV','OTHER_TRANSPORT',          \
    #                          'DOMESTIC','FUGITIVE_COAL','FUGITIVE_OIL','FUGITIVE_GAS',        \
    #                          'EXTRACTION','SOLVENTS','FERTILIZER',                            \
    #                          'MANURE_MANAGEMENT','ENTERIC_FERMENTATION',                      \
    #                          'RICE_CULTIVATION','SOIL','SOIL_DIRECT','SOIL_INDIRECT',         \
    #                          'WASTE','MISC']                    
    #                 'POWER_PLANTS_NON-POINT'    : (ene )Power and heat plants as non-point sources except for Japan
    #                 'POWER_PLANTS_NON-POINT_JPN : (ene )Power and heat plants as non-point sources for Japan
    #                 'INDUSTRY'                  : (ind )Industry (emissions both from fuel combustion and industrial processes)
    #                 'ROAD_TRANSPORT'            : (tra )Road transport (cars, buses, trucks, motor cycles, and other on-road vehicles)
    #                 'AVIATION'                  : (shp )Domestic and international aviation (0-1km)
    #                 'INTNNV'                    : (shp )International navigation
    #                 'OTHER_TRANSPORT'           : (shp )Domestic navigation, railway, and other off-road transports
    #                 'DOMESTIC'                  : (dom )Residential, commerce and public services, agricultural equipment, fishing, and others.
    #                 'FUGITIVE_COAL'             : (flr?)Fugitive emissions from production, processing, and distribution of coal (For CH4)
    #                 'FUGITIVE_OIL'              : (flr?)Fugitive emissions from production, processing, and distribution of oil (For CH4)
    #                 'FUGITIVE_GAS'              : (flr?)Fugitive emissions from production, processing, and distribution of gas (For CH4)
    #                 'EXTRACTION'                : (flr )Extraction and handling of fossil fuels (For NMVOC)
    #                 'SOLVENTS'                  : (slv )Solvent use (including paint use)
    #                 'FERTILIZER'                : (??? )Fertilizer application
    #                 'MANURE_MANAGEMENT'         : (??? )Manure management of livestock
    #                 'ENTERIC_FERMENTATION'      : (??? )Enteric fermentation of livestock (For CH4)
    #                 'RICE_CULTIVATION'          : (??? )Rice cultivation (For CH4)
    #                 'SOIL'                      : (soi )Soil NOx emissions
    #                 'SOIL_DIRECT'               : (soi )Direct soil N2O emissions
    #                 'SOIL_INDIRECT'             : (soi )Indirect soil N2O emissions
    #                 'WASTE'                     : (wst )Waste treatment (both solid and water waste)
    #                 'MISC'                      : (??? )Human respiration and perspiration, latrines, and others (For NH3)
    #            resolution : 0.25[deg] x 0.25[deg]
    #            latitutde  : From 11S [deg] to 77N [deg]
    #            longitude  : From 46E [deg] to 180E [deg]
    #            unit       : [ton/month]
    #           
    #          ____OUTPUT___
    #            keynames   : 'awb_<spec>','dom_<spec>','ene_<spec>','ind_<spec>','slv_<spec>','tra_<spec>','wst_<spec>','all_<spec>'
    #                 <spec>  : ['BC','CH4','CO','CO2','NH3','OC','PM25']
    #            dimension  : ['west_east','south_north','time'=12]
    #            unit       : [mol/sec/m2] or [ug/sec/m2] (for 'BC','OC','PM10','PM2.5')
    #            example    :
    #               emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
    #               emis_dict[keyname]['dimensions']={}
    #               emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
    #               emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
    #               emis_dict[keyname]['dimensions']['time']       =<# of points in time series> e.g.:12
    #               emis_dict[keyname]['west_east']={}
    #               emis_dict[keyname]['west_east']['dtype']='i4'
    #               emis_dict[keyname]['west_east']['dims' ]=['west_east']
    #               emis_dict[keyname]['west_east']['units']=''
    #               emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
    #               emis_dict[keyname]['south_north']={}
    #               emis_dict[keyname]['south_north']['dtype']='i4'
    #               emis_dict[keyname]['south_north']['dims' ]=['south_north']
    #               emis_dict[keyname]['south_north']['units']=''
    #               emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
    #               emis_dict[keyname]['longitude']={}
    #               emis_dict[keyname]['longitude']['dtype']='f4'
    #               emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['longitude']['units']='degrees_east'
    #               emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
    #               emis_dict[keyname]['latitude']={}
    #               emis_dict[keyname]['latitude']['dtype']='f4'
    #               emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['latitude']['units']='degrees_east'
    #               emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
    #               emis_dict[keyname]['time']={}
    #               emis_dict[keyname]['time']['dtype']='i4'
    #               emis_dict[keyname]['time']['dims' ]=['time']
    #               emis_dict[keyname]['time']['data' ]=np.arange(12) : [0,1,2,3,4,...,11]
    #               emis_dict[keyname]['voc']={}
    #               emis_dict[keyname]['voc']['dtype']='f4'
    #               emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] or ['west_east','south_north']
    #               # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['voc']['units']=[mol/m2/sec] or [ug/m2/sec]('BC','OC','PM25')
    #               emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
    #                    
    #====================================================================
    #--------------------------------------------------------------------
    #
    #  Open Edgar HTAP emission data file  'edgar_HTAP_<spec>_emi_<sec>_<year>_<month>.0.1x0.1.nc'
    #
    if REAS_ANTH_Other:
      year      = '2008'
      ## spec      = ['BC','CH4','CO','CO2','NH3','OC','PM25']
      spec      = ['BC','CH4','CO','NH3','OC','PM25','NOx']
      spec_     = {'BC':'BC_','CH4':'CH4','CO':'CO_','CO2':'CO2','NH3':'NH3','OC':'OC_','PM25':'PM2.5','NOx':'NOx'}
      sectors   = ['POWER_PLANTS_NON-POINT','POWER_PLANTS_NON-POINT_JPN','INDUSTRY', \
                   'ROAD_TRANSPORT','AVIATION','INTNNV','OTHER_TRANSPORT',          \
                   'DOMESTIC','FUGITIVE_COAL','FUGITIVE_OIL','FUGITIVE_GAS',        \
                   'EXTRACTION','SOLVENTS','FERTILIZER',                            \
                   'MANURE_MANAGEMENT','ENTERIC_FERMENTATION',                      \
                   'RICE_CULTIVATION','SOIL','SOIL_DIRECT','SOIL_INDIRECT',         \
                   'WASTE','MISC']                    
      # TEST
      #spec      = ['BC','CO']
      for ispec, spec_temp in enumerate(spec):
        for sec in sectors:
          if sec in ['POWER_PLANTS_NON-POINT','POWER_PLANTS_NON-POINT_JPN']:
            sec2 = 'ene'
          if sec == 'INDUSTRY':
            sec2 = 'ind'
          if sec == 'ROAD_TRANSPORT':
            sec2 = 'tra'
          if sec in ['AVIATION','INTNNV','OTHER_TRANSPORT']:
            sec2 = 'shp'
          if sec == 'DOMESTIC':
            sec2 = 'dom'
          if sec in [ 'FUGITIVE_COAL','FUGITIVE_OIL','FUGITIVE_GAS','EXTRACTION']:
            sec2 = 'flr'
          if sec == 'SOLVENTS':
            sec2 = 'slv'
          if sec in ['FERTILIZER','MANURE_MANAGEMENT','ENTERIC_FERMENTATION','RICE_CULTIVATION']:
            continue
          if sec in ['SOIL','SOIL_DIRECT','SOIL_INDIRECT']:
            sec2 = 'soi'
          if sec == 'WASTE':
            sec2 = 'wst'                    
          reas_dir   ='/data/onishi/REAS/'
          reas_file  = reas_dir+spec_[spec_temp]+'/'+year+'/REASv2.1_'+spec_[spec_temp] \
                                 +'_'+sec+'_'+year+'_0.25x0.25'
          if rank == 0:
            print(datetime.now(), 'reas_file = ',reas_file)
          if not os.path.isfile(reas_file):
            continue
          #
          lon, lat, emis_reas = read_REAS(filename=reas_file)
          if np.abs(np.abs(lon[1]-lon[0])-0.25) > 0.001:
            sys.exit("check the resolution of emission data :"+reas_file)
          #
          # ADD : 'awb_<spec>','dom_<spec>','ene_<spec>','ind_<spec>', etc...
          # UNIT: [ton/month]-->[mol/m2/sec] or [ug/m2/sec] ('OC','PM25','BC') 
          #
          print('....in AddDict_REAS_Anth_Other_OnWRF....')
          emis_dict = AddDict_REAS_Anth_Other_OnWRF(emis_dict,reas_file,\
                                  area_common_array025, area_common_dict025, emis_cell_area025,\
                                  XLON,XLAT,sec2,spec_temp)
        # END OF loop on sectors
      # END OF for ispec, spec_temp in enumerate(spec):
      ### sorted_key_list = [] 
      ### for key in emis_dict:
      ###   sorted_key_list.append(key)
      ### #  plot_emis_dict(key)
      ### for ikey, key in enumerate(sorted(sorted_key_list)):
      ###   print ikey, key
      ### # raw_input()
      ### 
      ### for key in emis_dict:
      ###   print key
      ###   data = np.array(emis_dict[key]['voc']['data'])
      ###   if np.amax(data) != 0.0:
      ###     print key
      ###     print data.shape
      ###     plot_emis_dict(key,0)
      ### raw_input('check after AddDict_REAS_Other_OnWRF')
      #
        
    
    #==========================================================================
    #
    #
    #	* HTAP Anthropogenic Emissions (NMVOC) 
    #          ____INPUT____
    #          - /data/raut/EMISSIONS/HTAP_v2/<year>/edgar_HTAP_NMVOC_emi_<sec>_<year>.0.1x0.1.nc
    #            <year>    : '2008' or '2010'
    #            <sec>     : ['ENERGY','INDUSTRY','RESIDENTIAL','TRANSPORT','AGRICULTURE']
    #            variable  : 'emi_nmvoc'
    #            resolution: 0.1 deg x 0.1 deg
    #            dimension : [lat,lon] (1800,3600) (-89.95<lat<89.95, 0.05<lon<359.95)
    #            unit      : [kg/m2/sec]
    #
    #            Sectors  : AGRICULTURE  --> awb    : Agriculture (waste burning on field)
    #                       RESIDENTIAL  --> dom    : Residential and commercial
    #                       ENERGY       --> ene    : Power plants, energy conversion, extraction 
    #                       INDUSTRY     --> ind    : Industry (combustion and processing)
    #                          * ind1 : Industrial (Stationary Combustion)
    #                          * ind2 : Production Process 
    #                       ???????      --> slv    : Solvent Use
    #                       ???????      --> flr    : Extraction and distribution of fossil
    #                       TRANSPORT    --> tra    : Road transport
    #                       AIR,SHIPS    --> shp    : Other transport ***** To be treated separately *****
    #                       ???????      --> wst    : Waste treatment and disposal
    #
    #          ____OUTPUT____
    #
    #            "emis_dict": dictionary : emission on WRF grid
    #            keynames   : 'awb-HTAP-VOC','dom-HTAP-VOC','ene-HTAP-VOC','ind-HTAP-VOC','slv-HTAP-VOC','tra-HTAP-VOC','wst-HTAP-VOC','all-HTAP-VOC'
    #            unit       : [mol/year/km2]   
    #            example    :
    #               emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
    #               emis_dict[keyname]['dimensions']={}
    #               emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
    #               emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
    #               # emis_dict[keyname]['dimensions']['time']       =<# of points in time series> e.g.:12
    #               emis_dict[keyname]['west_east']={}
    #               emis_dict[keyname]['west_east']['dtype']='i4'
    #               emis_dict[keyname]['west_east']['dims' ]=['west_east']
    #               emis_dict[keyname]['west_east']['units']=''
    #               emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
    #               emis_dict[keyname]['south_north']={}
    #               emis_dict[keyname]['south_north']['dtype']='i4'
    #               emis_dict[keyname]['south_north']['dims' ]=['south_north']
    #               emis_dict[keyname]['south_north']['units']=''
    #               emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
    #               emis_dict[keyname]['longitude']={}
    #               emis_dict[keyname]['longitude']['dtype']='f4'
    #               emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['longitude']['units']='degrees_east'
    #               emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
    #               emis_dict[keyname]['latitude']={}
    #               emis_dict[keyname]['latitude']['dtype']='f4'
    #               emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['latitude']['units']='degrees_east'
    #               emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
    #               # emis_dict[keyname]['time']={}
    #               # emis_dict[keyname]['time']['dtype']='i4'
    #               # emis_dict[keyname]['time']['dims' ]=['time']
    #               # emis_dict[keyname]['time']['data' ]=np.arange(12) : [0,1,2,3,4,...,11]
    #               emis_dict[keyname]['voc']={}
    #               emis_dict[keyname]['voc']['dtype']='f4'
    #               # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] or ['west_east','south_north']
    #               emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['voc']['units']='mol/sec/m2'
    #               emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
    #                    
    #====================================================================
    #--------------------------------------------------------------------
    #
    #if 1 == 0:
    if HTAP_ANTH_VOC:
      year      = '2010'
      htap_dir  ='/data/raut/EMISSIONS/HTAP_v2/'+year+'/'
      htap_voc_files = glob.glob(htap_dir+'edgar_HTAP_NMVOC_emi_*_'+year+'_*.0.1x0.1.nc')
      sectors   = ['ENERGY','INDUSTRY','RESIDENTIAL','TRANSPORT']
      #  
      for sec in sectors:
        htap_voc_filenames = [ x for x in htap_voc_files if sec in x ] 
        if rank == 0:
          print(datetime.now(), htap_voc_filenames)
        #
        nc        = Dataset(htap_voc_filenames[0],'r',format='NETCDF4')
        lon       = np.array(nc.variables["lon"])
        nc.close()
        if np.abs(np.abs(lon[1]-lon[0])-0.1) > 0.001:
          sys.exit("check the resolution of emission data :"+htap_voc_filenames[0])
        #
        # ADD : 'awb-HTAP-VOC','dom-HTAP-VOC','ene-HTAP-VOC','ind-HTAP-VOC', etc...
        # UNIT: [kg/m2/sec] 
        #
        emis_dict = AddDict_HTAP_Anth_VOC_OnWRF(emis_dict,htap_voc_filenames,\
                                area_common_array01, area_common_dict01, emis_cell_area01,\
                                XLON,XLAT)
      # END OF loop on sectors
      ## sorted_key_list = [] 
      ## for key in emis_dict:
      ##   sorted_key_list.append(key)
      ## #  plot_emis_dict(key)
      ## for ikey, key in enumerate(sorted(sorted_key_list)):
      ##   print ikey, key
      ## # raw_input()
      ## 
      ## for key in emis_dict:
      ##   print key
      ##   data = np.array(emis_dict[key]['voc']['data'])
      ##   if np.amax(data) != 0.0:
      ##     print key
      ##     print data.shape
      ##     plot_emis_dict(key,0)
      ## raw_input('check after Speciate_SAPRC_ECLIPSE_Anth_VOC_OnWRF')
      #
      #.............................
      #
      #  Speciate FROM ['dom-HTAP-VOC','ene-HTAP-VOC','ind-HTAP-VOC','slv-HTAP-VOC','tra-HTAP-VOC','wst-HTAP-VOC']
      #             TO <sec>-HTAP-<spec>
      # 
      #           WHERE
      #            <sec> = ['ene','dom','ind1','ind2','slv','tra','wst']
      #   CBMZ :   <spec>= ['C2H5OH','CH3OH','HC5','ETH','TOL','OL2','KET','XYL','HCHO','OLT','ALD','ISO','OLI']
      #   SAPRC:   <spec>= ['C2H6','C3H8','C2H2','ALK3','ALK4','ALK5','ETHENE','C3H6','OLE1','OLE2','ARO1','ARO2','CCHO','HCHO','ACET','MEK','TERP','MEOH','PROD2']
      #
      #           UNIT: [kg/m2/sec] --> [mol/m2/sec]
      #..............................
      #
      # 
      #emis_dict = Speciate_CBMZ_Anth_VOC_OnWRF(emis_dict)
      emis_dict = Speciate_SAPRC_Anth_VOC_OnWRF(emis_dict)
      #
      key_del_list = []
      for key in emis_dict:
        if 'VOC' in key:
          key_del_list.append(key)
      for key in key_del_list:
        #print 'key to be deleted : ', key
        del emis_dict[key]
      
      ## sorted_key_list = [] 
      ## for key in emis_dict:
      ##   sorted_key_list.append(key)
      ## #  plot_emis_dict(key)
      ## for ikey, key in enumerate(sorted(sorted_key_list)):
      ##   print 'sorted key : ', ikey, key
      ## # raw_input()
      ## 
      ## for key in emis_dict:
      ##   print key
      ##   data = np.array(emis_dict[key]['voc']['data'])
      ##   if np.amax(data) != 0.0:
      ##     print key
      ##     print data.shape
      ##     plot_emis_dict(key,0)
      ## raw_input('check after Speciate_SAPRC_ECLIPSE_Anth_VOC_OnWRF')
    
    #==========================================================================
    #==========================================================================
    #
    #
    #	* HTAP Anthropogenic Emissions (Other) 
    #          ____INPUT____
    #          - /data/raut/EMISSIONS/HTAP_v2/<year>/edgar_HTAP_<spec>_emi_<sec>_<year>_<month>.0.1x0.1.nc
    #            <year>    : '2008' or '2010'
    #            <spec>    : ['BC','CO','NH3','NOx','OC','PM10','PM25','SO2']
    #            <sec>     : ['ENERGY','INDUSTRY','RESIDENTIAL','TRANSPORT','AGRICULTURE']
    #            variable  : 'emi_<spec_lower>'
    #            resolution: 0.1 deg x 0.1 deg
    #            dimension : [lat,lon] (1800,3600) (-89.95<lat<89.95, 0.05<lon<359.95)
    #            unit      : [kg/m2/sec]
    #
    #            Sectors  : AGRICULTURE  --> awb    : Agriculture (waste burning on field)
    #                       RESIDENTIAL  --> dom    : Residential and commercial
    #                       ENERGY       --> ene    : Power plants, energy conversion, extraction 
    #                       INDUSTRY     --> ind    : Industry (combustion and processing)
    #                          * ind1 : Industrial (Stationary Combustion)
    #                          * ind2 : Production Process 
    #                       ???????      --> slv    : Solvent Use
    #                       ???????      --> flr    : Extraction and distribution of fossil
    #                       TRANSPORT    --> tra    : Road transport
    #                       AIR,SHIPS    --> shp    : Other transport ***** To be treated separately *****
    #                       ???????      --> wst    : Waste treatment and disposal
    #
    #          ____OUTPUT____
    #
    #            "emis_dict": dictionary : emission on WRF grid
    #            keynames   : 'awb-HTAP-<spec>','dom-HTAP-<spec>','ene-HTAP-<spec>','ind-HTAP-<spec>','slv-HTAP-<spec>','tra-HTAP-<spec>','wst-HTAP-<spec>','all-HTAP-<spec>'
    #            keynames   : <sec>-HTAP-<spec>
    #                 <sec>   : 'agr','awb','dom','ene','flr','ind','tra','wst','all'
    #                 <spec>  : ['BC','CO','NH3','NOx','OC','PM10','PM25','SO2']
    #            unit       : [mol/m2/sec] or [ug/m2/sec]('BC','OC','PM25')
    #            example    :
    #               emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
    #               emis_dict[keyname]['dimensions']={}
    #               emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
    #               emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
    #               # emis_dict[keyname]['dimensions']['time']       =<# of points in time series> e.g.:12
    #               emis_dict[keyname]['west_east']={}
    #               emis_dict[keyname]['west_east']['dtype']='i4'
    #               emis_dict[keyname]['west_east']['dims' ]=['west_east']
    #               emis_dict[keyname]['west_east']['units']=''
    #               emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
    #               emis_dict[keyname]['south_north']={}
    #               emis_dict[keyname]['south_north']['dtype']='i4'
    #               emis_dict[keyname]['south_north']['dims' ]=['south_north']
    #               emis_dict[keyname]['south_north']['units']=''
    #               emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
    #               emis_dict[keyname]['longitude']={}
    #               emis_dict[keyname]['longitude']['dtype']='f4'
    #               emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['longitude']['units']='degrees_east'
    #               emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
    #               emis_dict[keyname]['latitude']={}
    #               emis_dict[keyname]['latitude']['dtype']='f4'
    #               emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['latitude']['units']='degrees_east'
    #               emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
    #               # emis_dict[keyname]['time']={}
    #               # emis_dict[keyname]['time']['dtype']='i4'
    #               # emis_dict[keyname]['time']['dims' ]=['time']
    #               # emis_dict[keyname]['time']['data' ]=np.arange(12) : [0,1,2,3,4,...,11]
    #               emis_dict[keyname]['voc']={}
    #               emis_dict[keyname]['voc']['dtype']='f4'
    #               # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] or ['west_east','south_north']
    #               emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['voc']['units']=[mol/m2/sec] or [ug/m2/sec]('BC','OC','PM25')
    #               emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
    #                    
    #====================================================================
    #--------------------------------------------------------------------
    #
    #  Open Edgar HTAP emission data file  'edgar_HTAP_<spec>_emi_<sec>_<year>_<month>.0.1x0.1.nc'
    #
    #if 1 == 0:
    if HTAP_ANTH_Other:
      year      = '2010'
      spec      = ['BC','CO','NH3','NOx','OC','PM10','PM25','SO2']
      # TEST
      #spec      = ['BC','CO']
      for ispec, spec_temp in enumerate(spec):
        htap_dir   ='/data/raut/EMISSIONS/HTAP_v2/'+year+'/'
        htap_files = glob.glob(htap_dir+'edgar_HTAP_'+spec_temp+'_emi_*_'+year+'_*.0.1x0.1.nc')
        if rank == 0:
          print(datetime.now(), htap_files)
        # TEST
        #sectors = ['RESIDENTIAL']
        sectors   = ['ENERGY','INDUSTRY','RESIDENTIAL','TRANSPORT','AGRICULTURE']
        #  
        for sec in sectors:
          htap_filenames = [ x for x in htap_files if sec in x ] 
          if rank == 0:
            print(datetime.now(), htap_filenames)
          if len(htap_filenames) == 0:
            continue
          #
          nc        = Dataset(htap_filenames[0],'r',format='NETCDF4')
          lon       = np.array(nc.variables["lon"])
          nc.close()
          if np.abs(np.abs(lon[1]-lon[0])-0.1) > 0.001:
            sys.exit("check the resolution of emission data :"+htap_filenames[0])
          #
          # ADD : 'awb-HTAP-<spec>','dom-HTAP-<spec>','ene-HTAP-<spec>','ind-HTAP-<spec>', etc...
          # UNIT: [kg/m2/sec]-->[mol/m2/sec] or [ug/m2/sec] ('OC','PM25','BC') 
          #
          emis_dict = AddDict_HTAP_Anth_Other_OnWRF(emis_dict,htap_filenames,\
                                  area_common_array01, area_common_dict01, emis_cell_area01,\
                                  XLON,XLAT,spec_temp)
        # END OF loop on sectors
      # END OF for ispec, spec_temp in enumerate(spec):
      ### sorted_key_list = [] 
      ### for key in emis_dict:
      ###   sorted_key_list.append(key)
      ### #  plot_emis_dict(key)
      ### for ikey, key in enumerate(sorted(sorted_key_list)):
      ###   print ikey, key
      ### # raw_input()
      ### 
      ### for key in emis_dict:
      ###   print key
      ###   data = np.array(emis_dict[key]['voc']['data'])
      ###   if np.amax(data) != 0.0:
      ###     print key
      ###     print data.shape
      ###     plot_emis_dict(key,0)
      ### raw_input('check after AddDict_HTAP_Other_OnWRF')
      #
        
    
    #==========================================================================
    #==========================================================================
    #
    #
    #	* HTAP Anthropogenic Emissions (NMVOC) [AIR, SHIPS] 
    #          ____INPUT____
    #          - /data/raut/EMISSIONS/HTAP_v2/<year>/edgar_HTAP_NMVOC_emi_<sec>_<year>.0.1x0.1.nc
    #            <year>    : '2008' or '2010'
    #            <sec>     : ['AIR','SHIPS']
    #            variable  : 'emi_nmvoc'
    #            resolution: 0.1 deg x 0.1 deg
    #            dimension : [lat,lon] (1800,3600) (-89.95<lat<89.95, 0.05<lon<359.95)
    #            unit      : [kg/m2/sec]
    #
    #            Sectors  : AGRICULTURE  --> awb    : Agriculture (waste burning on field)
    #                       RESIDENTIAL  --> dom    : Residential and commercial
    #                       ENERGY       --> ene    : Power plants, energy conversion, extraction 
    #                       INDUSTRY     --> ind    : Industry (combustion and processing)
    #                          * ind1 : Industrial (Stationary Combustion)
    #                          * ind2 : Production Process 
    #                       ???????      --> slv    : Solvent Use
    #                       ???????      --> flr    : Extraction and distribution of fossil
    #                       TRANSPORT    --> tra    : Road transport
    #                       AIR,SHIPS    --> shp    : Other transport ***** Treated here *****
    #                       ???????      --> wst    : Waste treatment and disposal
    #
    #          ____OUTPUT____
    #
    #            "emis_dict": dictionary : emission on WRF grid
    #            keynames   : 'shp_VOC'
    #            unit       : [mol/sec/m2]   
    #            example    :
    #               emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
    #               emis_dict[keyname]['dimensions']={}
    #               emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
    #               emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
    #               # emis_dict[keyname]['dimensions']['time']       =<# of points in time series> e.g.:12
    #               emis_dict[keyname]['west_east']={}
    #               emis_dict[keyname]['west_east']['dtype']='i4'
    #               emis_dict[keyname]['west_east']['dims' ]=['west_east']
    #               emis_dict[keyname]['west_east']['units']=''
    #               emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
    #               emis_dict[keyname]['south_north']={}
    #               emis_dict[keyname]['south_north']['dtype']='i4'
    #               emis_dict[keyname]['south_north']['dims' ]=['south_north']
    #               emis_dict[keyname]['south_north']['units']=''
    #               emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
    #               emis_dict[keyname]['longitude']={}
    #               emis_dict[keyname]['longitude']['dtype']='f4'
    #               emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['longitude']['units']='degrees_east'
    #               emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
    #               emis_dict[keyname]['latitude']={}
    #               emis_dict[keyname]['latitude']['dtype']='f4'
    #               emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['latitude']['units']='degrees_east'
    #               emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
    #               # emis_dict[keyname]['time']={}
    #               # emis_dict[keyname]['time']['dtype']='i4'
    #               # emis_dict[keyname]['time']['dims' ]=['time']
    #               # emis_dict[keyname]['time']['data' ]=np.arange(12) : [0,1,2,3,4,...,11]
    #               emis_dict[keyname]['voc']={}
    #               emis_dict[keyname]['voc']['dtype']='f4'
    #               # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] or ['west_east','south_north']
    #               emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['voc']['units']='mol/sec/m2'
    #               emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
    #                    
    #====================================================================
    #--------------------------------------------------------------------
    #
    #if 1 == 0:
    if HTAP_SHIP_AIR_VOC:
      year      = '2010'
      htap_dir  ='/data/raut/EMISSIONS/HTAP_v2/'+year+'/'
      htap_voc_files = glob.glob(htap_dir+'edgar_HTAP_NMVOC_emi_*_'+year+'.0.1x0.1.nc')
      if rank == 0:
        print(datetime.now(), htap_voc_files)
      # TEST
      #sectors = ['AIR']
      sectors   = ['AIR','SHIPS']
      #  
      for sec in sectors:
        htap_voc_filenames = [ x for x in htap_voc_files if sec in x ] 
        if rank == 0:
          print(datetime.now(), htap_voc_filenames)
        if len(htap_voc_filenames) == 0:
          continue
        #
        nc        = Dataset(htap_voc_filenames[0],'r',format='NETCDF4')
        lon       = np.array(nc.variables["lon"])
        nc.close()
        if np.abs(np.abs(lon[1]-lon[0])-0.1) > 0.001:
          sys.exit("## check the resolution of emission data :"+htap_voc_filenames[0])
        #
        # ADD : 'shp-HTAP-VOC', etc...
        # UNIT: [kg/m2/sec]-->[kg/m2/sec]  
        #
        emis_dict = AddDict_HTAP_Anth_VOC_OnWRF(emis_dict,htap_voc_filenames,\
                                area_common_array01, area_common_dict01, emis_cell_area01,\
                                XLON,XLAT)
      # END OF loop on sectors
      ## sorted_key_list = [] 
      ## for key in emis_dict:
      ##   sorted_key_list.append(key)
      ## #  plot_emis_dict(key)
      ## for ikey, key in enumerate(sorted(sorted_key_list)):
      ##   print ikey, key
      ## # raw_input()
      ## 
      ## for key in emis_dict:
      ##   print key
      ##   data = np.array(emis_dict[key]['voc']['data'])
      ##   if np.amax(data) != 0.0:
      ##     print key
      ##     print data.shape
      ##     plot_emis_dict(key,0)
      ## raw_input('check after Speciate_SAPRC_ECLIPSE_Anth_VOC_OnWRF')
      #
      #.............................
      #
      #  Speciate FROM ['shp-HTAP-VOC']
      #             TO <sec>-HTAP-<spec>
      # 
      #           WHERE
      #            <sec> = ['shp']
      #   CBMZ :   <spec>= ['C2H5OH','CH3OH','HC5','ETH','TOL','OL2','KET','XYL','HCHO','OLT','ALD','ISO','OLI']
      #   SAPRC:   <spec>= ['C2H6','C3H8','C2H2','ALK3','ALK4','ALK5','ETHENE','C3H6','OLE1','OLE2','ARO1','ARO2','CCHO','HCHO','ACET','MEK','TERP','MEOH','PROD2']
      #
      #           UNIT: [kg/m2/sec] --> [mol/m2/sec]
      #..............................
      #
      # 
      #emis_dict = Speciate_CBMZ_Anth_VOC_OnWRF(emis_dict)
      emis_dict = Speciate_SAPRC_Anth_VOC_OnWRF(emis_dict)
      #
      key_del_list = []
      for key in emis_dict:
        if 'VOC' in key:
          key_del_list.append(key)
      for key in key_del_list:
        #print 'key to be deleted : ', key
        del emis_dict[key]
      #
      #sorted_key_list = [] 
      #for key in emis_dict:
      #  sorted_key_list.append(key)
      ##  plot_emis_dict(key)
      #for ikey, key in enumerate(sorted(sorted_key_list)):
      #  print 'sorted key : ', ikey, key
      ## raw_input()
      #
      #for key in emis_dict:
      #  print key
      #  data = np.array(emis_dict[key]['voc']['data'])
      #  if np.amax(data) != 0.0:
      #    print key
      #    print data.shape
      #    plot_emis_dict(key,0)
      #raw_input('check after Speciate_SAPRC_ECLIPSE_Anth_VOC_OnWRF')
    
    #==========================================================================
    #==========================================================================
    #
    #
    #	* HTAP Anthropogenic Emissions (AIR, SHIPS) 
    #          ____INPUT____
    #          - /data/raut/EMISSIONS/HTAP_v2/<year>/edgar_HTAP_<spec>_emi_<sec>_<year>_<month>.0.1x0.1.nc
    #            <year>    : '2008' or '2010'
    #            <spec>    : ['BC','CO','NOx','OC'->'OM','PM10','PM25','SO2']
    #            <sec>     : ['AIR','SHIPS']
    #            variable  : 'emi_<spec_lower>'
    #            resolution: 0.1 deg x 0.1 deg
    #            dimension : [lat,lon] (1800,3600) (-89.95<lat<89.95, 0.05<lon<359.95)
    #            unit      : [kg/m2/sec]
    #
    #            Sectors  : AGRICULTURE  --> awb    : Agriculture (waste burning on field)
    #                       RESIDENTIAL  --> dom    : Residential and commercial
    #                       ENERGY       --> ene    : Power plants, energy conversion, extraction 
    #                       INDUSTRY     --> ind    : Industry (combustion and processing)
    #                          * ind1 : Industrial (Stationary Combustion)
    #                          * ind2 : Production Process 
    #                       ???????      --> slv    : Solvent Use
    #                       ???????      --> flr    : Extraction and distribution of fossil
    #                       TRANSPORT    --> tra    : Road transport
    #                       AIR,SHIPS    --> shp    : Other transport ***** Treated here *****
    #                       ???????      --> wst    : Waste treatment and disposal
    #
    #          ____OUTPUT____
    #
    #            "emis_dict": dictionary : emission on WRF grid
    #            keynames   : 'shp_<spec>'
    #            keynames   : <sec>_<spec>
    #                 <sec>   : 'shp'
    #                 <spec>  : ['BC','CO','NOx','OC'->'OM','PM10','PM25','SO2']
    #            unit       : [mol/year/km2] or [ug/year/m2]('OM','BC','PM25')
    #            example    :
    #               emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
    #               emis_dict[keyname]['dimensions']={}
    #               emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
    #               emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
    #               # emis_dict[keyname]['dimensions']['time']       =<# of points in time series> e.g.:12
    #               emis_dict[keyname]['west_east']={}
    #               emis_dict[keyname]['west_east']['dtype']='i4'
    #               emis_dict[keyname]['west_east']['dims' ]=['west_east']
    #               emis_dict[keyname]['west_east']['units']=''
    #               emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
    #               emis_dict[keyname]['south_north']={}
    #               emis_dict[keyname]['south_north']['dtype']='i4'
    #               emis_dict[keyname]['south_north']['dims' ]=['south_north']
    #               emis_dict[keyname]['south_north']['units']=''
    #               emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
    #               emis_dict[keyname]['longitude']={}
    #               emis_dict[keyname]['longitude']['dtype']='f4'
    #               emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['longitude']['units']='degrees_east'
    #               emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
    #               emis_dict[keyname]['latitude']={}
    #               emis_dict[keyname]['latitude']['dtype']='f4'
    #               emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['latitude']['units']='degrees_east'
    #               emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
    #               # emis_dict[keyname]['time']={}
    #               # emis_dict[keyname]['time']['dtype']='i4'
    #               # emis_dict[keyname]['time']['dims' ]=['time']
    #               # emis_dict[keyname]['time']['data' ]=np.arange(12) : [0,1,2,3,4,...,11]
    #               emis_dict[keyname]['voc']={}
    #               emis_dict[keyname]['voc']['dtype']='f4'
    #               # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] or ['west_east','south_north']
    #               emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['voc']['units']='g/m2/sec' ('CO','NH3','NOx','SO2') or 'ug/m2/sec' ('OC','PM25','BC')
    #               emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
    #                    
    #====================================================================
    #--------------------------------------------------------------------
    #
    #  Open Edgar HTAP emission data file  'edgar_HTAP_<spec>_emi_<sec>_<year>_<month>.0.1x0.1.nc'
    #
    #if 1 == 0:
    if HTAP_SHIP_AIR_Other:
      year      = '2010'
      spec      = ['BC','CO','NOx','OC','PM10','PM25','SO2']
      # TEST
      #spec      = ['BC','CO']
      for ispec, spec_temp in enumerate(spec):
        htap_dir   ='/data/raut/EMISSIONS/HTAP_v2/'+year+'/'
        htap_files = glob.glob(htap_dir+'edgar_HTAP_'+spec_temp+'_emi_*_'+year+'.0.1x0.1.nc')
        sectors   = ['AIR','SHIPS']
        #  
        for sec in sectors:
          htap_filenames = [ x for x in htap_files if sec in x ] 
          if len(htap_filenames) == 0:
            continue
          #
          nc        = Dataset(htap_filenames[0],'r',format='NETCDF4')
          lon       = np.array(nc.variables["lon"])
          nc.close()
          if np.abs(np.abs(lon[1]-lon[0])-0.1) > 0.001:
            sys.exit("check the resolution of emission data :"+htap_filenames[0])
          #
          # ADD : 'shp_<spec>'.
          # UNIT: [kg/m2/sec]-->[ug/m2/sec] ('OC','PM25','BC') or [mol/m2/sec] 
          #
          emis_dict = AddDict_HTAP_Anth_Other_OnWRF(emis_dict,htap_filenames,\
                                  area_common_array01,area_common_dict01,emis_cell_area01,\
                                  XLON,XLAT,spec_temp)
        # END OF loop on sectors
      # END OF for ispec, spec_temp in enumerate(spec):
      ## sorted_key_list = [] 
      ## for key in emis_dict:
      ##   sorted_key_list.append(key)
      ## #  plot_emis_dict(key)
      ## for ikey, key in enumerate(sorted(sorted_key_list)):
      ##   print ikey, key
      ## # raw_input()
      ## 
      ## for key in emis_dict:
      ##   print key
      ##   data = np.array(emis_dict[key]['voc']['data'])
      ##   if np.amax(data) != 0.0:
      ##     print key
      ##     print data.shape
      ##     plot_emis_dict(key)
      ## raw_input('check in HTAP_SHIP_AIR_Other')
      #
        
    
    #==========================================================================
    #==========================================================================
    #
    #
    #	* ECLIPSE Anthropogenic Emissions (VOC) 
    #          ____INPUT____
    #          xxx - /data/quennehen/ECLIPSE/Anth/CP_WEO_2011_UPD_VOC_<year>.nc
    #          - /data/marelle/EMISSIONS/ECLIPSE_v5/ETP_base_CLE_V5_VOC_<year>.nc
    #            variable  : 'emis_awb','emis_dom','emis_ene','emis_ind',
    #                        'emis_slv','emis_tra','emis_wst','emis_all'
    #            resolution: 0.5 deg x 0.5 deg
    #            dimension : [lat,lon] (360,720)
    #            unit      : [kt/year]
    #
    #            Sectors  : awb    : Agriculture (waste burning on field)
    #                       dom    : Residential and commercial
    #                       ene    : Power plants, energy conversion, extraction 
    #                       ind    : Industry (combustion and processing)
    #                          * ind1 : Industrial (Stationary Combustion)
    #                          * ind2 : Production Process 
    #                       slv    : Solvent Use
    #                       flr    : Extraction and distribution of fossil
    #                       tra    : Road transport
    #                       shp    : Other transport
    #                       wst    : Waste treatment and disposal
    #
    #          ____OUTPUT____
    #
    #            "emis_dict": dictionary : emission on WRF grid
    #            keynames   : 'awb-ECLIPSE-VOC','dom-ECLIPSE-VOC','ene-ECLIPSE-VOC','ind-ECLIPSE-VOC','slv-ECLIPSE-VOC','tra-ECLIPSE-VOC','wst-ECLIPSE-VOC','all-ECLIPSE-VOC'
    #            unit       : [mol/year/km2]   
    #            example    :
    #               emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
    #               emis_dict[keyname]['dimensions']={}
    #               emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
    #               emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
    #               # emis_dict[keyname]['dimensions']['time']       =<# of points in time series> e.g.:12
    #               emis_dict[keyname]['west_east']={}
    #               emis_dict[keyname]['west_east']['dtype']='i4'
    #               emis_dict[keyname]['west_east']['dims' ]=['west_east']
    #               emis_dict[keyname]['west_east']['units']=''
    #               emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
    #               emis_dict[keyname]['south_north']={}
    #               emis_dict[keyname]['south_north']['dtype']='i4'
    #               emis_dict[keyname]['south_north']['dims' ]=['south_north']
    #               emis_dict[keyname]['south_north']['units']=''
    #               emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
    #               emis_dict[keyname]['longitude']={}
    #               emis_dict[keyname]['longitude']['dtype']='f4'
    #               emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['longitude']['units']='degrees_east'
    #               emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
    #               emis_dict[keyname]['latitude']={}
    #               emis_dict[keyname]['latitude']['dtype']='f4'
    #               emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['latitude']['units']='degrees_east'
    #               emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
    #               # emis_dict[keyname]['time']={}
    #               # emis_dict[keyname]['time']['dtype']='i4'
    #               # emis_dict[keyname]['time']['dims' ]=['time']
    #               # emis_dict[keyname]['time']['data' ]=np.arange(12) : [0,1,2,3,4,...,11]
    #               emis_dict[keyname]['voc']={}
    #               emis_dict[keyname]['voc']['dtype']='f4'
    #               # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] or ['west_east','south_north']
    #               emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['voc']['units']='mol/year/km2'
    #               emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
    #                    
    #====================================================================
    #--------------------------------------------------------------------
    #
    #  Open CP_WEO_2011_UPD emission data file  'CP_WEO_2011_UPD_VOC_'+year_str+'.nc'
    #
    #if 1 == 0:
    if ECLIPSE_ANTH_VOC:
      #path2iiasa='/data/quennehen/ECLIPSE/Anth/'
      #fname_voc = path2iiasa+'CP_WEO_2011_UPD_VOC_'+year_str+'.nc'
      #path2iiasa='/data/marelle/EMISSIONS/ECLIPSE_v5/'
      #fname_voc = path2iiasa+'ETP_base_CLE_V5_VOC_'+year_str+'.nc'
      path2iiasa='/data/onishi/ECLIPSE_V6b/'
      fname_voc     = path2iiasa+'ETP_base_CLE_V6_VOC_'+year_str+'.nc'
      fn_month_part = path2iiasa+'ECLIPSE_V6a_monthly_pattern.nc'
      #
      nc        = Dataset(fname_voc,'r',format='NETCDF4')
      lon       = np.array(nc.variables["lon"])
      nc.close()
      if lon[1]-lon[0] != 0.5:
        sys.exit("check the resolution of emission data :"+fname_voc)
      #
      # ADD : 'awb-ECLIPSE-VOC','dom-ECLIPSE-VOC','ene-ECLIPSE-VOC','ind-ECLIPSE-VOC', etc...
      # UNIT: [kt/year] --> [kg/sec/m2]
      #
      ### emis_dict = AddDict_ECLIPSE_Anth_VOC_OnWRF(emis_dict,fname_voc,\
      ###                         area_common_array05,area_common_dict05,emis_cell_area05,\
      ###                         XLON,XLAT)
      emis_dict = AddDict_ECLIPSE_Anth_VOC_OnWRF_v2(emis_dict,fname_voc,fn_month_part, \
                              area_common_array05,area_common_dict05,emis_cell_area05,\
                              XLON,XLAT)
      #
      #.............................
      #
      #  Speciate FROM ['dom-ECLIPSE-VOC','ene-ECLIPSE-VOC','ind-ECLIPSE-VOC','slv-ECLIPSE-VOC','tra-ECLIPSE-VOC','wst-ECLIPSE-VOC']
      #             TO <sec>-ECLIPSE-<spec>
      # 
      #           WHERE
      #            <sec> = ['ene','dom','ind1','ind2','slv','tra','wst']
      #   CBMZ :   <spec>= ['C2H5OH','CH3OH','HC5','ETH','TOL','OL2','KET','XYL','HCHO','OLT','ALD','ISO','OLI']
      #   SAPRC:   <spec>= ['C2H6','C3H8','C2H2','ALK3','ALK4','ALK5','ETHENE','C3H6','OLE1','OLE2','ARO1','ARO2','CCHO','HCHO','ACET','MEK','TERP','MEOH','PROD2']
      #
      #           UNIT: [kg/sec/m2] --> [mol/sec/m2]
      #..............................
      #
      # 
      #emis_dict = Speciate_CBMZ_Anth_VOC_OnWRF(emis_dict)
      emis_dict = Speciate_SAPRC_Anth_VOC_OnWRF(emis_dict)
      #
      key_del_list = []
      for key in emis_dict:
        if 'VOC' in key:
          key_del_list.append(key)
      for key in key_del_list:
        del emis_dict[key]
      
      ## sorted_key_list = [] 
      ## for key in emis_dict:
      ##   sorted_key_list.append(key)
      ## #  plot_emis_dict(key)
      ## for ikey, key in enumerate(sorted(sorted_key_list)):
      ##   print ikey, key
      ## # raw_input()
      ## 
      ## for key in emis_dict:
      ##   data = np.array(emis_dict[key]['voc']['data'])
      ##   if np.amax(data) != 0.0:
      ##     print key
      ##     print data.shape
      ##     plot_emis_dict(key)
      ## raw_input('check after Speciate_SAPRC_ECLIPSE_Anth_VOC_OnWRF')
    
    #==========================================================================
    #
    #
    #	* ECLIPSE Anthropogenic Emissions (Others) 
    #          ____INPUT____
    #          xxx - /data/quennehen/ECLIPSE/Anth/CP_WEO_2011_UPD_<spec>_<year>.nc
    #          - /data/marelle/EMISSIONS/ECLIPSE_v5/ETP_base_CLE_V5_<spec>_<year>.nc
    #            <spec>  = ['CO','CH4','BC','OM','SO2','NH3','PM25','NOx']
    #            variable:
    #               'CO'  : 'lat','lon',           'emis_awb','emis_dom','emis_ene',           'emis_ind','emis_tra','emis_wst','emis_all'
    #               'CH4' : 'lat','lon','emis_agr','emis_awb',,emis_dom','emis_ene','emis_flr','emis_ind','emis_tra','emis_wst','emis_all'
    #               'BC'  : 'lat','lon',           'emis_awb','emis_dom','emis_ene',           'emis_ind','emis_tra','emis_wst','emis_all'
    #               'OM'  : 'lat','lon',           'emis_awb','emis_dom','emis_ene',           'emis_ind','emis_tra','emis_wst','emis_all'
    #               'SO2' : 'lat','lon',           'emis_awb','emis_dom','emis_ene',           'emis_ind','emis_tra','emis_wst','emis_all'
    #               'NH3' : 'lat','lon','emis_agr','emis_awb','emis_dom','emis_ene',           'emis_ind','emis_tra','emis_wst','emis_all'
    #               'PM25': 'lat','lon','emis_agr','emis_awb','emis_dom','emis_ene',           'emis_ind','emis_tra','emis_wst','emis_all'
    #               'NOx' : 'lat','lon',           'emis_awb','emis_dom','emis_ene',           'emis_ind','emis_tra','emis_wst','emis_all'
    #            resolution: 0.5 deg x 0.5 deg
    #            dimension : [lat,lon] (360,720)
    #            unit      : [kt/year]
    #
    #            Sectors  : awb    : Agriculture (waste burning on field)
    #                       dom    : Residential and commercial
    #                       ene    : Power plants, energy conversion, extraction 
    #                       ind    : Industry (combustion and processing)
    #                          * ind1 : Industrial (Stationary Combustion)
    #                          * ind2 : Production Process 
    #                       slv    : Solvent Use
    #                       flr    : Extraction and distribution of fossil
    #                       tra    : Road transport
    #                       shp    : Other transport
    #                       wst    : Waste treatment and disposal
    #
    #          ____OUTPUT____
    #
    #            "emis_dict": dictionary : emission on WRF grid
    #            keynames   : <sec>-ECLIPSE-<spec>
    #                 <sec>   : 'agr','awb','dom','ene','flr','ind','tra','wst','all'
    #                 <spec>  : ['CO','CH4','BC','OM','SO2','NH3','PM25','NOx']
    #            unit       : [mol/sec/m2] or [ug/sec/m2]('OM','BC','PM25')
    #            example    :
    #               emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
    #               emis_dict[keyname]['dimensions']={}
    #               emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
    #               emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
    #               # emis_dict[keyname]['dimensions']['time']       =<# of points in time series> e.g.:12
    #               emis_dict[keyname]['west_east']={}
    #               emis_dict[keyname]['west_east']['dtype']='i4'
    #               emis_dict[keyname]['west_east']['dims' ]=['west_east']
    #               emis_dict[keyname]['west_east']['units']=''
    #               emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
    #               emis_dict[keyname]['south_north']={}
    #               emis_dict[keyname]['south_north']['dtype']='i4'
    #               emis_dict[keyname]['south_north']['dims' ]=['south_north']
    #               emis_dict[keyname]['south_north']['units']=''
    #               emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
    #               emis_dict[keyname]['longitude']={}
    #               emis_dict[keyname]['longitude']['dtype']='f4'
    #               emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['longitude']['units']='degrees_east'
    #               emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
    #               emis_dict[keyname]['latitude']={}
    #               emis_dict[keyname]['latitude']['dtype']='f4'
    #               emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['latitude']['units']='degrees_east'
    #               emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
    #               # emis_dict[keyname]['time']={}
    #               # emis_dict[keyname]['time']['dtype']='i4'
    #               # emis_dict[keyname]['time']['dims' ]=['time']
    #               # emis_dict[keyname]['time']['data' ]=np.arange(12) : [0,1,2,3,4,...,11]
    #               emis_dict[keyname]['voc']={}
    #               emis_dict[keyname]['voc']['dtype']='f4'
    #               # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] or ['west_east','south_north']
    #               emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['voc']['units']='mol/sec/m2' or 'ug/sec/m2'
    #               emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
    #                    
    #====================================================================
    #--------------------------------------------------------------------
    #
    #  Open CP_WEO_2011_UPD emission data file  'CP_WEO_2011_UPD_VOC_'+year_str+'.nc'
    #
    #if 1 == 0:
    if ECLIPSE_ANTH_Other:
      ### fnames = glob.glob('/data/quennehen/ECLIPSE/Anth/CP_WEO_2011_UPD_*_'+year_str+'.nc')
      ### fnames = glob.glob('/data/marelle/EMISSIONS/ECLIPSE_v5/ETP_base_CLE_V5_*_'+year_str+'.nc')
      fnames        = glob.glob('/data/onishi/ECLIPSE_V6b/ETP_base_CLE_V6_*_'+year_str+'.nc')
      fn_month_part = '/data/onishi/ECLIPSE_V6b/ECLIPSE_V6a_monthly_pattern.nc'
      for ifile, fn in enumerate(fnames): 
        #
        nc        = Dataset(fn,'r',format='NETCDF4')
        lon       = np.array(nc.variables["lon"])
        nc.close()
        if lon[1]-lon[0] != 0.5:
          sys.exit("check the resolution of emission data :"+fn)
        #
        # ADD  : keynames   : <sec>-ECLIPSE-<spec>
        #           <sec>   : 'agr','awb','dom','ene','flr','ind','tra','wst','all'
        #           <spec>  : ['CO','CH4','BC','OM','SO2','NH3','PM25','NOx']
        #        unit       : [kt/year] --> [mol/sec/m2] or [ug/sec/m2]
        #
        ### emis_dict = AddDict_ECLIPSE_Anth_Other_OnWRF(emis_dict,fn,\
        ###                         area_common_array05,area_common_dict05,emis_cell_area05,\
        ###                         XLON,XLAT)
        emis_dict = AddDict_ECLIPSE_Anth_Other_OnWRF_v2(emis_dict,fn, fn_month_part, \
                                area_common_array05,area_common_dict05,emis_cell_area05,\
                                XLON,XLAT)
        #
      #
      ### if rank == 0:
      ###   sorted_key_list = [] 
      ###   for key in emis_dict:
      ###   #  print key
      ###     sorted_key_list.append(key)
      ###   #  plot_emis_dict(key)
      ###   for key in sorted(sorted_key_list):
      ###     print key, (emis_dict[key]['voc']['data']).shape
      ###   #raw_input()
      ###   print 'plot_emis_dict'
      ###   plot_emis_dict('ind1_NO2')
      ###   raw_input()
      
    #==========================================================================
    #
    #
    #	* Huang BC Emissions from Russia  
    #          ____INPUT____
    #            .................................................................
    #            Emission grid : longitude : -180 to 360 with 0.1 deg resolution
    #                            latitude  :  -90 to  90 with 0.1 deg resolution
    #            Surface             : Surface area (unit: m2) of Emission grid
    #            lon_glob01          : 1D longitude points :(5400,)  : -179.95:359.95:0.1
    #            lat_glob01          : 1D latitude points  :(1800,)  :  -89.95: 89.95:0.1
    #            lon_bound_glob01    : 1D longitude points :(5401,)  : -180.00:360.00:0.1
    #            lat_bound_glob01    : 1D latitude points  :(1801,)  :  -90.00: 90.00:0.1
    #            lon_glob01_2D       : 2D longitude points :(5400,1800): np.repeat(lon_glob,1800).reshape(5400,1800)
    #            lat_glob01_2d       : 2D latitude points  :(5400,1800): np.tile(  lat_glob,5400).reshape(5400,1800)
    #            lon_bound_glob01_2D : 2D longitude points :(5401,1801): np.repeat(lon_glob,1801).reshape(5401,1801)
    #            lat_bound_glob01_2D : 2D latitude points  :(5401,1801): np.tile(  lat_glob,5401).reshape(5401,1801)
    #
    #          - /data/onishi/BC_RUS/Huang/RUS_BC_2010_Huang.nc
    #            variable:
    #               'lat(lat)'                   : Latitude [degrees_north]  
    #               'lon(lon)'                   : Longitude [degrees_east] 
    #               'cell_area(lat,lon)'         : Area of grid cell [m2]   
    #               'RUS_BC_FLARE(lat,lon)'      : Emissions rate of black carbon from gas flaring    in Russia in year 2010 
    #               'RUS_BC_INDUSTRY(lat,lon)'   : Emissions rate of black carbon from industries     in Russia in year 2010
    #               'RUS_BC_RESIDENTIAL(lat,lon)': Emissions rate of black carbon from residential    in Russia in year 2010
    #               'RUS_BC_TRANSPORT(lat,lon)'  : Emissions rate of black carbon from road transport in Russia in year 2010
    #                                              (*) non-road transportation BC emissions are not included.
    #            resolution: 0.1 deg x 0.1 deg
    #            dimension : [lat,lon] (1800,3600)
    #            unit      : [kg/m2/sec]
    #
    #          ____OUTPUT____
    #
    #            "emis_dict": dictionary : emission on WRF grid
    #            keynames   : <sec>-Huang-BC
    #                 <sec>   : 'dom','ene','flr','ind','tra','all'
    #            unit       : [ug/m2/sec]  
    #            example    :
    #               emis_dict[keyname]={}                                  # Create a dictionary entry with a keynameiable name 'keyname'
    #               emis_dict[keyname]['dimensions']={}
    #               emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
    #               emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
    #               # emis_dict[keyname]['dimensions']['time']       =<# of points in time series> e.g.:12
    #               emis_dict[keyname]['west_east']={}
    #               emis_dict[keyname]['west_east']['dtype']='i4'
    #               emis_dict[keyname]['west_east']['dims' ]=['west_east']
    #               emis_dict[keyname]['west_east']['units']=''
    #               emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
    #               emis_dict[keyname]['south_north']={}
    #               emis_dict[keyname]['south_north']['dtype']='i4'
    #               emis_dict[keyname]['south_north']['dims' ]=['south_north']
    #               emis_dict[keyname]['south_north']['units']=''
    #               emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
    #               emis_dict[keyname]['longitude']={}
    #               emis_dict[keyname]['longitude']['dtype']='f4'
    #               emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['longitude']['units']='degrees_east'
    #               emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
    #               emis_dict[keyname]['latitude']={}
    #               emis_dict[keyname]['latitude']['dtype']='f4'
    #               emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['latitude']['units']='degrees_east'
    #               emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
    #               # emis_dict[keyname]['time']={}
    #               # emis_dict[keyname]['time']['dtype']='i4'
    #               # emis_dict[keyname]['time']['dims' ]=['time']
    #               # emis_dict[keyname]['time']['data' ]=np.arange(12) : [0,1,2,3,4,...,11]
    #               emis_dict[keyname]['voc']={}
    #               emis_dict[keyname]['voc']['dtype']='f4'
    #               # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] or ['west_east','south_north']
    #               emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['voc']['units']='ug/sec/m2'
    #               emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
    #                    
    #====================================================================
    #--------------------------------------------------------------------
    #
    if HUANG_BC_EMISSION:
      ### #========================================================
      ### #
      ### #   Emission Grid (cell corner, 1D)
      ### #   "lat_bound_glob01" : 181 points : -90: 90:0.1 deg.
      ### #   "lon_bound_glob01" : 541 points :-180:360:0.1 deg. 
      ### #
      ### #--------------------------------------------------------
      ### 
      ### lat_bound_glob01 = np.linspace( -90., 90., 1801)
      ### lon_bound_glob01 = np.linspace(-180.,360., 5401)
    
      ### #========================================================
      ### #
      ### #   Emission Grid (Cell Center, 1D)
      ### #   "lat_glob01" : 180 points :  -89.95: 89.95:0.1
      ### #   "lon_glob01" : 540 points : -179.95:359.95:0.1
      ### #
      ### #--------------------------------------------------------
      ### #
      ### lat_glob01 = np.linspace( -89.95, 89.95,1800)
      ### lon_glob01 = np.linspace(-179.95,359.95,5400)
    
      ### #========================================================
      ### #
      ### #   Emission Grid (Cell Center, 2D) 
      ### #   "lat_glob01_2D" : (5400,1800)~"lat_glob01"
      ### #   "lon_glob01_2D" : (5400,1800)~"lon_glob01"
      ### #
      ### #--------------------------------------------------------
      ### #
      ### lat_glob01_2D = np.tile(  lat_glob01,5400).reshape(5400,1800)
      ### lon_glob01_2D = np.repeat(lon_glob01,1800).reshape(5400,1800)
      ### #
      ### #========================================================
      ### #
      ### #   Emission Grid (Cell Corner, 2D) 
      ### #   "lat_bound_glob01_2D" : (5401,1801)~"lat_bound_glob01"
      ### #   "lon_bound_glob01_2D" : (5401,1801)~"lon_bound_glob01"
      ### #
      ### #--------------------------------------------------------
      ### #
      ### lat_bound_glob01_2D = np.tile(  lat_bound_glob01,5401).reshape(5401,1801)
      ### lon_bound_glob01_2D = np.repeat(lon_bound_glob01,1801).reshape(5401,1801)
      ### #
      ### #--------------------------------------------------------------- 
      ### #  Grid area of [lon_anth,lat_anth] grid 
      ### #---------------------------------------------------------------
      ### #
      ### emis_cell_area01    = np.zeros(shape=(len(lon_glob01),len(lat_glob01)))
      ### 
      ### #
      ### #--------------------------------------------------------------- 
      ### #  Grid area of [lon_anth,lat_anth] grid 
      ### #---------------------------------------------------------------
      ### #
      ### for i,lat0 in enumerate(lat_glob01):
      ###   print 'Calculating cell area of emission grid... [unit:m^2]', i
      ###   for j,lon0 in enumerate(lon_glob01):
      ###     emis_cell_area01[j,i] = areaquad(lon_bound_glob01[j  ],lat_bound_glob01[i  ],\
      ###                                      lon_bound_glob01[j+1],lat_bound_glob01[i+1])
      ### # 
      ### #-------------------------------------------------------------------------------------
      ### #
      ### #  Ratio of common area w.r.t. wrf grids (XLONa,XLATa)
      ### #
      ### area_common_array01 = create_commonarea(0.1,XLONa,XLATa,\
      ###                                         lon_bound_glob01_2D,lat_bound_glob01_2D)
      ### #
      ### #-------------------------------------------------------------------------------------
      #
      #  Check the data resolution
      #
      #          - /data/onishi/BC_RUS/Huang/RUS_BC_2010_Huang.nc
      fn = '/data/onishi/BC_RUS/Huang/RUS_BC_2010_Huang.nc'
      nc        = Dataset(fn,'r',format='NETCDF4')
      lon       = np.array(nc.variables["lon"])
      nc.close()
      ## if lon[1]-lon[0] != 0.1:
      ##   sys.exit("check the resolution of emission data :"+fn)
      #
      #--------------------------------------------------------------------------------------
      # ADD  : keynames   : <sec>_BC
      #           <sec>   : 'dom','flr','ind','tra''all'
      #        unit       : [kg/m2/sec]->[ug/m2/sec]  
      #
      emis_dict = AddDict_Huang_BC_OnWRF(emis_dict,fn,\
                              area_common_array01,area_common_dict01,emis_cell_area01,\
                              XLON,XLAT)
      #
      #
      sorted_key_list = [] 
      for key in emis_dict:
      #  print key
        if 'BC' in key:
          sorted_key_list.append(key)
      #  plot_emis_dict(key)
      if rank == 0:
        print(datetime.now())
        for ikey, key in enumerate(sorted(sorted_key_list)):
          print(ikey, key)
      #raw_input()
      #sys.exit()
      
    #===================================================================================
    #
    #       * RCP60 NMVOC emissions from ship transport
    #          ____INPUT____
    #          - /data/quennehen/ECLIPSE/RCP60_AIR,SHP_2005,2010/
    #              1. IPCC_emissions_RCP60_NMVOC_ships_2005_0.5x0.5_v1_01_03_2010.nc
    #              2. IPCC_emissions_RCP60_NMVOC_ships_2010_0.5x0.5_v1_01_03_2010.nc
    #              3. IPCC_emissions_RCP60_<spec>_ships_2005_0.5x0.5_v1_01_03_2010.nc
    #              4. IPCC_emissions_RCP60_<spec>_ships_2010_0.5x0.5_v1_01_03_2010.nc
    #
    #            <spec>    : 'CO','CH4','BC','OC','SO2','NH3','NO'
    #            
    #            variable  : 'emiss_shp'
    #            resolution: 0.5deg x 0.5 deg
    #            dimension : [time,lat,lon] (12,360,720)
    #            unit      : [kg/m2/sec]
    #
    #          ____OUTPUT____
    #
    #            "emis_dict": dictionary : emission on WRF grid
    #            keynames   : 'shp-RCP60-NMVOC','shp-RCP60-CO','shp-RCP60-CH4','shp-RCP60-BC','shp-RCP60-OC','shp-RCP60-SO2','shp-RCP60-NH3','shp-RCP60-NO'
    #            dimension  : [lon,lat,time] (720,360,12)
    #            unit       : [mol/month/m2] or [ug/m2/sec] (BC & OC)   
    #            example    :
    #               emis_dict[keyname]={}                                  # Create a dictionary entry with a 'shp_NMVOC'iable name ''shp_NMVOC''
    #               emis_dict[keyname]['dimensions']={}
    #               emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
    #               emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
    #               emis_dict[keyname]['dimensions']['time']       = 12        # of points in time series> e.g.:12
    #               emis_dict[keyname]['west_east']={}
    #               emis_dict[keyname]['west_east']['dtype']='i4'
    #               emis_dict[keyname]['west_east']['dims' ]=['west_east']
    #               emis_dict[keyname]['west_east']['units']=''
    #               emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
    #               emis_dict[keyname]['south_north']={}
    #               emis_dict[keyname]['south_north']['dtype']='i4'
    #               emis_dict[keyname]['south_north']['dims' ]=['south_north']
    #               emis_dict[keyname]['south_north']['units']=''
    #               emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
    #               emis_dict[keyname]['longitude']={}
    #               emis_dict[keyname]['longitude']['dtype']='f4'
    #               emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['longitude']['units']='degrees_east'
    #               emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
    #               emis_dict[keyname]['latitude']={}
    #               emis_dict[keyname]['latitude']['dtype']='f4'
    #               emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['latitude']['units']='degrees_east'
    #               emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
    #               emis_dict[keyname]['time']={}
    #               emis_dict[keyname]['time']['dtype']='i4'
    #               emis_dict[keyname]['time']['dims' ]=['time']
    #               emis_dict[keyname]['time']['data' ]=np.arange(12) # [0,1,2,3,4,...,11]
    #               emis_dict[keyname]['voc']={}
    #               emis_dict[keyname]['voc']['dtype']='f4'
    #               emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] 
    #               # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['voc']['units']=[mol/sec/m2] or [ug/m2/sec] (BC & OC)
    #               emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
    #===================================================================================
    
    if ECLIPSE_RCP60_SHP:
      path2ships='/data/quennehen/ECLIPSE/RCP60_AIR,SHP_2005,2010/'
      if int(year_str) >= 2008:
        year_str_RCP60 = '2010'
      elif int(year_str) <= 2007:
        year_str_RCP60 = '2005'
      fname_voc = path2ships+'IPCC_emissions_RCP60_*_ships_'+year_str_RCP60+'_0.5x0.5_v1_01_03_2010.nc'
      filenames = glob.glob(fname_voc)
      #
      for ifilename, filename in enumerate(filenames):
        nc        = Dataset(filename,'r',format='NETCDF4')
        lon       = np.array(nc.variables["lon"])
        nc.close()
        if lon[1]-lon[0] != 0.5:
          sys.exit("check the resolution of emission data :"+fname_voc)
        #
        #-----------------------------------------------------------------------------------
        # ADD  : keynames : 'shp-RCP60-<spec>'
        #         <spec>  : 'VOC','CO','CH4','BC','OC','SO2','NH3','NO'
        #      new unit   : [kg/m2/sec]    
        #        dimension: ['west_east','south_north',12]
        #
        emis_dict = AddDict_RCP60_SHP_OnWRF( emis_dict,  filename,\
                                area_common_array05,area_common_dict05,emis_cell_area05,\
                                XLON,XLAT)
      # END of ifilename, filename ...
      #
      #.............................
      #
      #  Speciate FROM shp-RCP60-NMVOC
      #             TO shp-RCP60-<spec>
      # 
      #           WHERE
      #     SAPRC:<spec> = ['C2H6','C3H8','C2H2','ALK3','ALK4','ALK5','ETHENE','C3H6','OLE2','ARO1','ARO2','ECHO','CCHO','ACET','MEK','TERP','MEOH','PROD2']
      #     CBMZ :<spec> = ['C2H5OH','CH3OH','HC5','ETH','TOL','OL2','KET','XYL','HCHO','OLT','ALD','ISO','OLI']
      #
      #     RCP-RCP60-keynames = ['shp-RCP60-CO','shp-RCP60-CH4','shp-RCP60-BC','shp-RCP60-OC','shp-RCP60-SO2','shp-RCP60-NH3','shp-RCP60-NO']
      #     RCP_Mmolaire = {'CO':28,'CH4':16,'SO2':64,'NH3':17,'NO':30}
      #     sector = 'shp'
      #
      #           UNIT: [kg/m2/sec] --> [mol/m2/sec] or [ug/m2/sec] ('BC','OC')
      #..............................
      #
      #
      emis_dict = Speciate_RCP60_SHP_OnWRF(emis_dict,chem='SAPRC')
      #
      ### for key in emis_dict:
      ###   if 'shp' in key:
      ###     print key, emis_dict[key]['voc']['units']
      ### raw_input()
      ### print 'plot_emis_dict'
      ### plot_emis_dict('shp_BC',1)
      ### raw_input()
    
    
    #===================================================================================
    #
    #       * POLMIP Daily soil NOx surface emissions
    #
    #          ____INPUT____
    #
    #          - /data/quennehen/EMISSIONS/POLMIP/emissions.NO.surface.1x1.nc'
    #
    #            variable  : 'soil'
    #            resolution: 1.0 deg x 1.0 deg
    #            dimension : [time,lat,lon] (367,180,360)
    #            unit      : [molecules/cm2/sec]
    #
    #          ____OUTPUT____
    #
    #            "emis_dict": dictionary : emission on WRF grid
    #            keynames   : 'soil-POLMIP-NO'
    #            dimension  : [lon,lat,time] (180,360,367)
    #            unit       : [mol/m2/sec]   
    #            example    :
    #               emis_dict[keyname]={}                                  # Create a dictionary entry with a 'shp_NMVOC'iable name ''shp_NMVOC''
    #               emis_dict[keyname]['dimensions']={}
    #               emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
    #               emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
    #               emis_dict[keyname]['dimensions']['time']       = 12        # of points in time series> e.g.:12
    #               emis_dict[keyname]['west_east']={}
    #               emis_dict[keyname]['west_east']['dtype']='i4'
    #               emis_dict[keyname]['west_east']['dims' ]=['west_east']
    #               emis_dict[keyname]['west_east']['units']=''
    #               emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
    #               emis_dict[keyname]['south_north']={}
    #               emis_dict[keyname]['south_north']['dtype']='i4'
    #               emis_dict[keyname]['south_north']['dims' ]=['south_north']
    #               emis_dict[keyname]['south_north']['units']=''
    #               emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
    #               emis_dict[keyname]['longitude']={}
    #               emis_dict[keyname]['longitude']['dtype']='f4'
    #               emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['longitude']['units']='degrees_east'
    #               emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
    #               emis_dict[keyname]['latitude']={}
    #               emis_dict[keyname]['latitude']['dtype']='f4'
    #               emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['latitude']['units']='degrees_east'
    #               emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
    #               emis_dict[keyname]['time']={}
    #               emis_dict[keyname]['time']['dtype']='datetime'
    #               emis_dict[keyname]['time']['dims' ]=['time']
    #               emis_dict[keyname]['time']['data' ]=dt[:] # [datetime(2008,1,1),datetime(2008,1,2),....,datetime(2009,1,1)]
    #               emis_dict[keyname]['voc']={}
    #               emis_dict[keyname]['voc']['dtype']='f4'
    #               emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] 
    #               # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['voc']['units']='mol/m2/sec'
    #               emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
    #===================================================================================
    
    if POLMIP_DAILY_SOIL_NO:
      filename  = '/data/quennehen/EMISSIONS/POLMIP/emissions.NO.surface.1x1.nc'
      #
      nc        = Dataset(filename,'r',format='NETCDF4')
      lon       = np.array(nc.variables["lon"])
      nc.close()
      if lon[1]-lon[0] != 1.0:
        sys.exit("check the resolution of emission data :"+filename)
      #
      # ADD  : 'soil-POLMIP-NO'
      #
      # UNIT : [molecules/cm2/sec] --> [mol/sec/m2]
      #
      emis_dict = AddDict_POLMIP_Daily_Soil_NO_OnWRF( emis_dict,  filename,\
                              area_common_array10,area_common_dict10,emis_cell_area10,\
                              XLON,XLAT,start_dt,end_dt)
      # 
      # for key in emis_dict:
      #   if 'soil' in key:
      #     print key
      # raw_input()
      #
      # print 'plot_emis_dict'
      # plot_emis_dict('soil_NO',1)
      # raw_input()
    
    #===================================================================================
    #
    #       * POLMIP Daily Volcanic SO2 emissions
    #
    #          ____INPUT____
    #
    #          - /data/quennehen/EMISSIONS/POLMIP/emissions.SO2.surface.1x1.nc'
    #
    #            variable  : 'volcano'
    #            resolution: 1.0 deg x 1.0 deg
    #            dimension : [time,lat,lon] (367,180,360)
    #            unit      : [molecules/cm2/sec]
    #
    #          ____OUTPUT____
    #
    #            "emis_dict": dictionary : emission on WRF grid
    #            keynames   : 'vol-POLMIP-SO2'
    #            dimension  : [lon,lat,time] (180,360,367)
    #            unit       : [mol/sec/m2]   
    #            example    :
    #               emis_dict[keyname]={}                                  # Create a dictionary entry with a 'shp_NMVOC'iable name ''shp_NMVOC''
    #               emis_dict[keyname]['dimensions']={}
    #               emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
    #               emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
    #               emis_dict[keyname]['dimensions']['time']       = 367       # of points in time series> e.g.:12
    #               emis_dict[keyname]['west_east']={}
    #               emis_dict[keyname]['west_east']['dtype']='i4'
    #               emis_dict[keyname]['west_east']['dims' ]=['west_east']
    #               emis_dict[keyname]['west_east']['units']=''
    #               emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
    #               emis_dict[keyname]['south_north']={}
    #               emis_dict[keyname]['south_north']['dtype']='i4'
    #               emis_dict[keyname]['south_north']['dims' ]=['south_north']
    #               emis_dict[keyname]['south_north']['units']=''
    #               emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
    #               emis_dict[keyname]['longitude']={}
    #               emis_dict[keyname]['longitude']['dtype']='f4'
    #               emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['longitude']['units']='degrees_east'
    #               emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
    #               emis_dict[keyname]['latitude']={}
    #               emis_dict[keyname]['latitude']['dtype']='f4'
    #               emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['latitude']['units']='degrees_east'
    #               emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
    #               emis_dict[keyname]['time']={}
    #               emis_dict[keyname]['time']['dtype']='datetime'
    #               emis_dict[keyname]['time']['dims' ]=['time']
    #               emis_dict[keyname]['time']['data' ]=dt[:] # [datetime(2008,1,1),datetime(2008,1,2),...,datetime(2009,1,1)]
    #               emis_dict[keyname]['voc']={}
    #               emis_dict[keyname]['voc']['dtype']='f4'
    #               emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] 
    #               # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['voc']['units']='mol/sec/m2'
    #               emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
    #===================================================================================
    
    if POLMIP_DAILY_VOLC_SO2:
      filename  = '/data/quennehen/EMISSIONS/POLMIP/emissions.SO2.surface.1x1.nc'
      #
      if rank == 0:
        print(datetime.now(), filename)
      nc        = Dataset(filename,'r',format='NETCDF4')
      lon       = np.array(nc.variables["lon"])
      nc.close()
      if lon[1]-lon[0] != 1.0:
        sys.exit("check the resolution of emission data :"+filename)
      #
      # ADD   : 'vol-POLMIP-SO2'
      # UNIT  : [molecules/cm2/sec] --> [mol/sec/m2]
      # 
      emis_dict = AddDict_POLMIP_Daily_vol_SO2_OnWRF( emis_dict,  filename,\
                              area_common_array10,area_common_dict10,emis_cell_area10,\
                              XLON,XLAT,start_dt,end_dt)
      # 
      # for key in emis_dict:
      #   if 'vol' in key:
      #     print key
      # print 'plot_emis_dict'
      # plot_emis_dict('vol_SO2',1)
      # raw_input()
      #
      #sorted_key_list = [] 
      #for key in emis_dict:
      #  sorted_key_list.append(key)
      ##  plot_emis_dict(key)
      #for ikey, key in enumerate(sorted(sorted_key_list)):
      #  print ikey, key
      #raw_input()
    
    
    #===================================================================================
    #
    #       * Monthly DMS concentration : oceanic values from Lana et al.
    #
    #          ____INPUT____
    #
    #          - /data/marelle/EMISSIONS/DMS_LANA/DMSclim_<month>.csv
    #
    #            <month>   : 'JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'
    #            variable  : 'DMS_OC'
    #            resolution: 1.0 deg x 1.0 deg
    #            dimension : [time,lat,lon] (180,360)
    #            unit      : [mol/cm3]
    #
    #          ____OUTPUT____
    #
    #            "emis_dict": dictionary : emission on WRF grid
    #            keynames   : 'DMS_OC-LANA'
    #            dimension  : [lon,lat,time] (180,360,12)
    #            unit       : [mol/m3]   
    #            example    :
    #               emis_dict[keyname]={}                                  # Create a dictionary entry with a 'shp_NMVOC'iable name ''shp_NMVOC''
    #               emis_dict[keyname]['dimensions']={}
    #               emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
    #               emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
    #               emis_dict[keyname]['dimensions']['time']       = 12        # of points in time series> e.g.:12
    #               emis_dict[keyname]['west_east']={}
    #               emis_dict[keyname]['west_east']['dtype']='i4'
    #               emis_dict[keyname]['west_east']['dims' ]=['west_east']
    #               emis_dict[keyname]['west_east']['units']=''
    #               emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
    #               emis_dict[keyname]['south_north']={}
    #               emis_dict[keyname]['south_north']['dtype']='i4'
    #               emis_dict[keyname]['south_north']['dims' ]=['south_north']
    #               emis_dict[keyname]['south_north']['units']=''
    #               emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
    #               emis_dict[keyname]['longitude']={}
    #               emis_dict[keyname]['longitude']['dtype']='f4'
    #               emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['longitude']['units']='degrees_east'
    #               emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
    #               emis_dict[keyname]['latitude']={}
    #               emis_dict[keyname]['latitude']['dtype']='f4'
    #               emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['latitude']['units']='degrees_east'
    #               emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
    #               emis_dict[keyname]['time']={}
    #               emis_dict[keyname]['time']['dtype']='datetime'
    #               emis_dict[keyname]['time']['dims' ]=['time']
    #               emis_dict[keyname]['time']['data' ]=dt[:] # [1,2,3,4,....,12]
    #               emis_dict[keyname]['voc']={}
    #               emis_dict[keyname]['voc']['dtype']='f4'
    #               emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] 
    #               # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['voc']['units']='mol/m3'
    #               emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
    #===================================================================================
    
    if LANA_MONTHLY_DMS:
      dname  = '/data/marelle/EMISSIONS/DMS_LANA/'
      #
      # ADD   : 'DMS_OC'
      # UNIT  : [mol/cm3] --> [mol/m3]
      # 
      emis_dict = AddDict_LANA_DMS_Ocean_OnWRF( emis_dict,  dname,\
                              area_common_array10,area_common_dict10,emis_cell_area10,\
                              XLON,XLAT,start_dt,end_dt)
       
      ### for key in emis_dict:
      ###   if 'DMS' in key:
      ###     print key
      ### print 'plot_emis_dict'
      ### print (emis_dict[key]['voc']['data']).shape
      ### plot_emis_dict('DMS_OC',1)
      ### raw_input()
      #
      ## sorted_key_list = [] 
      ## for key in emis_dict:
      ##   sorted_key_list.append(key)
      ## #  plot_emis_dict(key)
      ## for ikey, key in enumerate(sorted(sorted_key_list)):
      ##   print ikey, key
      ## raw_input()
   
    
    #===================================================================================
    #
    #       * GFED fire emissions                                        
    #
    #          ____INPUT____
    #
    #          - /data/onishi/GFEDv4/fire_emissions_v4_R1/data/GFED4.1s_<year>.hdf5
    #
    #            <year>    : year
    #            variable  : DMs : Dry Matter burned                    [kg DM/m2/month]
    #                        part_DM[sec] : partitioning of each sector [unitless]
    #                        sec : 'SAVA','BORF','TEMF','DEFO','PEAT','AGRI'
    #                        EFs : (Emission Factor)[spec][sec]         [g <spec>/kg DM]
    #                        grid_cell_area : cell area                 [m2]
    #            resolution: 0.25 deg x 0.25 deg
    #            dimension : [time,lat,lon] (12,720,1440)
    #            unit      : [g <spec>/m2/month]
    #
    #          ____OUTPUT____
    #
    #            "emis_dict": dictionary : emission on WRF grid
    #            keynames   : '<spec>-GFED-YYYYMMDDhh'
    #            dimension  : [lon,lat,time] (720,1440,ntime) (ntime:hourly output from start_dt to end_dt)
    #            unit       : [ug/sec/m2] or [mol/sec/m2]   
    #            example    :
    #               emis_dict[keyname]={}                                  # Create a dictionary entry with a 'shp_NMVOC'iable name ''shp_NMVOC''
    #               emis_dict[keyname]['dimensions']={}
    #               emis_dict[keyname]['dimensions']['south_north']= ng_sn_wrf # of grid points in sn direction 
    #               emis_dict[keyname]['dimensions']['west_east']  = ng_we_wrf # of grid points in we direction
    #               emis_dict[keyname]['dimensions']['time']       = 12        # of points in time series> e.g.:12
    #               emis_dict[keyname]['west_east']={}
    #               emis_dict[keyname]['west_east']['dtype']='i4'
    #               emis_dict[keyname]['west_east']['dims' ]=['west_east']
    #               emis_dict[keyname]['west_east']['units']=''
    #               emis_dict[keyname]['west_east']['data' ]=np.arange(ng_we_wrf) # : [0,1,2,3,4,5,...,ng_we_wrf-1]
    #               emis_dict[keyname]['south_north']={}
    #               emis_dict[keyname]['south_north']['dtype']='i4'
    #               emis_dict[keyname]['south_north']['dims' ]=['south_north']
    #               emis_dict[keyname]['south_north']['units']=''
    #               emis_dict[keyname]['south_north']['data' ]=np.arange(ng_sn_wrf) # : [0,1,2,3,4,5,...,ng_sn_wrf-1]
    #               emis_dict[keyname]['longitude']={}
    #               emis_dict[keyname]['longitude']['dtype']='f4'
    #               emis_dict[keyname]['longitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['longitude']['units']='degrees_east'
    #               emis_dict[keyname]['longitude']['data' ]=XLON[:,:] # <WRF longitude grid>
    #               emis_dict[keyname]['latitude']={}
    #               emis_dict[keyname]['latitude']['dtype']='f4'
    #               emis_dict[keyname]['latitude']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['latitude']['units']='degrees_east'
    #               emis_dict[keyname]['latitude']['data' ]=XLAT[:,:]  # <WRF latitude grid>
    #               emis_dict[keyname]['time']={}
    #               emis_dict[keyname]['time']['dtype']='datetime'
    #               emis_dict[keyname]['time']['dims' ]=['time']
    #               emis_dict[keyname]['time']['data' ]=dt[:] # [1,2,3,4,....,12]
    #               emis_dict[keyname]['voc']={}
    #               emis_dict[keyname]['voc']['dtype']='f4'
    #               emis_dict[keyname]['voc']['dims' ]=['west_east','south_north','time'] 
    #               # emis_dict[keyname]['voc']['dims' ]=['west_east','south_north']
    #               emis_dict[keyname]['voc']['units']='mol/m2/month' or 'ug/m2/month'
    #               emis_dict[keyname]['voc']['data' ]= emis_wrf[:,:,:] # <emissions data on WRF Grid> : e.g.) emis_temp[:<we>,:<sn>,:<time>]
    #===================================================================================
    
    if GFED_FIRE:
      dname  = '/data/onishi/GFEDv4/fire_emissions_v4_R1/data/'
      #
      #
      # ADD   : '<spec>-GFED-YYYYMMDDhh'
      # UNIT  : [mol/m2/month] or [ug/m2/month] --> [mol/m2/sec] or [ug/m2/sec]
      # 

      # <spec_SAPRC>:
      #        ['ACET', 'ALK3', 'ALK4', 'ALK5', \
      #         'ARO1', 'ARO2', 'BACL', 'C2H2', \
      #         'C2H6', 'C3H6', 'C3H8', 'CCHO', \
      #         'CH4' , 'CO'  , 'EC'  , 'ETHENE', \
      #         'HCHO', 'HCOOH', 'HONO', 'MEK', \
      #         'MEOH', 'MVK', 'NH3', 'NO', \
      #         'OLE1', 'OLE2', 'ORGJ', 'PHEN', \
      #         'RCHO', 'SO2', 'TERP', 'CCO_OH', \
      #         'ISOPRENE', 'ISOPROD', 'METHACRO', \
      #         'NO3J', 'PM25J', 'SO4J']
     
      emis_dict = AddDict_GFED_Fire_OnWRF( emis_dict,  dname,\
                              area_common_array025,area_common_dict025,emis_cell_area025,\
                              XLON,XLAT,start_dt,end_dt, ECLIPSEinUse=ECLIPSEinUse)
      
      ### if rank==0: 
      ###   for key in emis_dict:
      ###     if 'GFED' in key:
      ###       print(key)
      ###   print( 'plot_emis_dict')
      ###   print( (emis_dict[key]['voc']['data']).shape)
      ###   #plot_emis_dict('CO_GFED',1)
      ### input()
      #
      ### sorted_key_list = [] 
      ### for key in emis_dict:
      ###   if 'CO-GFED' in key:
      ###     sorted_key_list.append(key)
      ### #  plot_emis_dict(key)
      ### for ikey, key in enumerate(sorted(sorted_key_list)):
      ###   print('check 4 : ', ikey, key, np.array(emis_dict[key]['voc']['data'])[145,54])
      ### sys.exit()
   
    
    #=================================================================================
    #
    #  In case of use of FINN, remove "awb" keys from emis_dict
    #
    #.................................................................................

    ### if not GFED_FIRE:
    ###   key_del_list = []
    ###   for key in emis_dict:
    ###     if 'awb' in key:
    ###       key_del_list.append(key)
    ###   for key in key_del_list:
    ###     #print 'key to be deleted : ', key
    ###     del emis_dict[key]
    
    
    #=================================================================================
    #
    #  INTERPOLATION TO HOURLY OUTPUT 
    #
    #=================================================================================
    if rank == 0:
      print('------------------------------------')
      print('')
      print(datetime.now(),' SECTION : INTERPOLATION TO HOURLY OUTPUT')
      print('')
      print('------------------------------------')
    
    hourly_factors, daily_factors, monthly_factors = Obtain_factors()
    
    ### icount = 0
    ### for keyname in emis_dict:
    ###   if 'sec'.lower() in emis_dict[keyname]['voc']['units'].lower():
    ###     print icount, keyname, emis_dict[keyname]['voc']['units'], np.array(emis_dict[keyname]['voc']['data']).shape
    ###     icount += 1
    ### for keyname in emis_dict:
    ###   if 'day'.lower() in emis_dict[keyname]['voc']['units'].lower():
    ###     print icount, keyname, emis_dict[keyname]['voc']['units'], np.array(emis_dict[keyname]['voc']['data']).shape
    ###     icount += 1
    ### for keyname in emis_dict:
    ###   if 'month'.lower() in emis_dict[keyname]['voc']['units'].lower():
    ###     print icount, keyname, emis_dict[keyname]['voc']['units'], np.array(emis_dict[keyname]['voc']['data']).shape
    ###     icount += 1
    ### for keyname in emis_dict:
    ###   if 'year'.lower() in emis_dict[keyname]['voc']['units'].lower():
    ###     print icount, keyname, emis_dict[keyname]['voc']['units'], np.array(emis_dict[keyname]['voc']['data']).shape
    ###     icount += 1
    ### print icount, len(emis_dict) 
    
    ### sorted_key_list = [] 
    ### for key in emis_dict:
    ###   sorted_key_list.append(key)
    ### #  plot_emis_dict(key)
    ### for ikey, key in enumerate(sorted(sorted_key_list)):
    ###   print ikey, key
    ### raw_input('sorted_key_list  ')
    #-------------------------------------------------------------------------------
    #
    #  Delete keyname of "all"
    #
    #-------------------------------------------------------------------------------
    
    hourly_keynames  = []
    
    keyname_to_delete = []
    for keyname in emis_dict:
      if 'all' in keyname.lower():
        keyname_to_delete.append(keyname)
    
    for keyname in keyname_to_delete:
      del emis_dict[keyname]
    
    # sorted_key_list = [] 
    # for key in emis_dict:
    #   sorted_key_list.append(key)
    # #  plot_emis_dict(key)
    # for ikey, key in enumerate(sorted(sorted_key_list)):
    #   print ikey, key
    # raw_input()

    #--------------------------------------------------------------------------------
    #
    # INTERPOLATION : (lon,lat,12)[monthly] --> (lon,lat)[hourly] 
    # REAS          : e.g.) ene-REAS-C2H6, ind-REAS-CO, etc...
    # HTAP          : except for 'SHIPS' and 'AIR', e.g.) ene-HTAP-C2H2, ind1-HTAP-ALK3, tra-HTAP-CO, etc...
    # RCP60         : shp, e.g.) shp-RCP60-ALK3, shp-RCP60-CO, etc
    # LANA_DMS      : 'DMS_OC-LANA' 
    # 
    # except for GFED: GFED is already in a format of <spec>-GFED-<YYYY-MM-DD_hh:00:00>
    # <spec> of GFED are the following (already WRF-Chem emission variable names)
    # GFED_SAPRC_spec = \
    #              ['ACET', 'ALK3', 'ALK4', 'ALK5', \
    #               'ARO1', 'ARO2', 'BACL', 'C2H2', \
    #               'C2H6', 'C3H6', 'C3H8', 'CCHO', \
    #               'CH4' , 'CO'  , 'EC'  , 'ETHENE', \
    #               'HCHO', 'HCOOH', 'HONO', 'MEK', \
    #               'MEOH', 'MVK', 'NH3', 'NO', \
    #               'OLE1', 'OLE2', 'ORGJ', 'PHEN', \
    #               'RCHO', 'SO2', 'TERP', 'CCO_OH', \
    #               'ISOPRENE', 'ISOPROD', 'METHACRO', \
    #               'NO3J', 'PM25J', 'SO4J']
    #       
    #    
    #
    #---------------------------------------------------------------------------------
    if rank == 0:
      print('------------------------------------')
      print('')
      print(datetime.now(),' SECTION : INTERPOLATION TO HOURLY OUTPUT')
      print('INTERPOLATION : (lon,lat,12)[monthly] --> (lon,lat)[hourly]') 
      print('')
      print('------------------------------------')

    
    keyname_list = []
    for keyname in emis_dict:
      dims = emis_dict[keyname]['voc']['dims']
      if len(dims) == 3:
        ntimes = emis_dict[keyname]['dimensions']['time']
        if ntimes == 12:
          keyname_list.append(keyname)
        #if 'sec'.lower() in emis_dict[keyname]['voc']['units'].lower():
        #  keyname_list.append(keyname)
        #if 'month'.lower() in emis_dict[keyname]['voc']['units'].lower():
        #  keyname_list.append(keyname)
    
    #
    for ikeyname, keyname in enumerate(keyname_list):
      if rank == 0:
        print('INTERPOLATION 01 : ', keyname,' (', ikeyname, '/', len(keyname_list),')')
      sec     = keyname.split('-')[0]
      dt_temp = start_dt
      while dt_temp <= end_dt:
        if rank == 0:
          print('date time : ', start_dt, ' < ', dt_temp, ' < ',end_dt)
        start_tm    = tm.time()
        year        = dt_temp.year
        month       = dt_temp.month
        day         = dt_temp.day
        hour        = dt_temp.hour
        day_of_week = dt_temp.weekday()
        #
        #---- calculate hourly_factors -------------------------------
        #---- Local time (longitude) is taken into account -----------
        #
        if 'DMS_OC' not in keyname:
          hourly_factors_temp = np.zeros_like(XLON)
          hour2_temp = hour2+hour
          hour2_temp[hour2_temp >= 48] -= 48
          hour2_temp[hour2_temp >= 24] -= 24
          for ihour in np.arange(24):
            hourly_factors_temp[hour2_temp == ihour] = hourly_factors[sec][ihour]
        #
        #-------------------------------------------------------------- 
        #
        # ECLIPSE : v2 with ECLIPSE v6a monthly_partition
        # REAS    : BC, OC, PM10 and PM2.5
        # HTAP    : 'BC', 'OC', 'PM25' except for 'SHIPS' and 'AIR'
        # RCP60   : shp for 'BC' and 'OC'
        #
        units = emis_dict[keyname]['voc']['units'].lower()
        # print keyname, year, month, day, hour, units
        # unit : ug/m2/sec
        if '/sec' in units and 'ug' in units and '/m2' in units:
          # unit : ug/m2/sec --> ug/m2/week
          first_day_of_week = monthrange(year,month)[0]
          number_of_days    = monthrange(year,month)[1]
          # emis_hour = np.array(emis_dict[keyname]['voc']['data'])*60.0*60.0*24.0*number_of_days
          #debug#emis_data = np.array(emis_dict[keyname]['voc']['data'])[:,:,month-1]*60.0*60.0*24.0*7.0
          emis_data = np.copy(emis_dict[keyname]['voc']['data'][:,:,month-1]*60.0*60.0*24.0*7.0)
          # unit : ug/m2/week --> ug/m2/day
          emis_data *= daily_factors[sec][day_of_week]
          # unit : ug/m2/day  --> ug/m2/hour
          emis_data *= hourly_factors_temp
          # unit : ug/m2/hour  --> ug/m2/sec
          emis_data /= (60.0*60.0)
          #
          new_keyname = Create_Hourly_Keyname(keyname,dt_temp) 
          hourly_keynames.append(new_keyname)
          Add_EmisData_Dictionary(emis_dict,new_keyname,emis_data,'ug/m2/sec',\
                                  XLON, XLAT )
    
        ### # unit : mol/month/km2 
        ### if 'month' in units and 'mol' in units and 'km2' in units:
        ###   # print units.lower()
        ###   # 
        ###   number_of_days  = monthrange(year,month)[1]
        ###   # unit : mol/month/km2 --> mol/week/km2
        ###   emis_data = np.array(emis_dict[keyname]['voc']['data'])[:,:,month-1]/number_of_days*7.0
        ###   # unit : mol/km2/week --> mol/km2/day
        ###   emis_data = emis_data*daily_factors[sec][day_of_week]
        ###   # unit : mol/km2/day  --> mol/km2/hour
        ###   emis_data = emis_data*hourly_factors[sec][hour]
        ###   # create a new keyname e.g.) shp_CO_2008-06-07_01:00:00
        ###   new_keyname = Create_Hourly_Keyname(keyname,dt_temp) 
        ###   Add_EmisData_Dictionary(emis_dict,new_keyname,emis_data,'mol/km2/hour',\
        ###                           XLON, XLAT )
           
        # unit    : [mol/m2/sec] -> [mol/km2/hour] 
        # ECLIPSE : v2 with ECLIPSE v6a monthly partition
        # REAS    : Anth_VOC, Gas except for 'BC', 'OC', 'PM10' and 'PM2.5'
        # HTAP    : Anth_VOC, Gas except for 'BC', 'OC', 'PM25' except for 'SHIPS' and 'AIR'
        # RCP60   : shp           except for 'BC', 'OC'
        #
        if '/sec' in units and 'mol' in units and '/m2' in units:
          # print units.lower()
          # 
          number_of_days  = monthrange(year,month)[1]
          # unit : mol/m2/sec --> mol/week/km2
          
          #debug#emis_data = np.array(emis_dict[keyname]['voc']['data'])[:,:,month-1]*60.0*60.0*24.0*7.0*1.e6
          emis_data = np.copy(emis_dict[keyname]['voc']['data'][:,:,month-1]*60.0*60.0*24.0*7.0*1.e6)
          # unit : mol/km2/week --> mol/km2/day
          emis_data *= daily_factors[sec][day_of_week]
          # unit : mol/km2/day  --> mol/km2/hour
          emis_data *= hourly_factors_temp
          # create a new keyname e.g.) shp_CO_2008-06-07_01:00:00
          new_keyname = Create_Hourly_Keyname(keyname,dt_temp) 
          hourly_keynames.append(new_keyname)
          Add_EmisData_Dictionary(emis_dict,new_keyname,emis_data,'mol/km2/hour',\
                                  XLON, XLAT )
           
   
        # LANA_DMS : 'DMS_OC-LANA' 
        # unit : mol/m3 
        if '/m3' in units and 'mol' in units:
          # print units.lower()
          # 
          # pick up data of the month 'month'
          #debug#emis_data = np.array(emis_dict[keyname]['voc']['data'])[:,:,month-1]
          emis_data = np.copy(emis_dict[keyname]['voc']['data'][:,:,month-1])
          # create a new keyname e.g.) DMS_OC-2008-06-07_01:00:00
          new_keyname = Create_Hourly_Keyname(keyname,dt_temp) 
          hourly_keynames.append(new_keyname)
          Add_EmisData_Dictionary(emis_dict,new_keyname,emis_data,'mol/m3',\
                                  XLON, XLAT )
        dt_temp += timedelta(hours=int(dhour))
        end_tm = tm.time()
        if rank == 0:
          print('time spent : ', end_tm - start_tm, '// d.t.:',dt_temp,'// keyname:',keyname)
      # END of while dt_temp <= end_dt:
    # END of for keyname in keyname_list:
    
    # Delete unnecessary emis_dict[keyname]
    
    for keyname in keyname_list:
      if rank == 0:
        print(datetime.now(), 'deleting '+keyname+' .....')
      del emis_dict[keyname]
    
    
    #
    #--------
    #
    ## sorted_key_list = [] 
    ## for key in emis_dict:
    ##   sorted_key_list.append(key)
    ## #  plot_emis_dict(key)
    ## for ikey, key in enumerate(sorted(sorted_key_list)):
    ##   print ikey, key, emis_dict[key]['voc']['units'], emis_dict[key]['voc']['dims']
    #--------------------------------------------------------------------------------
    #
    # INTERPOLATION : (lon,lat,367)[daily] --> (lon,lat)[hourly] 
    # POLMIP        : NO SOIL and SO2 VOLCAN 
    # 
    #    Input unit : [mol/m2/sec]
    #    Output unit: [mol/km2/hour]
    #
    #---------------------------------------------------------------------------------
    if rank == 0:
      print('------------------------------------')
      print('')
      print(datetime.now(),' SECTION : INTERPOLATION TO HOURLY OUTPUT')
      print('INTERPOLATION : (lon,lat,367)[daily] --> (lon,lat)[hourly] (POLMIP NO SOIL and SO2 VOLCAN)') 
      print('')
      print('------------------------------------')
    
    keyname_list = []
    for keyname in emis_dict:
      dims = emis_dict[keyname]['voc']['dims']
      if len(dims) == 3:
        ntimes = emis_dict[keyname]['dimensions']['time']
        if ntimes == 367:
          keyname_list.append(keyname)
    #
    for ikeyname, keyname in enumerate(keyname_list):
      if rank == 0:
        print('INTERPOLATION 02 : ', keyname,' (', ikeyname, '/', len(keyname_list),')')
      sec     = keyname.split('-')[0]
      # change sec "soil" to "soi"
      if 'soi' in sec:
        sec   = 'soi'
      dt_temp = start_dt
      while dt_temp <= end_dt:
        doy      = dt_temp.timetuple().tm_yday
        year     = dt_temp.year
        month    = dt_temp.month
        day      = dt_temp.day
        hour     = dt_temp.hour
        day_of_week = dt_temp.weekday()
        units    = emis_dict[keyname]['voc']['units'].lower()
        # print keyname, year, month, day, hour, units, doy
        #debug#emis_data= np.array(emis_dict[keyname]['voc']['data'])[:,:,doy]
        emis_data= np.copy(emis_dict[keyname]['voc']['data'][:,:,doy])
        # unit : mol/m2/sec   --> mol/km2/day
        emis_data *= 1.e6*60.0*60.0*24.0
        # unit : mol/km2/day  --> mol/km2/hour

        ## hour2 = np.rint(XLON/15.0)+hour
        hour2_temp = hour2+hour
        hour2_temp[hour2_temp >= 48] -= 48
        hour2_temp[hour2_temp >= 24] -= 24
        for ihour in np.arange(24):
          emis_data[hour2_temp == ihour] *= hourly_factors[sec][ihour]
        ## for ind, XLON_temp in np.ndenumerate(XLON):
        ##   if XLON_temp < 360.:
        ##     XLON_temp += 360.
        ##   dhour_loc = int(round(XLON_temp/15.0))
        ##   hour2 = hour+dhour_loc
        ##   while hour2 >= 24:
        ##     hour2 -= 24
        ##   emis_data[ind] *= hourly_factors[sec][hour2]
        # 
        new_keyname = Create_Hourly_Keyname(keyname,dt_temp) 
        hourly_keynames.append(new_keyname)
        Add_EmisData_Dictionary(emis_dict,new_keyname,emis_data,'mol/km2/hour',\
                                XLON, XLAT )
        dt_temp += timedelta(hours=int(dhour))
      # END of while dt_temp <= end_dt:
    # END of for keyname in keyname_list:
    
    # Delete unnecessary emis_dict[keyname]
    
    for keyname in keyname_list:
      del emis_dict[keyname]
    
    #
    #--------
    #
    # sorted_key_list = [] 
    # for key in emis_dict:
    #   sorted_key_list.append(key)
    # #  plot_emis_dict(key)
    # for ikey, key in enumerate(sorted(sorted_key_list)):
    #   print 'check : ',ikey, key
    # raw_input()
    #
    #--------------------------------------------------------------------------------
    #
    # INTERPOLATION : (lon,lat)[yeary] --> (lon,lat)[hourly]
    # ECLIPSE       : Anth VOC and Other (with monthly factors : DO NOT USE THIS. USE v2)
    # HTAP          : AIR and SHIPS
    # Huang         : BC 
    #
    #  Input  unit  : [mol/sec/m2] or [ug/sec/m2] ('BC','OC','PM25')
    #  Output unit  : [mol/hour/km2] or [ug/sec/m2] ('BC','OC','PM25')
    #
    #---------------------------------------------------------------------------------
    if rank == 0:
      print('------------------------------------')
      print('')
      print(datetime.now(),' SECTION : INTERPOLATION TO HOURLY OUTPUT')
      print('INTERPOLATION : (lon,lat)[yeary] --> (lon,lat)[hourly]') 
      print('')
      print('------------------------------------')
    
    keyname_list = []
    for keyname in emis_dict:
      dims = emis_dict[keyname]['voc']['dims']
      units= emis_dict[keyname]['voc']['units'].lower()
      if len(dims) == 2 and keyname not in hourly_keynames and 'GFED' not in keyname:
        print( keyname, emis_dict[keyname]['voc']['units'].lower(),' <---- check')
        keyname_list.append(keyname)
    #raw_input()
    
    for ikeyname, keyname in enumerate(keyname_list):
      if rank == 0:
        print('INTERPOLATION 03 : ', keyname,' (', ikeyname, '/', len(keyname_list),')')
      sec     = keyname.split('-')[0]
      # change sec "soil" to "soi"
      if 'soi' in sec:
        sec   = 'soi'
      dt_temp = start_dt
      while dt_temp <= end_dt:
        year     = dt_temp.year
        month    = dt_temp.month
        day      = dt_temp.day
        hour     = dt_temp.hour
        day_of_week = dt_temp.weekday()
        
        first_day_of_week = monthrange(year,month)[0]
        number_of_days    = monthrange(year,month)[1]
        ### if rank == 0:
        ###   hour2_elements = []
        ###   for hr2 in hour2:
        ###     hour2_elements.append(hr2)
        ###   hour2_elements = np.array(hour2_elements)
        ###   hour2_elements = np.sort(hour2_elements)
        ###   hour2_elements = np.unique(hour2_elements)
        ###   print('hour2 unique elements : ',hour2_elements)
        ###   print('factoring at ',year, month, day, hour, day_of_week, hour2, \
        ###         first_day_of_week, number_of_days)
    
        units    = emis_dict[keyname]['voc']['units'].lower()
        
        emis_data = np.copy(emis_dict[keyname]['voc']['data'])[:,:]
        #debug#emis_data = emis_dict[keyname]['voc']['data'][:,:]
        if 'mol' in units and '/m2' in units and 'sec' in units:
          # unit : mol/m2/sec [HTAP]--> mol/km2/year
          emis_data *= (1.e6*60.0*60.0*24.0*365.0)
        if 'ug' in units and '/m2' in units and 'sec' in units:
          # unit : ug/m2/sec [HTAP]--> ug/m2/year
          emis_data *= (60.0*60.0*24.0*365.0)
        # unit : mol/km2/year --> mol/km2/month
        # unit : ug/m2/year   -->  ug/ m2/month
        emis_data *= monthly_factors[sec][month-1]
        # unit : mol/km2/month --> mol/km2/week
        # unit :  ug/ m2/month -->  ug/ m2/week
        emis_data *= 7.0/number_of_days
        # unit : mol/km2/week --> mol/km2/day
        # unit :  ug/ m2/week -->  ug/ m2/day
        emis_data *= daily_factors[sec][day_of_week]
        # unit : mol/km2/day  --> mol/km2/hour
        # unit :  ug/ m2/day  -->  ug/ m2/hour

        ## hour2 = np.rint(XLON/15.0)+hour
        hour2_temp = hour2+hour
        hour2_temp[hour2_temp >= 48] -= 48
        hour2_temp[hour2_temp >= 24] -= 24
        for ihour in np.arange(24):
          emis_data[hour2_temp == ihour] *= hourly_factors[sec][ihour]
        ## for ind, XLON_temp in np.ndenumerate(XLON):
        ##   if XLON_temp < 360.:
        ##     XLON_temp += 360.
        ##   dhour_loc = int(round(XLON_temp/15.0))
        ##   hour2 = hour+dhour_loc
        ##   while hour2 >= 24:
        ##     hour2 -= 24
        ##   emis_data[ind] *= hourly_factors[sec][hour2]
        # unit (BC,OM,PM25) : ug/m2/hour --> ug/m2/sec
        if 'BC' in keyname or 'OM' in keyname \
           or 'PM25' in keyname or ('OC' in keyname and 'DMS_OC' not in keyname):
          emis_data /= 3600.0
        #
        if 'mol' in units:
          new_units = 'mol/km2/hour'
        elif 'ug' in units:
          new_units = 'ug/m2/sec'
    
        new_keyname = Create_Hourly_Keyname(keyname,dt_temp) 
        hourly_keynames.append(new_keyname)
        Add_EmisData_Dictionary(emis_dict,new_keyname,emis_data,new_units,\
                                XLON, XLAT )
        ## if 'NO_' in new_keyname:
        ##   print new_keyname, '<-- check'
        ##   print np.amax(emis_dict[new_keyname]['voc']['data'])
        dt_temp += timedelta(hours=int(dhour))
      # END of while dt_temp <= end_dt:
    # END of for keyname in keyname_list:
    for keyname in keyname_list:
      del emis_dict[keyname]
    #
    #--------
    #
    ### sorted_key_list = [] 
    ### for key in emis_dict:
    ###   sorted_key_list.append(key)
    ### #  plot_emis_dict(key)
    ### for ikey, key in enumerate(sorted(sorted_key_list)):
    ###   print( 'check 2 : ',ikey, key, emis_dict[key]['voc']['units'])
    # sorted_key_list = [] 
    # for key in emis_dict:
    #   sorted_key_list.append(key)
    # #  plot_emis_dict(key)
    # for ikey, key in enumerate(sorted(sorted_key_list)):
    #   if 'PM25' in key or 'OM' in key or 'BC' in key:
    #     print 'check 9 : ',ikey, key, emis_dict[key]['voc']['units']
    # raw_input()  
    # 
    #-----------------------------------------------------------------------
    #
    #  loop over output variable names 
    #
    #------------------------------------------------------------------------
    if rank == 0:
      print('------------------------------------')
      print('')
      print(datetime.now(),' SECTION : loop over output variable names')
      print('')
      print('------------------------------------')
    
    # for keyname in emis_dict:
    #   print keyname, emis_dict[keyname]['voc']['units']
    #   print np.array(emis_dict[keyname]['voc']['data']).shape
    #
    gas_spec   = [    'CO',  'CH4', 'BC', 'OM', 'OC','SO2','NH3','PM25','NO','NO2']
    #chem_spec  = ['C2H5OH','CH3OH','HC5','ETH','TOL','OL2', 'KET','XYL','HCHO','OLT','ALD','ISO','OLI']
    chem_spec  = ['C2H6','C3H8','C2H2','ALK3','ALK4','ALK5','ETHENE','C3H6','OLE2',\
                  'ARO1','ARO2','ECHO','CCHO','ACET','MEK','TERP','MEOH','PROD2','DMS_OC']
    gfed_spec  = ['ISOPRENE','HCHO','HCOOH','CCO','MGLY']
    spec_list  = gas_spec + chem_spec + gfed_spec

 
    spec_list  = \
                 ['ACET', 'ALK3', 'ALK4', 'ALK5', \
                  'ARO1', 'ARO2', 'BACL', 'C2H2', \
                  'C2H6', 'C3H6', 'C3H8', 'CCHO', \
                  'CH4' , 'CO'  , 'ECJ-EC-BC'  , 'ETHENE', \
                  'HCHO', 'HCOOH', 'HONO', 'MEK', \
                  'MEOH', 'MGLY', 'MVK', 'NH3', 'NO', 'NO2', \
                  #'OLE1', 'OLE2', 'ORG-ORGJ-OM-OC', 'PHEN', \
                  'OLE1', 'OLE2', 'ORGJ-ORG-OM-OC', 'PHEN', \
                  'RCHO', 'SO2', 'TERP', 'CCO_OH', \
                  'ISOPRENE', 'ISOPROD', 'METHACRO', \
                  'NO3J-NO3', 'PM25J-PM25', 'SO4J-SO4', 'DMS_OC']
 

    
    dt_temp = start_dt
    while dt_temp <= end_dt: # Loop over time 
      year  = dt_temp.year
      month = dt_temp.month
      day   = dt_temp.day
      hour  = dt_temp.hour
      format_string = "{:04d}-{:02d}-{:02d}_{:02d}:00:00"
      date_string   = format_string.format(year,month,day,hour)
      date_string_GFED = str(year)+str(month)+str(day)+str(hour)
      for spec in spec_list:      # Loop over ['CO','CH4','BC','OM','SO2','NH3','PM25','NO','NO2','C2H5OH','CH3OH','HC5','ETH','TOL','OL2', 'KET','XYL','HCHO','OLT','ALD','ISO','OLI'
        spec_choices = spec.split('-')
        spec_for_new_keyname = spec_choices[0]
        list_keyname2sum = []
        for keyname in emis_dict:
          if date_string in keyname and '=' not in keyname:
            if 'GFED' not in keyname and 'DMS_OC' not in keyname:
            # keyname : <sec>-<INVENTORY:HTAP,ECLIPSE etc>-<spec>-<YYYY-MM-DD_hh:00:00>
              keyname_spec = keyname.split('-')[2]
              if keyname_spec in spec_choices:
                #if spec == 'ORG-ORGJ-OM-OC':
                if spec == 'ORGJ-ORG-OM-OC':
                  print('check xxx : ',keyname_spec, keyname)
                list_keyname2sum.append(keyname)
            else:
              if 'DMS_OC' in keyname:
                if 'DMS_OC' in spec_choices:
                  list_keyname2sum.append(keyname)

            # keyname : <spec>-GFED-<YYYY-MM-DD_hh:00:00>
              if 'GFED' in keyname:
                #print('GFED is in ',keyname)
                keyname_spec = keyname.split('-GFED-')[0]
                if keyname_spec in spec_choices:
                  ## print(keyname,' is added tp list_keyname2sum')
                  list_keyname2sum.append(keyname)
        # END of for keyname in emis_dict:
        #
        list_keyname2sum = sorted(list_keyname2sum)
        # 
        for key in list_keyname2sum:
           print( 'check 4', dt_temp, spec, key)
        #
        if len(list_keyname2sum) != 0:
          for ikey, key in enumerate(list_keyname2sum):
            ## print('check key : ', ikey, key, emis_dict[key]['voc']['units'].lower())
            if ikey == 0:
              #debug#emis_data = np.array(emis_dict[key]['voc']['data'])
              emis_data = np.copy(emis_dict[key]['voc']['data'])
              units     = emis_dict[key]['voc']['units'].lower()
            else:
              #emis_data2 = np.array(emis_dict[key]['voc']['data'])
              #debug#emis_data += np.array(emis_dict[key]['voc']['data'])
              emis_data += np.copy(emis_dict[key]['voc']['data'])
          # END of for ikey, key in enumerate(list_keyname2sum):
          #
          new_keyname   = spec_for_new_keyname+'='+date_string
          print(new_keyname,' is being created')
          Add_EmisData_Dictionary(emis_dict,new_keyname,emis_data,units,XLON,XLAT)
        #
        for key in list_keyname2sum:
          del emis_dict[key]
        # END of for key in list_keyname2sum:
      # END of for spec in CBMZ_spec:
      dt_temp += timedelta(hours=int(dhour))
    # END of while dt_temp <= end_dt:
   
    #
    #--------
    #
    sorted_key_list = [] 
    for key in emis_dict:
      sorted_key_list.append(key)
    #  plot_emis_dict(key)
    ### if rank == 0:
    ###   for ikey, key in enumerate(sorted(sorted_key_list)):
    ###     print('check 3 : ', key)
    ###     if 'PM25' in key:
    ###       print('check 3 : ',ikey, key, emis_dict[key]['voc']['units'])
    
    #------------------------------------------------------------------------------------------
    #
    #  Construct variables to be stored in wrfchemi files
    #
    #------------------------------------------------------------------------------------------
    if rank == 0:
      print('------------------------------------')
      print('')
      print(datetime.now(),' SECTION : Construct variables to be stored in wrfchemi files')
      print('')
      print('------------------------------------')
    
    wrfinput_filename = wrfdir+'wrfinput_d'+str(dd).zfill(2)
   
    fn_XLON  = './XLON_d'+str(dd).zfill(2)+'.npy'
    fn_XLAT  = './XLAT_d'+str(dd).zfill(2)+'.npy'
    fn_XLONa = './XLONa_d'+str(dd).zfill(2)+'.npy'
    fn_XLATa = './XLATa_d'+str(dd).zfill(2)+'.npy'
    if not(os.path.isfile(fn_XLON ) and \
           os.path.isfile(fn_XLAT ) and \
           os.path.isfile(fn_XLONa) and \
           os.path.isfile(fn_XLATa)):
      XLON, XLAT, XLONa, XLATa = WRF_Grids2(wrfinput_filename)
      np.save(fn_XLON ,XLON )
      np.save(fn_XLAT ,XLAT )
      np.save(fn_XLONa,XLONa)
      np.save(fn_XLATa,XLATa)
    else:
      if rank == 0:
        print(datetime.now(), 'reading XLON, XLAT, XLONa, XLATa from npy files....')

      XLON  = np.load(fn_XLON )
      XLAT  = np.load(fn_XLAT )
      XLONa = np.load(fn_XLONa)
      XLATa = np.load(fn_XLATa)
    
    
    
    nc = Dataset(wrfinput_filename,'r',format='NETCDF4')
    list_glob_att = {}
    for att in nc.ncattrs():
      list_glob_att[att] = nc.getncattr(att)
    
    #
    #   Attribute Name List Kept for output
    #------------------------------------------------------------
    glob_att_name = ['DX','DY','CEN_LAT','CEN_LON','TRUELAT1','TRUELAT2','MOAD_CEN_LAT','STAND_LON',
                     'POLE_LAT','POLE_LON','GMT','JULYR','JULDAY','MAP_PROJ','MMINLU','NUM_LAND_CAT',
                     'ISWATER','ISLAKE','ISICE','ISURBAN','ISOILWATER','WEST-EAST_GRID_DIMENSION',
                     'SOUTH-NORTH_GRID_DIMENSION','BOTTOM-TOP_GRID_DIMENSION']
    
    dim_Time       = 1
    dim_DateStrLen = 19
    dim_west_east  = int(list_glob_att['WEST-EAST_GRID_DIMENSION'])-1
    dim_south_north= int(list_glob_att['SOUTH-NORTH_GRID_DIMENSION'])-1
    dim_bottom_top = int(list_glob_att['BOTTOM-TOP_GRID_DIMENSION'])-1
    dim_emissions_zdim_stag = 10
    
    gas_spec   = [    'CO',  'CH4', 'BC', 'OM', 'OC','SO2','NH3','PM25','NO','NO2']
    #chem_spec  = ['C2H5OH','CH3OH','HC5','ETH','TOL','OL2', 'KET','XYL','HCHO','OLT','ALD','ISO','OLI']
    chem_spec  = ['C2H6','C3H8','C2H2','ALK3','ALK4','ALK5','ETHENE','C3H6','OLE2','ARO1','ARO2','ECHO','CCHO','ACET','MEK','TERP','MEOH','PROD2','DMS_OC']
    gfed_spec  = ['ISOPRENE','HCHO','HCOOH','CCO','MGLY']
    ### chem_spec  = \
    ###            ['acet', 'alk3', 'alk4', 'alk5', \
    ###             'aro1', 'aro2', 'bacl', 'c2h2', \
    ###             'c2h6', 'c3h6', 'c3h8', 'ccho', \
    ###              , 'ethene', \
    ###             'hcho', 'hcooh', 'hono', 'mek', \
    ###             'meoh', 'mvk', \
    ###             'ole1', 'ole2', 'org', 'phen', \
    ###             'rcho', 'so2', 'terp', 'cco_oh', \
    ###             'isoprene', 'isoprod', 'methacro', \
    ###             'no3j', 'pm25j', 'so4j']
    spec_list  = gas_spec + chem_spec + gfed_spec
    
    spec_list  = \
                 ['ACET', 'ALK3', 'ALK4', 'ALK5', \
                  'ARO1', 'ARO2', 'BACL', 'C2H2', \
                  'C2H6', 'C3H6', 'C3H8', 'CCHO', \
                  'CH4' , 'CO'  , 'ECJ'  , 'ETHENE', \
                  'HCHO', 'HCOOH', 'HONO', 'MEK', \
                  'MEOH', 'MGLY', 'MVK', 'NH3', 'NO', 'NO2', \
                  'OLE1', 'OLE2', 'ORGJ', 'PHEN', \
                  'RCHO', 'SO2', 'TERP', 'CCO_OH', \
                  'ISOPRENE', 'ISOPROD', 'METHACRO', \
                  'NO3J', 'PM25J', 'SO4J', 'DMS_OC']

    dt_temp = start_dt
    while dt_temp <= end_dt:
      year  = dt_temp.year
      month = dt_temp.month
      day   = dt_temp.day
      hour  = dt_temp.hour
      format_string = "{:04d}-{:02d}-{:02d}_{:02d}:00:00"
      date_string   = format_string.format(year,month,day,hour)
      ofilename     = out_dir+'./wrfchemi_d'+str(dd).zfill(2)+'_'+date_string 
      if os.path.isfile(ofilename):
        os.remove(ofilename)
      #
      #nc_out = Dataset(ofilename,'w',format='NETCDF3_64BIT')
      nc_out = Dataset(ofilename,'w',format='NETCDF4')
      #
      #-- Copy global attributes from wrfinput file ----
      for att in list_glob_att:
        if att in glob_att_name:
          if isinstance(list_glob_att[att],str):
            nc_out.setncattr(att,list_glob_att[att].encode('ascii'))
          else:
            nc_out.setncattr(att,list_glob_att[att]) 
        ## END OF if att in glob_att_name:
      ## END OF for att in list_glob_att:
      #
      nc_out.setncattr('Data_Description',data_description)
      nc_out.setncattr('TITLE','Created For WRF V3. and V4.')
      #
      #-- 
      nc_out.createDimension('Time',dim_Time)
      nc_out.createDimension('DateStrLen',dim_DateStrLen)
      nc_out.createDimension('west_east',dim_west_east)
      nc_out.createDimension('south_north',dim_south_north)
      nc_out.createDimension('bottom_top',dim_bottom_top)
      nc_out.createDimension('emissions_zdim_stag',dim_emissions_zdim_stag)
     
      Times = nc_out.createVariable('Times','c',('Time','DateStrLen',))
      Times[:] = [date_string]
    
      XLATnc = nc_out.createVariable('XLAT','f4',('south_north','west_east',))
      XLATnc.setncattr('FieldType',104)
      XLATnc.setncattr('MemoryOrder','XY ')
      XLATnc.setncattr('description','Coordinates')
      XLATnc.setncattr('units','degrees')
      XLATnc.setncattr('coordinates','XLONG XLAT')
      XLATnc[:,:] = np.transpose(XLAT[:,:])
    
      XLONGnc = nc_out.createVariable('XLONG','f4',('south_north','west_east',))
      XLONGnc.setncattr('FieldType',104)
      XLONGnc.setncattr('MemoryOrder','XY ')
      XLONGnc.setncattr('description','Coordinates')
      XLONGnc.setncattr('units','degrees')
      XLONGnc.setncattr('coordinates','XLONG XLAT')
      XLONGnc[:,:] = np.transpose(XLON[:,:])
    
      ### OC_OM_count = 0
    
      for spec in spec_list:
        ### if spec == 'CCO':
        ###   wrfchemi_var = 'E_CCO_OH'
        ### elif spec == 'BC':
        ###   wrfchemi_var = 'E_ECJ'
        ### elif spec == 'OC' or spec == 'OM':
        ###   wrfchemi_var = 'E_ORGJ'
        ###   OC_OM_count += 1
        ###   if OC_OM_count == 2:
        ###     continue
        ### else:
        ###  wrfchemi_var = 'E_'+spec
        wrfchemi_var = 'E_'+spec
    
        ## if rank == 0:
        ##   print(datetime.now(), 'spec, wrfchemi_var = ',spec, wrfchemi_var)
        #
        #list_keyname = []
        variable_exist = False
        #
        data2write = np.zeros(shape=(dim_Time,dim_emissions_zdim_stag,dim_south_north,dim_west_east))
        for key in emis_dict:
          print('check 5 key, spec, date_string = ',key,spec,date_string)
          key_split = key.split('=')
          if (spec == key_split[0] and date_string == key_split[1]):
            data2write[0,0,:,:] = np.transpose(emis_dict[key]['voc']['data'][:,:])
            units2write= emis_dict[key]['voc']['units']
            variable_exist = True 
          ## END OF if (spec == key_split[0]):

          ### if ((spec == key_split[0] and spec != 'DMS_OC') or (spec == 'DMS_OC' and spec in key)) \
          ###    and date_string in key:
          ###   data2write = np.zeros(shape=(dim_Time,dim_emissions_zdim_stag,dim_south_north,dim_west_east))
          ###   units2write= emis_dict[key]['voc']['units']
          ###   variable_exist = True
          ###   if spec == 'OC' or spec == 'OM':
          ###     key1 = 'OC_'+date_string
          ###     key2 = 'OM_'+date_string
          ###     if key1 in emis_dict:
          ###       data2write[0,0,:,:] += np.transpose(emis_dict[key1]['voc']['data'][:,:]) 
          ###     if key2 in emis_dict:
          ###       data2write[0,0,:,:] += np.transpose(emis_dict[key2]['voc']['data'][:,:])
          ###   else:
          ###     ## if 'NO' in key:
          ###     ##   print key, np.transpose(emis_dict[key]['voc']['data'][:,:]), '<--- check'
          ###     data2write[0,0,:,:] = np.transpose(emis_dict[key]['voc']['data'][:,:]) 
          ### ## END OF if ((spec == key_split[0] and spec != 'DMS_OC') or (spec == 'DMS_OC' and spec in key)) \
        ## END OF for key in emis_dict:
        #
        if rank == 0:
          print(datetime.now(), 'variable_exist = ',variable_exist)
        if variable_exist:
          Varnc = nc_out.createVariable(wrfchemi_var,'f4',('Time','emissions_zdim_stag','south_north','west_east',))
          Varnc.setncattr('FieldType',104)
          Varnc.setncattr('MemoryOrder','XYZ')
          Varnc.setncattr('description','EMISSIONS')
          Varnc.setncattr('coordinates','XLONG XLAT')
          Varnc.setncattr('units',units2write)
          Varnc[:,:,:,:] = data2write[:,:,:,:]
        ## END OF if variable_exist:
      ## END OF for spec in spec_list:
    
      nc_out.close
      dt_temp += timedelta(hours=int(dhour))
    ## END OF while dt_temp <= end_dt:
    
    del emis_dict  
