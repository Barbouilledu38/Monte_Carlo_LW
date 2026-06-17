#!/usr/bin/env python
# coding: utf-8

from Functions import *

if len(sys.argv) != 3:
    print("Usage: python3 from_nc_to_polygon.py polygon ds_s2m")
    sys.exit(1)
    
from_nc_to_polygon_cam_tgt(
	"/home/barroisl/Transect_MC_auto/polygon/%s.geojson" %sys.argv[1], # path to polygon : BaseGeometry #extension fichier ??
	"/home/barroisl/Transect_MC_auto/s2m_simu/%s.nc" %sys.argv[2], # path to previously calculated .nc
	"/home/barroisl/Transect_MC_auto/camera_tgt/polygon_%s_test.txt" %sys.argv[1], # path to res.txt to be calculated 
	"/home/barroisl/Transect_MC_auto/topographie/chartreuse_thomas_topo_params.nc", # path to topo_params.netcdf issu de topocalc
	"/home/barroisl/Transect_MC_auto/camera_tgt/topo_polygon_%s.txt" %sys.argv[1],  # path to param topo de transect .txt to be calculated
    )
