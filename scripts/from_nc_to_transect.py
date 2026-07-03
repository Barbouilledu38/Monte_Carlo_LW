#!/usr/bin/env python
# coding: utf-8

from Functions import *

if len(sys.argv) != 4:
    print("Usage: python3 from_nc_to_transect.py pts_transect name_experience res")
    sys.exit(1)

from_dem_pts_to_cam_tgt(
	"/home/barroisl/Transect_MC_auto/pts_transect/%s.txt" %sys.argv[1], # path to .txt de transect
	"/home/barroisl/Transect_MC_auto/s2m_simu/%s.nc" %sys.argv[2], # path to previously calculated .nc
	"/home/barroisl/Transect_MC_auto/camera_tgt/%s.txt" %sys.argv[1], # path to res.txt to be calculated
	"/home/barroisl/Transect_MC_auto/topographie/chartreuse_thomas_topo_params.nc", # path to topo_params.netcdf issu de topocalc
	"/home/barroisl/Transect_MC_auto/camera_tgt/topo_transect_%s.txt" %sys.argv[1],  # path to param topo de transect .txt to be calculated
	int(sys.argv[3])//2,
    )
