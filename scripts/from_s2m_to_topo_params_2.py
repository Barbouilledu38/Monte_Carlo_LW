#!/usr/bin/env python
# coding: utf-8

from Functions import *

if len(sys.argv) != 2:
    print("Usage: python3 from_nc_to_topo_params.py ds_s2m")
    sys.exit(1)
	
compute_dem_param("/home/barroisl/Transect_MC_auto/s2m_simu/%s.nc" %sys.argv[1],
                  "/home/barroisl/Transect_MC_auto/topographie/%s_topo_params.nc" %sys.argv[1])
