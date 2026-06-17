#!/usr/bin/env python
# coding: utf-8

from Functions import *

#if __name__ == "__from_s2m_to_dem_ts_sc__":

if len(sys.argv) != 7:
	print("Nbre d'argument non valide %d : " %len(sys.argv))
	print("Usage: python3 script.py date xmin xmax ymin ymax name_experience")
	sys.exit(1)

from_s2m_to_dem_ts_sc(
	sys.argv[1], #: date datetime.datetime,
	"/home/barroisl/Transect_MC_auto/s2m_simu/s2m_simu.nc", # path to s2m simu
	"/home/barroisl/Transect_MC_auto/s2m_simu/s2m_shapefile.nc", # path to shapefile de la simu
	sys.argv[2], #: xmin int,
	sys.argv[3], #: xmax int,
	sys.argv[4], #: ymin int,
	sys.argv[5], #: ymax int,
	"/home/barroisl/edstar/Simus/models/%s.obj" %sys.argv[6], # path to future .obj
	"/home/barroisl/Transect_MC_auto/s2m_simu/%s.nc" %sys.argv[6], # path to future .nc
	"/home/barroisl/edstar/Simus/materials/%s.mtls" %sys.argv[6], # path to future .mtls
)
