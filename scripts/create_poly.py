#!/usr/bin/env python
# coding: utf-8

from Functions import *

#if __name__ == "__from_s2m_to_dem_ts_sc__":

if len(sys.argv) != 10:
	print("Nbre d'argument non valide %d : " %len(sys.argv))
	print("Usage: python3 script.py     path_to_poly  x1 : int, x2 : int, x3 : int, x4 : int, y1 : int, y2 : int, y3 : int, y4 : int âth_to_save : str")
	sys.exit(1)
	
creating_polygon(
    "/home/barroisl/Transect_MC_auto/polygon/%s.geojson" %sys.argv[9], #path to save .geojson
    sys.argv[1], #x1
    sys.argv[2], #x2
    sys.argv[3], #x3
    sys.argv[4], #x4
    sys.argv[5], #y1
    sys.argv[6], #y2
    sys.argv[7], #y3
    sys.argv[8], #y4
    )
