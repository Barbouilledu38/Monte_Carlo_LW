#!/usr/bin/env python
# coding: utf-8


from util import *
from util_htrdr import *

import numpy as np
import rasterio
import netCDF4 as nc
import os
import matplotlib.pyplot as plt
import numba as nb
import tqdm
import random
from pyproj import Transformer
import rioxarray as riox
import xarray as xr
import datetime
import sys

from topocalc import gradient
from topocalc import viewf
from topocalc import horizon

#################### Interpolation sur une grille 2D ###################################

@nb.njit(cache=True)
def _searchsorted(arr: np.ndarray, val: float) -> int:
    """Recherche binaire : retourne l'index i tel que arr[i] <= val < arr[i+1]."""
    lo, hi = 0, arr.shape[0] - 2          # on veut l'index gauche
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if arr[mid] <= val:
            lo = mid
        else:
            hi = mid - 1
    return lo

@nb.njit(cache=True, parallel=False)
def bilinear_interpolation(
    xs: np.ndarray,   # coordonnées x de la grille  (Nx,)  triées croissantes
    ys: np.ndarray,   # coordonnées y de la grille  (Ny,)  triées croissantes
    svf: np.ndarray,  # valeurs SVF                 (Ny, Nx)
    points: np.ndarray,  # points cibles             (N, 3)  colonnes x, y, z
) -> np.ndarray:
    """
    Interpolation bilinéaire de `svf` aux positions (x, y) de chaque point.
 
    Parameters
    ----------
    xs     : 1-D array de longueur Nx — axe x de la grille (croissant)
    ys     : 1-D array de longueur Ny — axe y de la grille (croissant)
    svf    : 2-D array (Ny, Nx)       — valeurs à interpoler
    points : 2-D array (N, 3)         — col 0 = x, col 1 = y, col 2 = z (ignoré)
 
    Returns
    -------
    result : 1-D array (N,) de valeurs SVF interpolées (NaN si hors grille)
    """
    N = points.shape[0]
    result = np.empty(N, dtype=np.float64)
 
    x_min, x_max = xs[0], xs[-1]
    y_min, y_max = ys[0], ys[-1]
 
    for k in range(N):
        px = points[k, 0]
        py = points[k, 1]
 
        # Hors domaine → NaN
        if px < x_min or px > x_max or py < y_min or py > y_max:
            result[k] = np.nan
            continue
 
        ix = _searchsorted(xs, px)
        iy = _searchsorted(ys, py)
        
        while (np.isnan(svf[iy,ix]) == True or svf[iy+1,ix+1] == True or
            svf[iy+1,ix] == True or svf[iy,ix+1] == True):
            
            ix = ix+1
            iy = iy+1
            
        x0, x1 = xs[ix], xs[ix + 1]
        y0, y1 = ys[iy], ys[iy + 1]
 
        # Poids de pondération
        tx = (px - x0) / (x1 - x0)
        ty = (py - y0) / (y1 - y0)
 
        # Coins de la cellule  svf[iy, ix] → ligne = y, colonne = x
        f00 = svf[iy,     ix    ]
        f10 = svf[iy,     ix + 1]
        f01 = svf[iy + 1, ix    ]
        f11 = svf[iy + 1, ix + 1]
 
        result[k] = (
            f00 * (1 - tx) * (1 - ty)
            + f10 *      tx  * (1 - ty)
            + f01 * (1 - tx) *      ty
            + f11 *      tx  *      ty
        )
 
    return result

def interpolate(
    ds: xr.Dataset,
    points: np.ndarray,
    var_name: str, #= "SVF",
    x_dim: str = "x",
    y_dim: str = "y",
) -> np.ndarray:
    """
    Interpole la variable `var_name` du dataset xarray aux positions (x, y)
    définies par un tableau de points 3D.
 
    Parameters
    ----------
    ds       : xr.Dataset contenant la variable et des coordonnées x/y régulières
    points   : np.ndarray de forme (N, 2) — colonnes [x, y]
    x_dim    : nom de la dimension x dans le dataset  (défaut : 'x')
    y_dim    : nom de la dimension y dans le dataset  (défaut : 'y')
    var_name : nom de la variable à interpoler        (défaut : 'SVF')
 
    Returns
    -------
    np.ndarray (N,) de valeurs interpolées (float64, NaN hors domaine)
    """
    # Extraction des tableaux NumPy (opération Python — hors numba)
    
    xs  = ds[x_dim].values.astype(np.float64)
    ys  = ds[y_dim].values.astype(np.float64)
    svf = ds[var_name].values.astype(np.float64)
 
    # Garantit que les axes sont croissants (xarray peut les stocker décroissants)
    if xs[-1] < xs[0]:
        xs  = xs[::-1].copy()
        svf = svf[:, ::-1].copy()
    if ys[-1] < ys[0]:
        ys  = ys[::-1].copy()
        svf = svf[::-1, :].copy()
 
    pts = np.ascontiguousarray(points, dtype=np.float64)
 
    return bilinear_interpolation(xs, ys, svf, pts)
    
@nb.njit()
def from_pts_to_2D(pts : np.ndarray,
                   nx : int,
                   ny : int
                  )->np.ndarray:
    zs = np.zeros((nx,ny))
    for j in range(ny):
        zs[:,j] = pts[j*nx:(j+1)*nx,]
        
    return zs.T
    
@nb.njit()   
def remove_nan_from_2D(D2_arr : np.ndarray
    )-> np.ndarray:
    res = D2_arr.copy()
    mask = np.isnan(D2_arr)
    a,b = mask.shape
    for i in range(a):
        for j in range(b):
            if mask[i,j] == True :
                k = 0
                while np.isnan(mask[i+k,j]) == True :
                    k += 1
                res[i,j] = res[i+k,j]
    return res

################################ Extraction de paramètres topo le long d'un transect ###################

from pyproj import Transformer

def convert_epsg_pts(xs,ys, epsg_src=2154, epsg_tgt=32632):
    """
    Simple function to convert a list fo poitn from one projection to another oen using PyProj

    Args:
        xs (array): 1D array with X-coordinate expressed in the source EPSG
        ys (array): 1D array with Y-coordinate expressed in the source EPSG
        epsg_src (int): source projection EPSG code
        epsg_tgt (int): target projection EPSG code

    Returns: 
        array: Xs 1D arrays of the point coordinates expressed in the target projection
        array: Ys 1D arrays of the point coordinates expressed in the target projection
    """
    #print('Convert coordinates from EPSG:{} to EPSG:{}'.format(epsg_src, epsg_tgt))
    trans = Transformer.from_crs("epsg:{}".format(epsg_src), "epsg:{}".format(epsg_tgt), always_xy=True)
    Xs, Ys = trans.transform(xs, ys)
    return Xs, Ys

def topo_params(ds: xr.Dataset,
    pts: np.ndarray,
) -> np.ndarray:
    
    svf_transect = interpolate(ds,pts[:,:2],var_name = 'svf',x_dim = 'x',y_dim = 'y')
    svf_LSLOPE_transect = interpolate(ds,pts[:,:2],var_name = 'svf_LSLOPE',x_dim = 'x',y_dim = 'y')
    aspect_transect = interpolate(ds,pts[:,:2],var_name = 'aspect',x_dim = 'x',y_dim = 'y')
    slope_transect = interpolate(ds,pts[:,:2],var_name = 'slope',x_dim = 'x',y_dim = 'y')
    elevation_transect = interpolate(ds,pts[:,:2],var_name = 'zs',x_dim = 'x',y_dim = 'y')

    return np.column_stack([svf_transect,svf_LSLOPE_transect,aspect_transect,slope_transect,elevation_transect])
    
############################## From s2m to obj ###################################################

def from_dem_ts_sc_to_obj(
    x: np.ndarray,   # coordonnées x de la grille  (Nx,)  triées croissantes
    y: np.ndarray,   # coordonnées y de la grille  (Ny,)  triées croissantes
    zs: np.ndarray,  # valeurs z                 (Ny, Nx)
    sc: np.ndarray,  # valeurs z                 (Ny, Nx)
    ts: np.ndarray,  # valeurs z                 (Ny, Nx)
    fic_obj : str,
    fic_mtls : str,
):
    """
    A partir d'un ndarray de topo, température de surface et couverture de neige sort un .obj et .mtls
    dans le répertoire spécifié
    
    [Input]
    - x
    - y
    - zs ndarray d'elevation
    - sc ndarray de couverture neigeuse
    - ts ndarray de temperature de surface
    
    [Output]
    - Aucune, écriture des fichier .obj et .mtls pour htrdr
    """
        
    #x,y,lons,lats = xy_bord_dem(zs, src, transform)
    #mask = np.isnan(zs)
    mask = np.full( (len(y),len(x)), False)
    
    print("x,y,zs OK")
    
    lignes_obj = []
    
    # numérotation des sommets
    num = 0 # numéro du sommet dans le .obj
    d_num = {} # { (indice ific dans l_fic, j, i de ific) : n° de sommet}
    nli,ncol = zs.shape # nb de lignes (en j), de colonnes (i)
    for j in range(nli):
        for i in range(ncol):
                # la case i,j utilisera aussi les sommets
                # (en x,y) i+1,j i,j+1 i+1,j+1: sommet i,j utile
                # si case i,j i-1,j i,j-1 ou i-1,j-1 utilisée
            if (not mask[j,i] or not mask[j,i-1]
                or not mask[j-1,i]
                or not mask[j-1,i-1]):
                    # x,y,z arrondi à 3 décimales
                    # 2D pour x[-1],y[-1]
                    
                x1 = round(x[i] if x.ndim
                    == 1 else x[j,i], 3)
                y1 = round(y[j] if y.ndim
                    == 1 else y[j,i], 3)
                zs1 = round(zs[j,i],3)
                
                ligne = f'v {x1} {y1} {zs1}\n'
                lignes_obj.append(ligne)
                num += 1 # numéro commençant à 1
                d_num[(j,i)] = num
    lignes_obj.append('\n') # ligne vide
    
    print("lignes_obj OK : Nbre %d ; " %len(lignes_obj), lignes_obj[0])
    
    # Ts sur chaque milieu de case
    d_ts = {} # {Ts arrondi (= 1 matériau): liste des (indice ific, j,i)} + pentes venant d'un array en input
    nli,ncol = zs.shape # nb de lignes (en j), de colonnes (i)
    # -1: autant de y utiles que de coord. j de Ts (50m: 401->400)
    for j in range(nli - 1):
        # -1: autant de x utiles que de coord i de Ts (501->500)
        for i in range(ncol - 1):
            if not mask[j,i] :
                # Ts en K, arrondi à 0.1
                ts_arr = round(ts[j,i],1)
                sc_arr = round(sc[j,i],2)
                zs_arr = round(zs[j,i],0)
                if ts_arr not in d_ts:
                    d_ts[ts_arr] = []
                
                d_ts[ts_arr].append( (j,i,sc_arr,zs_arr) )
                
    print("d_ts features OK : ", len(d_ts),min(list(d_ts)),max(list(d_ts)))

    lignes_mtls =  []
    
    #cwd = os.getcwd() # répertoire actuel
    cwd='${HTRDR_ATMOSPHERE_SPK}'
    for ts_arr in sorted(d_ts): # liste triée des Ts arrondies   
                    
        indices_snow = [i for i, v in enumerate(d_ts[ts_arr]) if v[2] > 0.01 and np.isnan(v[2]) == False]
        #indices_no_snow = [i for i, v in enumerate(d_ts[ts_arr]) if v[2] < 0.01]
        indice_forest = [i for i, v in enumerate(d_ts[ts_arr]) if v[2] < 0.01 and v[3] < 1500 and np.isnan(v[2]) == False]
        indice_grass = [i for i, v in enumerate(d_ts[ts_arr]) if v[2] < 0.01 and v[3] > 1500 and v[3] < 2000 and np.isnan(v[2]) == False]
        indice_rock = [i for i, v in enumerate(d_ts[ts_arr]) if v[2] < 0.01 and v[3] > 2000 and np.isnan(v[2]) == False]
        indice_nan = [i for i, v in enumerate(d_ts[ts_arr]) if np.isnan(v[2]) == True]
        
        ts_arr_snow = [d_ts[ts_arr][k] for k in indices_snow]
        ts_arr_forest = [d_ts[ts_arr][k] for k in indice_forest]
        ts_arr_grass = [d_ts[ts_arr][k] for k in indice_grass]
        ts_arr_rock = [d_ts[ts_arr][k] for k in indice_rock]
        ts_arr_nan = [d_ts[ts_arr][k] for k in indice_nan]
        #ts_arr_no_snow = [d_ts[ts_arr][k] for k in indices_no_snow]
        
        if len(ts_arr_snow) > 0 :
        
            mrumtl = 'snow'
            nom_mat = mrumtl + str(ts_arr).replace('.','_') # ex.'snow238_1'
            lignes_obj.append(f'usemtl air:{nom_mat}\n')

	    # Boucles sur les facettes de température ts_arr d'indice inférieur à frac_neige*len(d_ts)
	    # dans d_ts.sorted() --> Toutes les facettes recouvertes de neige
            for j,i,cs,zs in ts_arr_snow :
	        # 2 triangles en ht à dr de (j,i), sens aig. montre (cf.
	        #  htrdr-Atmosphere-Starter-Pack-0.8.0/models/plane.obj)
	        
                lignes_obj.append( # a en haut à droite du point j,i
	            f'f {d_num[(j,i)]} ' # j,i
	            f'{d_num[(j+1,i)]} ' # + haut que j,i
	            f'{d_num[(j,i+1)]}\n') # + à droite que j,i
                lignes_obj.append( # b même case, plus en haut à droite
	            f'f {d_num[(j,i+1)]} ' # + à droite que j,i
	            f'{d_num[(j+1,i)]} ' # + haut que j,i
	            f'{d_num[(j+1,i+1)]}\n')# + haut + à droite
	            
            lignes_obj.append('\n') # ligne vide
	    
            lignes_mtls.append(f'{nom_mat} '
	            f'"{cwd}/materials/legacy/{mrumtl}.mrumtl" {ts_arr}\n')
           
	    
        if len(ts_arr_forest) > 0 : 
        
            mrumtl = 'forest_green'
            nom_mat = mrumtl + str(ts_arr).replace('.','_') # ex.'snow238_1'
            lignes_obj.append(f'usemtl air:{nom_mat}\n')

	    # Boucles sur les facettes de température ts_arr d'indice inférieur à frac_neige*len(d_ts)
	    # dans d_ts.sorted() --> Toutes les facettes recouvertes de neige
            for j,i,cs,zs in ts_arr_forest :
	        # 2 triangles en ht à dr de (j,i), sens aig. montre (cf.
	        #  htrdr-Atmosphere-Starter-Pack-0.8.0/models/plane.obj)
	        
                lignes_obj.append( # a en haut à droite du point j,i
	            f'f {d_num[(j,i)]} ' # j,i
	            f'{d_num[(j+1,i)]} ' # + haut que j,i
	            f'{d_num[(j,i+1)]}\n') # + à droite que j,i
                lignes_obj.append( # b même case, plus en haut à droite
	            f'f {d_num[(j,i+1)]} ' # + à droite que j,i
	            f'{d_num[(j+1,i)]} ' # + haut que j,i
	            f'{d_num[(j+1,i+1)]}\n')# + haut + à droite
	            
            lignes_obj.append('\n') # ligne vide
	    
            lignes_mtls.append(f'{nom_mat} '
	            f'"{cwd}/materials/legacy/{mrumtl}.mrumtl" {ts_arr}\n')
            
        if len(ts_arr_grass) > 0 : 
        
            mrumtl = 'grass'
            nom_mat = mrumtl + str(ts_arr).replace('.','_') # ex.'snow238_1'
            lignes_obj.append(f'usemtl air:{nom_mat}\n')

	    # Boucles sur les facettes de température ts_arr d'indice inférieur à frac_neige*len(d_ts)
	    # dans d_ts.sorted() --> Toutes les facettes recouvertes de neige
            for j,i,cs,zs in ts_arr_grass :
	        # 2 triangles en ht à dr de (j,i), sens aig. montre (cf.
	        #  htrdr-Atmosphere-Starter-Pack-0.8.0/models/plane.obj)
	        
                lignes_obj.append( # a en haut à droite du point j,i
	            f'f {d_num[(j,i)]} ' # j,i
	            f'{d_num[(j+1,i)]} ' # + haut que j,i
	            f'{d_num[(j,i+1)]}\n') # + à droite que j,i
                lignes_obj.append( # b même case, plus en haut à droite
	            f'f {d_num[(j,i+1)]} ' # + à droite que j,i
	            f'{d_num[(j+1,i)]} ' # + haut que j,i
	            f'{d_num[(j+1,i+1)]}\n')# + haut + à droite
	            
            lignes_obj.append('\n') # ligne vide
	    
            lignes_mtls.append(f'{nom_mat} '
	            f'"{cwd}/materials/legacy/{mrumtl}.mrumtl" {ts_arr}\n')
                
        if len(ts_arr_rock) > 0 :
        
            mrumtl = 'limestone'
            nom_mat = mrumtl + str(ts_arr).replace('.','_') # ex.'snow238_1'
            lignes_obj.append(f'usemtl air:{nom_mat}\n')

	    # Boucles sur les facettes de température ts_arr d'indice inférieur à frac_neige*len(d_ts)
	    # dans d_ts.sorted() --> Toutes les facettes recouvertes de neige
            for j,i,cs,zs in ts_arr_rock :
	        # 2 triangles en ht à dr de (j,i), sens aig. montre (cf.
	        #  htrdr-Atmosphere-Starter-Pack-0.8.0/models/plane.obj)
	        
                lignes_obj.append( # a en haut à droite du point j,i
	            f'f {d_num[(j,i)]} ' # j,i
	            f'{d_num[(j+1,i)]} ' # + haut que j,i
	            f'{d_num[(j,i+1)]}\n') # + à droite que j,i
                lignes_obj.append( # b même case, plus en haut à droite
	            f'f {d_num[(j,i+1)]} ' # + à droite que j,i
	            f'{d_num[(j+1,i)]} ' # + haut que j,i
	            f'{d_num[(j+1,i+1)]}\n')# + haut + à droite
	            
            lignes_obj.append('\n') # ligne vide
	    
            lignes_mtls.append(f'{nom_mat} '
	            f'"{cwd}/materials/legacy/{mrumtl}.mrumtl" {ts_arr}\n')
                    
        lignes_obj.append('\n') # ligne vide
        
    lignes_mtls.append('air none\n')
    
    print("lignes_mtls OK : Nbre %d ; " %len(lignes_mtls), lignes_mtls[0])

    #On change les lignes de l\'.obj pour faire apparaître une fraction neigeuse
    
    with open(fic_obj,'w') as fid: # w: créer, défaut: encoding='UTF-8'
        fid.writelines(lignes_obj)
    with open(fic_mtls,'w') as fid:
        fid.writelines(lignes_mtls)

def select_spatial_extent(
    ds: xr.Dataset,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_dim: str = "xx",
    y_dim: str = "yy",
) -> xr.Dataset:
    """
    Select a dataset within [x_min, x_max] × [y_min, y_max].
    
    x_min = int(x_min)
    x_max = int(x_max)
    y_min = int(y_min)
    y_max = int(y_max)
    """
    return ds.sel(
        {
            x_dim: ds[x_dim][(ds[x_dim] >= x_min) & (ds[x_dim] <= x_max)],
            y_dim: ds[y_dim][(ds[y_dim] >= y_min) & (ds[y_dim] <= y_max)],
        }
    )

def from_s2m_to_dem_ts_sc(
    date : datetime.datetime,
    ds_s2m : str,
    ds_shapefile : str,
    xmin : int,
    xmax : int,
    ymin : int,
    ymax : int,
    fic_obj : str,
    fic_ds : str,
    fic_mtls : str,
):
    ds = xr.open_dataset(ds_s2m)
    ds_sel = ds.sel(time=date)
    ds_sel_cropted = select_spatial_extent(ds_sel,xmin,xmax,ymin,ymax)
    
    xs = ds_sel_cropted['xx'].values.astype(np.float64) # Penser à changer xx pour x si jamais, pareil pour yy et y
    ys = ds_sel_cropted['yy'].values.astype(np.float64)
    
    #zs = ds_sel_cropted[''].values.astype(np.float64) # Penser à potentiellement changer le nom de cette variable pour quelque chose comme zs dans la simu de la chartreuse
    cs = ds_sel_cropted['DSN_T_ISBA'].values.astype(np.float64)
    ts = ds_sel_cropted['TG1_ISBA'].values.astype(np.float64)
    
    xx,yy = np.meshgrid(xs,ys)
    ny,nx = xx.shape
    xx = xx.ravel()
    yy = yy.ravel()
    pts_z_interp = np.column_stack((xx,yy))
    
    ds_topo = xr.open_dataset(ds_shapefile)
    zs_interp = interpolate(ds_topo,pts_z_interp,var_name = 'ZS', x_dim = 'x', y_dim = 'y')
    zs_2D = from_pts_to_2D(zs_interp,nx,ny)
    
    x_32632,y_32632 = convert_epsg_pts(xx,yy, epsg_src=2154, epsg_tgt=32632)
    
    x_2D = from_pts_to_2D(x_32632,nx,ny)
    y_2D = from_pts_to_2D(y_32632,nx,ny)
    
    xs = x_2D[0]
    ys = y_2D[:,0]
    
    zs = remove_nan_from_2D(zs_2D)
    cs = remove_nan_from_2D(cs)
    ts = remove_nan_from_2D(ts)
    
    print(len(xs))
    if xs[-1] < xs[0]:
        xs  = xs[::-1].copy()
        zs = zs[:, ::-1].copy()
        cs = cs[:, ::-1].copy()
        ts = ts[:, ::-1].copy()
    print(len(ys))
    if ys[-1] < ys[0]:
        ys  = ys[::-1].copy()
        zs = zs[::-1, :].copy()
        cs = cs[::-1, :].copy()
        ts = ts[::-1, :].copy()
  
    ds_fin = xr.Dataset(
        {'zs': (["y", "x"], zs),
         'cs': (["y", "x"], cs),
         'ts': (["y", "x"], ts),
         
        },
        coords={"x": xs, "y": ys},
    )
    
    ds_fin.to_netcdf(fic_ds)
    
    print("cs, ts, zs feild OK")
    
    from_dem_ts_sc_to_obj(xs,ys,zs,cs,ts,fic_obj,fic_mtls)
    
    print(".obj, .mtls OK")
    
############################################ Creating ds_topo #############################################

def compute_dem_param(path_to_ds_s2m : str,
                      path_to_topo_params : str):
    """
    Function to compute and derive DEM parameters: slope, aspect, sky view factor
    """
    
    ds = xr.open_dataset(path_to_ds_s2m)
        
    # Excract horizontal deplacement
    dx = ds.x.diff('x').median().values
    dy = ds.y.diff('y').median().values
   
    # Elevation data
    dem_arr = ds.zs.values
    
    # Computing slope, aspect et ajout dans le dataframe
    slope, aspect = gradient.gradient_d8(dem_arr, dx, dy)
    ds['slope'] = (["y", "x"], slope)
    ds['aspect'] = (["y", "x"], np.deg2rad(aspect))
    ds['aspect_cos'] = (["y", "x"], np.cos(np.deg2rad(aspect)))
    ds['aspect_sin'] = (["y", "x"], np.sin(np.deg2rad(aspect)))
    
    print("slope, aspect OK")
    
    # SVF
    svf = viewf.viewf(np.double(dem_arr), dx)[0]
    ds['svf'] = (["y", "x"], svf)
    
    svf_LSLOPE = (np.pi-slope)/np.pi
    ds['svf_LSLOPE'] = (["y", "x"], svf_LSLOPE)
    
    print("SVFs OK")
    
    # Attribues dataframe
    ds.x.attrs = {'units': 'm'}
    ds.y.attrs = {'units': 'm'}
    ds.slope.attrs = {'units': 'rad'}
    ds.aspect.attrs = {'units': 'rad'}
    ds.aspect_cos.attrs = {'units': 'cosinus'}
    ds.aspect_sin.attrs = {'units': 'sinus'}
    ds.svf.attrs = {'units': 'ratio', 'standard_name': 'svf', 'long_name': 'Sky view factor'}
    ds.svf_LSLOPE.attrs = {'units': 'ratio', 'standard_name': 'svf_LSLOPE', 'long_name': 'Sky view factor SURFEX'}
    
    ds.to_netcdf(path_to_topo_params)

############################################ From position to luminence (surface) #########################

H = 6.62607015e-34   # Planck constant      [J·s]
C = 2.99792458e8     # Speed of light        [m/s]
KB = 1.380649e-23    # Boltzmann constant    [J/K]
sigma = 5.67e-8

@nb.njit(cache = True)
def Planck_law(
    wavelength: float | np.ndarray, 
    temperature: float) -> float | np.ndarray:
              
    return (2 * H * C**2 / wavelength**5) / (np.exp(H * C / (wavelength * KB * temperature)) - 1)  

def from_position_to_luminence(abs_nd : np.ndarray, 
                               ds : xr.Dataset,
                               Tref : float,
                              )-> np.ndarray :
    """
    [Input]
    - abs_nd : ndarray (Nbre_chemin, 4) x,y,z absorption & lambda
    - ds de température de surface de l'.obj
    - Tref : température de référence pour la lancer de rayons
    """
    a,b = positions_abs.shape
    ts_interp = interpolate(ds,abs_nd[:,:2],var_name = 'ts')
    
    return Planck_law(abs_nd[:,7],ts_interp)/(Planck_law(abs_nd[:,7],Tref)*np.pi/(sigma*Tref**4))
    
    
############################ From DEM and points to camera_tgt htrdr ######################################

@nb.njit(cache = True)
def pdt_scallar_test_dem(facet_arr_seg,vect_seg,length,eps):
    cos_theta = np.sum(vect_seg*facet_arr_seg)
    if np.sqrt(1-cos_theta**2)*length>eps :
        return False
    else :
        return True
    
@nb.njit(cache = True) 
def normal_to_sommet(v1,v2):
     
    #normal = np.cross(v1,v2)   
    normal = np.cross(v2, v1)
    if normal[2] < 0 :
    	normal = np.cross(v1, v2)
    #print(np.dot(v1,normal),np.dot(v2,normal))
    norm = np.linalg.norm(normal)
    
    return normal / norm if norm != 0 else normal

@nb.njit(cache = True)
def vect_segment_DEM(v1,v2):
    vect = v1-v2
    norm = np.linalg.norm(vect)
    return vect/norm
    
@nb.njit(cache = True)
def adjacent_slope_normal(pts,i,j):
    
    vect_xslope = pts[i+1,j,:] - pts[i-1,j,:]
    vect_yslope = pts[i,j+1,:] - pts[i,j-1,:]
    
    normal = normal_to_sommet(vect_xslope,vect_yslope)
    
    return normal
    
@nb.njit() 
def normal_to_facet_tgt(center, normal):
    l = 7000
    return center + normal*l
    
@nb.njit()
def chose(points_arr : np.ndarray,
          pts : np.ndarray,
          eps = 15,
         )-> np.ndarray:
    """
    Cherche les points du DEM qui serviront de support de caméra 
    en mesurant l'écart orthogonal au transect renseigné
    [INPUT]
    - points_arr : le transect
    - pts : ndarray (Nbre_points, 3) x,y,z de chaque sommets du DEM
    
    [OUTPUT]
    - ndarray (Nbre_camera,6) des position et tgt par caméra
    """
    
    max_x = np.max(points_arr[:,0])
    min_x = np.min(points_arr[:,0])
    max_y = np.max(points_arr[:,1])
    min_y = np.min(points_arr[:,1])
        
    a,b,c = np.shape(pts)
    Nbre_pts = a*b
    res = np.zeros((Nbre_pts,6), dtype = np.float32)
    
    vect_seg = vect_segment_DEM(points_arr[1],points_arr[0])
    
    for i in range(a):
        for j in range(b):
        
            if pts[i,j,0] > min_x and pts[i,j,0] < max_x and pts[i,j,1] > min_y and pts[i,j,1] < max_y :

                length_h = np.sqrt((pts[i,j,0] - points_arr[0,0])**2 + (pts[i,j,1] - points_arr[0,1])**2)
                
                pts_seg = vect_segment_DEM(pts[i,j,:2],points_arr[0,:2])
                if pdt_scallar_test_dem(pts_seg,vect_seg,length_h,eps) == True :
                    res[i,:3] = pts[i,j,:]
                    normal = adjacent_slope_normal(pts,i,j)
                    res[i,3:6] = normal_to_facet_tgt(pts[i,j,:], normal)

    return res[res[:,0] != 0]

def from_dem_pts_to_cam_tgt(
        fic_pts_arr : str,
        fic_ds : str,
        fic_res : str,
        fic_ds_topo : str,
        fic_res_topo : str,
        eps : int,
        ):
    
    ds = xr.open_dataset(fic_ds)
    a,b = ds.zs.shape

    xx,yy = np.meshgrid(ds.x.values,ds.y.values)
    pts = np.zeros((a,b,3))
    pts[:,:,0] = xx
    pts[:,:,1] = yy
    pts[:,:,2] = ds.zs.values+5
    
    points_arr = np.loadtxt(fic_pts_arr)
    
    res = chose(points_arr,pts,eps)
    
    np.savetxt(fic_res,res,delimiter = '\t',newline='\n')
    
    print("Camera & target OK")
    
    ds_topo = xr.open_dataset(fic_ds_topo)
    
    topo_transect = topo_params(ds_topo,res[:,:3])
    
    np.savetxt(fic_res_topo,topo_transect,delimiter = '\t',newline='\n')
    
    print("transect param topo OK")
    
    
############ Creer les caméras internes à un polygone ##########################"

from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry

def filter_dataset_by_polygon(
    ds: xr.Dataset,
    polygon: BaseGeometry,
    x_coord: str = "x",
    y_coord: str = "y",
) -> xr.Dataset:
    """
    Retourne les points d'un xarray.Dataset situés à l'intérieur d'un polygone.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset dont les coordonnées ``x_coord`` et ``y_coord`` définissent
        la position spatiale de chaque point.
    polygon : shapely.geometry.BaseGeometry
        Polygone (ou MultiPolygone) de référence.
    x_coord : str, optional
        Nom de la coordonnée X dans le dataset (défaut : ``"x"``).
    y_coord : str, optional
        Nom de la coordonnée Y dans le dataset (défaut : ``"y"``).

    Returns
    -------
    xr.Dataset
        Sous-ensemble du dataset ne contenant que les points à l'intérieur
        du polygone. Le masque booléen utilisé est conservé en tant que
        variable ``"in_polygon"``.

    Raises
    ------
    ValueError
        Si ``x_coord`` ou ``y_coord`` est absent du dataset.
    TypeError
        Si ``polygon`` n'est pas une géométrie Shapely valide.
    """
    # --- Vérifications ---
    if x_coord not in ds.coords and x_coord not in ds:
        raise ValueError(f"Coordonnée '{x_coord}' introuvable dans le dataset.")
    if y_coord not in ds.coords and y_coord not in ds:
        raise ValueError(f"Coordonnée '{y_coord}' introuvable dans le dataset.")
    if not isinstance(polygon, BaseGeometry):
        raise TypeError(
            f"'polygon' doit être une géométrie Shapely, reçu : {type(polygon)}"
        )

    x_vals = ds[x_coord].values
    y_vals = ds[y_coord].values

    """
    # --- Cas 1 : coordonnées 1-D (points individuels, ex. dim "points") ---
    if x_vals.ndim == 1 and y_vals.ndim == 1 and x_vals.shape == y_vals.shape:
        mask = _build_mask_1d(x_vals, y_vals, polygon)
        dim = ds[x_coord].dims[0]
        ds_filtered = ds.isel({dim: mask})

    # --- Cas 2 : grille 2-D (x et y sont des tableaux 2-D) ---
    elif x_vals.ndim == 2 and y_vals.ndim == 2:
        mask = _build_mask_2d(x_vals, y_vals, polygon)
        ds_filtered = _apply_2d_mask(ds, mask, x_coord, y_coord)
    """

    # --- Cas 3 : coordonnées séparées 1-D → grille (meshgrid implicite) ---
    if x_vals.ndim == 1 and y_vals.ndim == 1:
        xx, yy = np.meshgrid(x_vals, y_vals)
        mask = _build_mask_2d(xx, yy, polygon)
        #print(mask.shape)
        #ds_filtered = _apply_2d_mask(ds, mask, x_coord, y_coord)
        #print('ho')

    else:
        raise ValueError(
            "Forme inattendue pour les coordonnées x/y. "
            f"x.shape={x_vals.shape}, y.shape={y_vals.shape}"
        )

    # Ajout du masque booléen comme variable informationnelle
    #ds_filtered["in_polygon"] = xr.full_like(
    #   da_filtered, fill_value=True, dtype=bool)
   
    #print(ds_filtered["in_polygon"].values.shape)
	
    #return ds_filtered
    return mask

def _build_mask_1d(
    x: np.ndarray, y: np.ndarray, polygon: BaseGeometry
) -> np.ndarray:
    """Masque booléen pour un tableau 1-D de points (x[i], y[i])."""
    return np.array(
        [polygon.contains(Point(xi, yi)) for xi, yi in zip(x, y)],
        dtype=bool,
    )


def _build_mask_2d(
    xx: np.ndarray, yy: np.ndarray, polygon: BaseGeometry
) -> np.ndarray:
    """
    Masque booléen pour une grille 2-D.

    Utilise ``polygon.contains`` via vectorisation NumPy pour de meilleures
    performances par rapport à une double boucle Python.
    """
    flat_x = xx.ravel()
    flat_y = yy.ravel()
    mask_flat = np.array(
        [polygon.contains(Point(xi, yi)) for xi, yi in zip(flat_x, flat_y)],
        dtype=bool,
    )
    return mask_flat.reshape(xx.shape)


def _apply_2d_mask(
    ds: xr.Dataset,
    mask: np.ndarray,
    x_coord: str,
    y_coord: str,
) -> xr.Dataset:
    """
    Convertit un masque 2-D en sélection de points et retourne un Dataset 1-D.

    Les points sélectionnés sont aplatis en une nouvelle dimension ``"points"``.
    """
    idx_y, idx_x = np.where(mask)
    x_dim = ds[x_coord].dims[-1]   # dernière dim = colonnes
    y_dim = ds[y_coord].dims[0]    # première dim = lignes
    
    mask_da = xr.DataArray(
        mask, dims=[y_dim, x_dim]
    )

    return ds.isel(points=mask_da.values)

    
@nb.njit()
def chose_polygone(
         pts : np.ndarray
         )-> np.ndarray:
    """
    A partir d'une données d'altitude (x,y) donne les centres et les tgt de caméra htrdr.
    A associé à la recherche de caméras dans un polygone.
    [INPUT]
    - pts : ndarray 3D (nx,ny,4) pour un tableau x,y,z,bool
    
    [OUTPUT]
    - ndarray (nx,ny,6) des position et tgt par caméra
    """
    

    nx,ny,c = pts.shape
    
    res = np.zeros((nx,ny,6), dtype = np.float32)
    
    for i in range(nx):
        for j in range(ny):
            if pts[i,j,3] == True :
                res[i,j,:3] = pts[i,j,:3]
                normal = adjacent_slope_normal(pts[:,:,:3],i,j)
                res[i,j,3:6] = normal_to_facet_tgt(pts[i,j,:3], normal)
                
    res = res.reshape(nx * ny, 6)
    
    return res[res[:,0] != 0]

def from_nc_to_polygon_cam_tgt(
    polygon: str,
    ds_s2m_path : str,
    fic_res : str,
    fic_ds_topo : str,
    fic_res_topo : str
    ) -> np.ndarray:
    

    filepath = Path(polygon)
    
    polygon_read = from_geojson(filepath.read_text())

    ds_s2m = xr.open_dataset(ds_s2m_path)
    
    mask = filter_dataset_by_polygon(ds_s2m, polygon_read, x_coord="x", y_coord="y")
    
    nx = ds_s2m.x.size
    ny = ds_s2m.y.size
    xx,yy = np.meshgrid(ds_s2m.x.values,ds_s2m.y.values)
 
    pts = np.zeros((ny,nx,4))
    pts[:,:,0] = xx
    pts[:,:,1] = yy
    pts[:,:,2] = ds_s2m['zs'].values + 5 # Les caméras sont elles bien positionnées ??
    pts[:,:,3] = mask
    
    res = chose_polygone(pts)
    np.savetxt(fic_res,res,delimiter = '\t',newline='\n')
    
    ds_topo = xr.open_dataset(fic_ds_topo)
    
    topo_polygon_params = topo_params(ds_topo,res[:,:3])
    
    np.savetxt(fic_res_topo,topo_polygon_params,delimiter = '\t',newline='\n')
    
    print("transect param topo OK")
    
####################################################### Polygon #############################################

from shapely import Polygon, to_geojson, from_geojson
from pathlib import Path

def creating_polygon(
    path_to_poly : str,
    x1 : int,
    x2 : int,
    x3 : int,
    x4 : int,
    y1 : int,
    y2 : int,
    y3 : int,
    y4 : int,  
    ):
    
    polygon = Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
    
    filepath = Path(path_to_poly)
    filepath.write_text(to_geojson(polygon))
    
####################################################### Les bassins versants ##############################################

import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
from shapely import Polygon, to_geojson, from_geojson

def selecting_bv(
    ):

    # Charger le shapefile national
    bv = gpd.read_file("/home/barroisl/Bassin versant/BassinVersantTopographique_FXX-shp/BassinVersantTopographique_FXX.shp")

    # Bounding box approximative de la Chartreuse (epsg 32632)
    chartreuse_bbox = box(235000, 5010000, 265000, 5045000)

    # Reprojeter si nécessaire
    if bv.crs.to_epsg() != 32632:
        bv = bv.to_crs(epsg=32632)
    
    # Filtrer les bassins versants qui intersectent la Chartreuse
    bv_chartreuse = bv[bv.intersects(chartreuse_bbox)]

    #bv_chartreuse.to_file("/home/barroisl/Bassin versant/BassinVersantTopographique_FXX-shp/bassins_chartreuse.gpkg", driver="GPKG")
    
    for i in range(len(bv_chartreuse['geometry'].values)):
        fic_bv = "/home/barroisl/Bassin versant/chartreuse_%s.geojson" %i
        filepath = Path(fic_bv)

        # To GeoJSON
        filepath.write_text(to_geojson(bv_chartreuse['geometry'].values[i]))

########################################### Créer l'arborescence de l'expérience pour quand on change de simu ##############################################

########################################### Température indéxé sur la topography ####################################

@nb.njit()
def altit_T(dem, T0, z0, lapse_rate = -0.0065):
    
    return T0+lapse_rate*(dem-z0)


