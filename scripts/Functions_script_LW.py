#!/usr/bin/env python
# coding: utf-8

# In[ ]:

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
    points   : np.ndarray de forme (N, 3) — colonnes [x, y, z]
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

################################ Extraction de paramètres topo le long d'un transect ###################

def topo_params(ds: xr.Dataset,
    points: np.ndarray,
) -> np.ndarray:
    
    svf_transect = interpolate(ds_chartreuse,pts,var_name = 'svf')
    svf_LSLOPE_transect = interpolate(ds_chartreuse,pts,var_name = 'svf_LSLOPE')
    aspect_transect = interpolate(ds_chartreuse,pts,var_name = 'aspect')
    slope_transect = interpolate(ds_chartreuse,pts,var_name = 'slope')
    elevation_transect = interpolate(ds_chartreuse,pts,var_name = 'elevation')

    return np.column_stack([svf_transect, svf_LSLOPE_transect, aspect_transect,slope_transect,                          elevation_transect])
    
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
            if not mask[j,i]:
                # Ts en K, arrondi à 0.1
                ts_arr = round(ts[j,i],1)
                if ts_arr not in d_ts:
                    d_ts[ts_arr] = []
                d_ts[ts_arr].append( (j,i,sc[j,i],zs[j,i]) )
                
    print("d_ts features OK : ", len(d_ts),min(list(d_ts)),max(list(d_ts)))

    lignes_mtls =  []
    
    #cwd = os.getcwd() # répertoire actuel
    cwd='${HTRDR_ATMOSPHERE_SPK}'
    for ts_arr in sorted(d_ts): # liste triée des Ts arrondies            
                    
        if d_ts[ts_arr][2] > 0 :            
            mrumtl = 'snow'
                    
        elif d_ts[3] < 1500 :
            mrumtl = 'forest_green'

        elif d_ts[3] > 1500 and d_ts[3] < 2000 :
            mrumtl = 'grass'

        else :
            mrumtl = 'limestone'
                    
        nom_mat = mrumtl + str(ts_arr).replace('.','_') # ex.'snow238_1'
        lignes_obj.append(f'usemtl air:{nom_mat}\n')

        # Boucles sur les facettes de température ts_arr d'indice inférieur à frac_neige*len(d_ts)
        # dans d_ts.sorted() --> Toutes les facettes recouvertes de neige
        #for j,i in d_ts[ts_arr][indices_snow]:# points avec la Ts arrondie ts_arr
        for j,i in ts_arr :
            # 2 triangles en ht à dr de (j,i), sens aig. montre (cf.
            #  htrdr-Atmosphere-Starter-Pack-0.8.0/models/plane.obj)

            lignes_obj.append( # en haut à droite du point j,i
                f'f {d_num[(j,i)]} ' # j,i
                f'{d_num[(j+1,i)]} ' # + haut que j,i
                f'{d_num[(j,i+1)]}\n') # + à droite que j,i
            lignes_obj.append( # même case, plus en haut à droite
                f'f {d_num[(j,i+1)]} ' # + à droite que j,i
                f'{d_num[(j+1,i)]} ' # + haut que j,i
                f'{d_num[(j+1,i+1)]}\n')# + haut + à droite

        lignes_obj.append('\n') # ligne vide

        lignes_mtls.append(f'{nom_mat} '
            f'"{cwd}/materials/legacy/{mrumtl}.mrumtl" {ts_arr}\n')

    # Boucles sur les facettes de température ts_arr d'indice supérieur à frac_neige*len(d_ts)
    # dans d_ts.sorted() --> Toutes les facettes non recouvertes de neige
        
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
    x_dim: str = "x",
    y_dim: str = "y",
) -> xr.Dataset:
    """
    Select a dataset within [x_min, x_max] × [y_min, y_max].
    """
    return ds.sel(
        {
            x_dim: ds[x_dim][(ds[x_dim] >= x_min) & (ds[x_dim] <= x_max)],
            y_dim: ds[y_dim][(ds[y_dim] >= y_min) & (ds[y_dim] <= y_max)],
        }
    )

def from_s2m_to_dem_ts_sc(
    date : datetime.datetime,
    ds : xr.Dataset,
    xmin : int,
    xmax : int,
    ymin : int,
    ymax : int,
    fic_obj : str,
    fic_ds : str,
    fic_mtls : str,
):
    
    ds_sel = ds.sel(time=date)
    ds_sel_cropted = select_spatial_extent(ds_sel,xmin,xmax,ymin,ymax)
    
    xs = ds_sel_cropted['x'].values.astype(np.float64)
    ys = ds_sel_cropted['y'].values.astype(np.float64)
    zs = ds_sel_cropted['elevation'].values.astype(np.float64)
    cs = ds_sel_cropted['DSN_T_ISBA'].values.astype(np.float64)
    ts = ds_sel_cropted['T_ISBA'].values.astype(np.float64)

    if xs[-1] < xs[0]:
        xs  = xs[::-1].copy()
        zs = zs[:, ::-1].copy()
        cs = cs[:, ::-1].copy()
        ts = ts[:, ::-1].copy()
        
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
    
    from_dem_ts_sc_to_obj(xs,ys,zs,cs,ts,fic_obj,fic_mtls)
    
############################################ From position to luminence (surface) #########################

H = 6.62607015e-34   # Planck constant      [J·s]
C = 2.99792458e8     # Speed of light        [m/s]
KB = 1.380649e-23    # Boltzmann constant    [J/K]

@nb.njit(cache = True)
def Planck_law(
    wavelength: float | np.ndarray, 
    temperature: float) -> float | np.ndarray:
              
    return (2 * H * C**2 / wavelength**5) / (np.exp(H * C / (wavelength * KB * temperature)) - 1)  

def from_position_to_luminence(abs_nd : np.ndarray, 
                               ds : xr.Dataset 
                              )-> np.ndarray :
    """
    [Input]
    - abs_nd : ndarray (Nbre_chemin, 4) x,y,z absorption & lambda
    - ds de température de surface de l'.obj
    """
    a,b = positions_abs.shape
    ts_interp = interpolate(ds,abs_nd[:,:3],var_name = 'ts')
    
    return Planck_law(abs_nd[:,3],ts_interp)

