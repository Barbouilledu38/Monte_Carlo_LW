#!/usr/bin/env python

############################# Imports ###############################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
import numba as nb
from numba import prange
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
import math
import time
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib import pylab
from tqdm import tqdm
import os
import matplotlib.ticker as ticker
import numpy as np
import rasterio
import netCDF4 as nc
import os
import numba
from tqdm import tqdm
import random
from pyproj import Transformer
import xarray as xr
import csv
from matplotlib.patches import FancyArrowPatch
from ipywidgets import FloatSlider, interact
from matplotlib.widgets import Button, Slider
from ipywidgets import interact

############################## Class ###########################

################## Coordinates stuff ##################################

def convert_epsg_pts(xs,ys, epsg_src=4326, epsg_tgt=32632):
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

################### Interpolation ########################################

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

################### Class mc_set #####################################

@nb.njit()
def get_collisions(index, collision):
    collisions_index = collision[collision[:,0]==index][:,1:]
    return collisions_index

@nb.njit()
def get_pos_abs(coll):
    collision_finale = coll[coll[:,0] != 2][0,1:4]
    return collision_finale

@nb.njit()
def get_reflections(collisions):
    reflections = collisions[collisions[:,0] == 2][:,1:4]
    return reflections

@nb.njit()
def get_dist(Pos_abs,pos_camera):
    return np.sqrt(np.sum((Pos_abs[:2]-pos_camera[:2])**2))

@nb.njit()
def get_full_dist(Pos_abs,pos_camera):
    return np.sqrt(np.sum((Pos_abs-pos_camera)**2))

@nb.njit()
def get_Azimuth(Pos_abs,pos_camera):
    return math.atan2(Pos_abs[1]-pos_camera[1],Pos_abs[0]-pos_camera[0])

@nb.njit()
def get_zenithal(Pos_abs,pos_camera):
    return math.atan2(Pos_abs[2]-pos_camera[2],get_dist(Pos_abs,pos_camera))

class MC_chemin:

    """
    Class pour un chemin. Donne son index, sa finallité (surf ou atm), son nombre de diffusions,
    de réflexion, son poids en triant dans le 2D array collisions
    """
    
    def __init__(self, data_path,pos_camera,W_tot): 
        
        self.Index = data_path[0]
        self.Abs_surf = np.isclose(data_path[1],0.)
        self.Abs_atm = np.isclose(data_path[1],1.)
        self.Space = np.isclose(data_path[1],2.)

        self.wlen = data_path[7]
        self.Nbre_refl = data_path[2]
        # self.Nbre_diff = chemin[3] : pas de diffusion LW
        self.W = data_path[6] #/W_tot pour avoir la participation relative
        self.relative_weight = self.W/W_tot
        self.pos_camera = pos_camera

        #self.collision_self = get_collisions(self.Index, collisions)
        self.Pos_abs = np.array(data_path[3:6])
        #self.reflections = get_reflections(self.collision_self)
        self.Dist = get_dist(self.Pos_abs,pos_camera)
        self.full_dist = get_full_dist(self.Pos_abs,pos_camera)
        self.Azimuth = get_Azimuth(self.Pos_abs,pos_camera)
        self.zenithal = get_zenithal(self.Pos_abs,pos_camera)
        
class MC_Set:
    
    """
    Classe sortant d'une simulation de MC. Statistique sur les chemins et figures
    """
    
    def __init__(self,data_path,pos_camera):
        
        # Caractéristique simu
        self.Nbre_photon = data_path.shape[0]
        self.pos_camera = pos_camera
        
        # Statistique de chemin   
        self.Nbre_reflexion = np.sum(data_path[:,2])
        self.Nbre_moyen_reflexion_exc = self.Nbre_reflexion/data_path[data_path[:,2] != 0].shape[0]
        self.Nbre_surf = data_path[(data_path[:,1] == 0.) & (data_path[:,2] == 0.)].shape[0]
        self.Nbre_atm = data_path[(data_path[:,1] == 1)  & (data_path[:,2] == 0)].shape[0]
        self.Nbre_reflect_surf = data_path[(data_path[:,1] == 0) & (data_path[:,2] != 0)].shape[0]
        self.Nbre_reflect_atm = data_path[(data_path[:,1] == 1)  & (data_path[:,2] != 0)].shape[0]            
        self.Nbre_space = data_path[data_path[:,1] == 2].shape[0]     
        
        # Poids par catégorie
        
        self.W_tot = np.sum(data_path[:,6])
        self.W_atm = np.sum(data_path[(data_path[:,1] == 1) & (data_path[:,2] == 0)][:,6])/self.W_tot
        
        self.W_reflechis_atm = np.sum(data_path[(data_path[:,2] != 0) & (data_path[:,1] == 1)][:,6])/self.W_tot
        self.W_reflechis_surf = np.sum(data_path[(data_path[:,2] != 0) & (data_path[:,1] == 0)][:,6])/self.W_tot
        
        if len(data_path[data_path[:,1] == 0]) > 0 :
            self.W_surf = np.sum(data_path[(data_path[:,1] == 0) & (data_path[:,2] == 0)][:,6])/self.W_tot
        else :
            self.W_surf = 0
        
        # Collection de classe chemin
        self.paths = [MC_chemin(data_path[i],pos_camera, self.W_tot) for i in range(data_path.shape[0])]        
    
    # Data methods ------------------------------------------------------------------------------------
        
    def class_selfs(self):
        print("La simu : \n",
              "\n pos_camera : x = %d\t" %self.pos_camera[0] \
              + "y = %d\t" %self.pos_camera[1] + "z = %d\t" %self.pos_camera[2],
             "\n Nbre_photon : %d" %self.Nbre_photon,
              
             "\n \n Stat sur les chemins : \n",
             "\n Nbre_moyen_reflexion_exc %.2f" %self.Nbre_moyen_reflexion_exc,
             "\n Nbre_reflexion : %d" %self.Nbre_reflexion,
             "\n Nbre_surf : %d" %self.Nbre_surf,
             "\n Nbre_atm : %d" %self.Nbre_atm,
             "\n Nbre_space : %d" %self.Nbre_space,
             "\n Nbre_reflect_surf : %d" %self.Nbre_reflect_surf,
             "\n Nbre_reflect_atm : %d" %self.Nbre_reflect_atm,
              
             "\n \n Poids par catégorie : \n",
             "\n W_tot : %f" %self.W_tot,
             "\n W_atm : %f" %self.W_atm,
             "\n W_surf : %f" %self.W_surf,
             "\n W_reflechis_atm : %f" %self.W_reflechis_atm,
             "\n W_reflechis_surf : %f" %self.W_reflechis_surf)
        
    def get_coord(self, coord, booli):
        
        if coord == 'x':
            indice = 0
        elif coord == 'y':
            indice = 1
        elif coord == 'z':
            indice = 2
            
        return [self.paths[i].Pos_abs[indice] for i in range(len(self.paths)) \
                if self.paths[i].Abs_surf == booli]
        
    def distance_to_plane(self,phi,Pos_abs,max_dist):
        
        n_plan = np.array([-np.sin(phi),np.cos(phi)])
        dist_ortho = abs(np.dot(n_plan,(Pos_abs[:2]-self.pos_camera[:2])))/np.sqrt(np.dot(n_plan,n_plan))
        
        if dist_ortho > max_dist :
            return False
        
        else :
            return True
        
    def projection_on_plane(self,phi,Pos_abs):
        
        t_plan = np.array([np.cos(phi), np.sin(phi)])
        
        return np.dot(t_plan,(Pos_abs[:2] - self.pos_camera[:2]))/np.sqrt(np.dot(t_plan,t_plan))
    
    
    def coord_proj_plane(self, phi, max_dist):
        
        a = len(self.paths)
            
        Surface = [np.array([self.projection_on_plane(phi,self.paths[i].Pos_abs),self.paths[i].Pos_abs[2],self.paths[i].W]) \
                   for i in range(a) if \
                   (self.distance_to_plane(phi,self.paths[i].Pos_abs,max_dist) == True and \
                    self.paths[i].Abs_surf == True)]
        Atm = [np.array([self.projection_on_plane(phi,self.paths[i].Pos_abs),self.paths[i].Pos_abs[2],self.paths[i].W]) for i in range(a) if \
                   (self.distance_to_plane(phi,self.paths[i].Pos_abs,max_dist) == True and \
                    self.paths[i].Abs_surf == False)]
    
        return Surface, Atm
    
    def z_profile(self, Surface):
        
        z = [self.paths[i].Pos_abs[2] for i in range(len(self.paths)) if self.paths[i].Abs_surf == Surface and \
            self.paths[i].Space == False]
        W = [self.paths[i].W for i in range(len(self.paths)) if self.paths[i].Abs_surf == Surface and \
            self.paths[i].Space == False]
        
        if len(z) >0 :
                
            paired = list(zip(z, W/self.W_tot))
            paired_sorted = sorted(paired, key=lambda x: x[0])
            z_sorted, var_sorted = zip(*paired_sorted)    
            
        else :
            z_sorted, var_sorted = [np.nan, np.nan]
        
        return z_sorted, var_sorted
    
    def dist_profile(self, Surface):
        
        Dist = [self.paths[i].Dist for i in range(len(self.paths)) if self.paths[i].Abs_surf == Surface and \
            self.paths[i].Space == False]
        W = [self.paths[i].W for i in range(len(self.paths)) if self.paths[i].Abs_surf == Surface and \
            self.paths[i].Space == False]
        
        if len(W) >0 :

            paired = list(zip(Dist, W/self.W_tot))
            paired_sorted = sorted(paired, key=lambda x: x[0])
            z_sorted, var_sorted = zip(*paired_sorted)    
        
        else :
            z_sorted, var_sorted = [np.nan, np.nan]
        
        
        return z_sorted, var_sorted
    
    def Z_radiance_tot(self,z_profile,T_profile):

        sigma = 5.67*(10**(-8))
        radiance_profile = sigma*(T_profile**4)
        
        diff_liste = abs(radiance_profile - (self.W_atm+self.W_reflechis_atm)*(self.W_tot/self.Nbre_photon))
        
        indice_z = diff_liste.argmin()
                    
        return z_profile[indice_z]
    
    def Matrice_cumuls(self,z_max,r_max,n_z,n_r, surface):
        
        Mat_cumul = np.zeros((n_z,n_r))
        
        zs = np.linspace(0,z_max,n_z)
        rs = np.linspace(0,r_max,n_r)
        
        Nbre_tot = self.Nbre_photon - self.Nbre_space
                
        for i in range(n_z):
            for j in range(n_r):
                                
                Mat_cumul[i,j] = len([self.paths[k].Dist for k in range(self.Nbre_photon) if \
                                        (self.paths[k].Abs_surf == surface) and (self.paths[k].Space == False) and \
                                           (self.paths[k].Dist < rs[j]) and \
                                           (self.paths[k].Pos_abs[2] - self.pos_camera[2] < zs[i])])/Nbre_tot
        
        return Mat_cumul,zs,rs
    
    def quartil_cumul_collision(self,type_,quantil):
        
        pathes = [self.paths[i] for i in range(len(self.paths)) if \
             self.paths[i].Abs_surf == True and self.paths[i].Space == False]
        
        if len(pathes) > 0 :
        
            if type_ == "dist" :

                Dist = np.array([pathes[i].Dist for i in range(len(pathes))])
                res = np.percentile(Dist, quantil)

            elif type_ == "alt" :

                Alt = np.array([pathes[i].Pos_abs[2] for i in range(len(pathes))])
                res = np.percentile(Alt, quantil)         

            return res
        
        else :
            
            return np.nan
    
    
    def quartil_cumul_flux(self,type_,quantil):
        
        pathes = [self.paths[i] for i in range(len(self.paths)) if \
             self.paths[i].Abs_surf == True and self.paths[i].Space == False]   
        
        if len(pathes) > 0 :
        
            if type_ == "dist" :

                Dist = np.array([pathes[i].Dist for i in range(len(pathes))])
                W = np.array([pathes[i].W/self.Nbre_photon for i in range(len(pathes))])

                paired = list(zip(Dist, W))
                paired_sorted = sorted(paired, key=lambda x: x[0])
                dist_sorted, W_sorted = zip(*paired_sorted)
                W_sorted_cum = np.cumsum(W_sorted)
                index_90 = np.argmax(W_sorted_cum > (quantil/100)*W_sorted_cum[-1])
                return dist_sorted[index_90]

            elif type_ == "alt" :


                Alt = np.array([pathes[i].Pos_abs[2] for i in range(len(pathes))])
                W = np.array([pathes[i].W/self.Nbre_photon for i in range(len(pathes))])

                paired = list(zip(Alt, W))
                paired_sorted = sorted(paired, key=lambda x: x[0])
                Alt_sorted, W_sorted = zip(*paired_sorted)
                W_sorted_cum = np.cumsum(W_sorted)
                index_90 = np.argmax(W_sorted_cum > (quantil/100)*W_sorted_cum[-1])
                return Alt_sorted[index_90]
            
        else :
            return np.nan

    def Pos_surf(self,
                ds : xr.Dataset):
        
        sigma = 5.67e-8
        pathes = [self.paths[i] for i in range(len(self.paths)) if \
             self.paths[i].Abs_surf == True and self.paths[i].Space == False]
        
        if len(pathes) > 0 :
            
            Pos = np.array([pathes[i].Pos_abs[:2] for i in range(len(pathes))])
            Ts = interpolate(
                    ds = ds,
                    points = Pos,
                    var_name ='ts',
                    x_dim = "x",
                    y_dim = "y",
                    )

            return np.sum(sigma*Ts**4)/self.Nbre_photon
        
        else :
            return np.nan
        
    def T_surf(self,
                ds : xr.Dataset):
        
        sigma = 5.67e-8
        pathes = [self.paths[i] for i in range(len(self.paths)) if \
             self.paths[i].Abs_surf == True and self.paths[i].Space == False]
        
        if len(pathes) > 0 :
            
            Pos = np.array([pathes[i].Pos_abs[:2] for i in range(len(pathes))])
            Ts = interpolate(
                    ds = ds,
                    points = Pos,
                    var_name ='ts',
                    x_dim = "x",
                    y_dim = "y",
                    )

            return Ts
        
        else :
            return np.nan
        
    def distrib_lambda(self,surface):
        
        if surface :
            pathes = [self.paths[i] for i in range(len(self.paths)) if \
                      self.paths[i].Abs_surf == surface and self.paths[i].Space == False]
            
        else :
            pathes = [self.paths[i] for i in range(len(self.paths))]
            
        return [path.wlen for path in pathes]
        
    def lambda_exchange(self,z):
        
        pathes = [self.paths[i] for i in range(len(self.paths)) if \
                      self.paths[i].Abs_atm == True]
                      
        dz = z[1]-z[0]
        res = [[] for i in range(z.size -1)]
        for path in pathes :
            index_z = int(path.Pos_abs[2]//dz)
            if index_z >= (z.size-1) :
                index_z = int(len(res)-1)
            res[index_z].append(path.wlen/1000) # µm
            
        return res
        
    def energy_exchange(self,z):
        
        pathes = [self.paths[i] for i in range(len(self.paths)) if \
                      self.paths[i].Abs_atm == True]
                      
        dz = z[1]-z[0]
        res = np.zeros(z.size-1)
        for path in pathes :
            index_z = int(path.Pos_abs[2]//dz)
            if index_z >= (z.size-1) :
                index_z = int(len(res)-1)
                
            res[index_z] += path.W/self.Nbre_photon
            
        return res
        

    
# Plot methods ------------------------------------------------------------------------------------

    def density_dist_profile_wlen(self, ax, surface, wlen_1, wlen_2, color, name, linestyle):
        
        pathes = [self.paths[i] for i in range(len(self.paths)) if \
                  self.paths[i].Abs_surf == surface and self.paths[i].Space == False\
                  and self.paths[i].wlen > wlen_1 and self.paths[i].wlen < wlen_2]
        
        Dist = sorted([path.full_dist for path in pathes])
        
        if surface :
            ax.plot(Dist,1-np.cumsum(np.ones(len(Dist)))/len(Dist), c = color, linestyle = linestyle)
        
        else :
            ax.plot(Dist,1-np.cumsum(np.ones(len(Dist)))/len(Dist), c = color, label = name, linestyle = linestyle)
   
        ax.spines[['left','right', 'top', 'bottom']].set_visible(False)
        
        if surface :
            type_abs = "surface"
        else :
            type_abs = "atmosphère"
        ax.set_ylabel(f'P(l > x | atm : - ; surf : -- )')
        ax.set_xlabel('3D distance of absorption (m)')
    
    def density_dist_profile(self, ax, color, name):
        
        pathes = [self.paths[i] for i in range(len(self.paths)) if\
                 self.paths[i].Abs_surf == False]
        
        Dist = sorted([path.full_dist for path in pathes])
        
        ax.plot(Dist,np.cumsum(np.ones(len(Dist)))/len(Dist), c = color, label = name)
   
        ax.spines[['left','right', 'top', 'bottom']].set_visible(False)
        ax.set_ylabel('Cumulative density')
        ax.set_xlabel('3D distance of absorption (m)')
        
    def dist_profile(self, ax, surface, color, name, linestyle):
        
        pathes = [self.paths[i] for i in range(len(self.paths)) if\
                 self.paths[i].Space == False and self.paths[i].Abs_surf == surface]
        
        Dist = sorted([path.full_dist for path in pathes])
        
        if surface :
            ax.plot(Dist,1-np.cumsum(np.ones(len(Dist)))/len(Dist), c = color, linestyle = linestyle)
        else :
            ax.plot(Dist,1-np.cumsum(np.ones(len(Dist)))/len(Dist), c = color, label = name, linestyle = linestyle)
        ax.spines[['left','right', 'top', 'bottom']].set_visible(False)
        ax.set_ylabel(f'P(l > x | atm - ; surf --)')
        ax.set_xlabel('3D distance of absorption (m)')
        
        
    def plot_atm_radiance_profile_4(self,ax,color,Surface, name):

        z = [self.paths[i].Pos_abs[2] - self.pos_camera[2] for i in range(len(self.paths)) if \
             self.paths[i].Abs_surf == Surface and self.paths[i].Space == False]
        
        z_sorted = sorted(z)
        
        Collisions = np.ones((1,len(z)))[0]
        
        Nbre_abs = self.Nbre_photon - self.Nbre_space
        ax.plot(np.cumsum(Collisions)/Nbre_abs,z_sorted,c = color, linestyle = 'solid', \
                 label = name)
    
    def plot_atm_radiance_profile_Planck(self,ax,color,Surface = False):
        
        pathes = [self.paths[i] for i in range(len(self.paths)) if \
             self.paths[i].Abs_surf == Surface and self.paths[i].Space == False]

        z = [pathes[i].Pos_abs[2] - self.pos_camera[2] for i in range(len(pathes))]
        
        norm = [np.sqrt(np.dot(pathes[i].Pos_abs - self.pos_camera,self.paths[i].Pos_abs - self.pos_camera)) for i in \
                 range(len(pathes))]
        
        ctheta = [np.dot(pathes[i].Pos_abs - self.pos_camera,np.array([0,0,1]))/norm[i] for i in \
                 range(len(pathes))]
        
        W = [ctheta[i]*pathes[i].W/np.pi for i in range(len(pathes))]

        ax.scatter(W,z,c = color)
    
    def plot_atm_radiance_profile_3(self,ax,color,Surface, name):

        z = [self.paths[i].Pos_abs[2] - self.pos_camera[2] for i in range(len(self.paths)) if \
             self.paths[i].Abs_surf == Surface and self.paths[i].Space == False]
        W = [self.paths[i].W for i in range(len(self.paths)) if self.paths[i].Abs_surf == Surface and \
            self.paths[i].Space == False]
        
        paired = list(zip(z, W/self.W_tot))
        paired_sorted = sorted(paired, key=lambda x: x[0])
        z_sorted, var_sorted = zip(*paired_sorted)

        ax.plot(np.cumsum(var_sorted),z_sorted,c = color, linestyle = 'solid', \
                 label = name)
                             
    def dist_weight_4(self, ax, Surface, color, name):
                
        Dist = [self.paths[i].Dist for i in range(len(self.paths)) if self.paths[i].Abs_surf == Surface and \
            self.paths[i].Space == False]
        Dist_sorted = sorted(Dist)
    
        Collisions = np.ones((1,len(Dist)))[0]                  

        Nbre_abs = self.Nbre_photon - self.Nbre_space
        ax.plot(Dist_sorted,np.cumsum(Collisions)/Nbre_abs, c = color, linestyle = 'solid',\
                label = name)
        
    def dist_weight_3(self, ax, Surface, color, name):
        
                
        Dist = [self.paths[i].Dist for i in range(len(self.paths)) if self.paths[i].Abs_surf == Surface  and \
            self.paths[i].Space == False]
        
        W = [self.paths[i].W for i in range(len(self.paths)) if self.paths[i].Abs_surf == Surface and \
            self.paths[i].Space == False]                
    
        paired = list(zip(Dist, W))
        paired_sorted = sorted(paired, key=lambda x: x[0])
        Dist_sorted, W = zip(*paired_sorted)    

        ax.plot(Dist_sorted,np.cumsum(W)/self.W_tot, c = color, linestyle = 'solid',\
                label = name)

    def plot_atm_radiance_profile_2(self,ax,color,Surface, name,vmin,vmax):
        
        z = [self.paths[i].Pos_abs[2] - self.pos_camera[2] for i in range(len(self.paths)) if self.paths[i].Abs_surf == Surface and \
            self.paths[i].Space == False]
        W = [self.paths[i].W for i in range(len(self.paths)) if self.paths[i].Abs_surf == Surface and \
            self.paths[i].Space == False]
        
        cax = ax.scatter(W,z, s = 0.5,c = W, vmin = vmin, vmax= vmax, cmap = 'Reds', \
                         marker = 'o', label = 'Chemins ' + name)
        
        ax2 = ax.twiny()            
        paired = list(zip(z, W/self.W_tot))
        paired_sorted = sorted(paired, key=lambda x: x[0])
        z_sorted, var_sorted = zip(*paired_sorted)

        ax2.plot(np.cumsum(var_sorted),z_sorted,c = color, linestyle = 'solid', \
                 label = 'Cumulative relative flux ' + name)
        Nbre_photon_abs = self.Nbre_photon - self.Nbre_space
        ax2.plot(np.cumsum(np.ones((1,len(var_sorted))))/Nbre_photon_abs,z_sorted,c = color, \
                   label = 'Cumulative count ' + name, linestyle = 'dashed')

        ax2.spines[['left','right', 'top', 'bottom']].set_visible(False)
        
        ax.set_xlabel('Thermal flux ($W/m^2$)')
        ax.set_ylabel('Altitude of absorption (m)')
        ax.spines[['left','right', 'top', 'bottom']].set_visible(False)
        
        ax2.set_xlim((0,1))
        
        ax2.legend()
        #ax.legend()
       
    def temperature_profile(self):
    
        directory = "/home/barroisl/edstar/Simus/"
        fname = "ecrad_opt_prop.txt"
        data_path = np.loadtxt(directory+fname, dtype = np.float32,skiprows = 172)

        Temperatures = data_path[:165]-100
        Altitudes = data_path[165:330]

        return Temperatures, Altitudes
    

    def plot_atm_profile_2_integer_spectral(self,ax,color,vmin,vmax):
        
        directory = "/home/barroisl/edstar/Simus/"
        fname = "ecrad_opt_prop.txt"
        data_path = np.loadtxt(directory+fname, dtype = np.float32,skiprows = 172)

        Temperatures = data_path[:165]
        Altitudes = data_path[165:330]
        
        Temperatures = Temperatures[Altitudes > self.pos_camera[2]]
        Altitudes = Altitudes[Altitudes > self.pos_camera[2]] - self.pos_camera[2]

        sigma = 5.67*10**(-8)
        plt.plot(sigma*Temperatures**4,Altitudes,'r')
        
        cax = ax.scatter(sigma*Temperatures**4,Altitudes, s = 0.5,c = 'blue', \
                         vmin = vmin, vmax= vmax, \
                         marker = 'o', label = 'SB')       
        
        """
        Temperatures, Altitudes = self.temperature_profile()
        
        Paths_liste = [self.paths[i] for i in range(len(self.paths)) if \
                      self.paths[i].Abs_atm == True and self.paths[i].Space == False]
        
        z = np.array([Paths_liste[i].Pos_abs[2] for i in range(len(Paths_liste))])
        
        sigma = 5.67*10**(-8)
        W = sigma*np.interp(z, Altitudes, Temperatures)**4
        
        cax = ax.scatter(W,z, s = 0.5,c = W, vmin = vmin, vmax= vmax, cmap = 'Blues', \
                         marker = 'o', label = 'SB')       
        """
        
    def plot_atm_radiance_profile_2_clouds(self,ax,color,z_bottom_clouds,z_top_clouds,name,vmin,vmax):
        
        Temperatures, Altitudes = self.temperature_profile()
        
        Paths_liste = [self.paths[i] for i in range(len(self.paths)) if \
                      self.paths[i].Abs_atm == True and self.paths[i].Space == False]
           
        if self.pos_camera[2] < z_bottom_clouds :
            z = np.array([Paths_liste[i].Pos_abs[2] if Paths_liste[i].Pos_abs[2] < z_bottom_clouds else z_bottom_clouds \
                 for i in range(len(Paths_liste))])
            
        elif self.pos_camera[2] > z_bottom_clouds and self.pos_camera[2] < z_top_clouds:
            z = np.array([Paths_liste[i].pos_camera[2] for i in range(len(Paths_liste))])  
            
        else :
            z = np.array([Paths_liste[i].Pos_abs[2] for i in range(len(Paths_liste))])
        
        sigma = 5.67*10**(-8)
        W = sigma*np.interp(z, Altitudes, Temperatures)**4
        
        cax = ax.scatter(W,z, s = 0.5,c = W, vmin = vmin, vmax= vmax, cmap = 'Reds', \
                         marker = 'o', label = 'Chemins ' + name)
        
        ax2 = ax.twiny()            
        paired = list(zip(z, W/self.W_tot))
        paired_sorted = sorted(paired, key=lambda x: x[0])
        z_sorted, var_sorted = zip(*paired_sorted)

        ax2.plot(np.cumsum(var_sorted),z_sorted,c = color, linestyle = 'solid', \
                 label = 'Cumulative relative flux ' + name)
        Nbre_photon_abs = self.Nbre_photon - self.Nbre_space
        ax2.plot(np.cumsum(np.ones((1,len(var_sorted))))/Nbre_photon_abs,z_sorted,c = color, \
                   label = 'Cumulative count ' + name, linestyle = 'dashed')

        ax2.spines[['left','right', 'top', 'bottom']].set_visible(False)
        
        ax.set_xlabel('Thermal flux ($W/m^2$)')
        ax.set_ylabel('Altitude of absorption (m)')
        ax.spines[['left','right', 'top', 'bottom']].set_visible(False)
        
        ax2.set_xlim((0,1.5))
        
        ax2.legend()
        #ax.legend()
            
    def plot_atm_radiance_profile(self,axs,Variable, save_title):
        
        sigma = 5.67*10**(-8)
        
        # Atm absorbed
        
        z = [self.paths[i].Pos_abs[2] for i in range(len(self.paths)) if self.paths[i].Abs_surf == False]
        if Variable == 'W' :
            var = [self.paths[i].W for i in range(len(self.paths)) if self.paths[i].Abs_surf == False]
            xlabel = 'Weight ($W/m^2$)'
                        
            ax2 = axs[0].twiny()            
            paired = list(zip(z, var/self.W_tot))
            paired_sorted = sorted(paired, key=lambda x: x[0])
            z_sorted, var_sorted = zip(*paired_sorted)

            ax2.plot(np.cumsum(var_sorted),z_sorted,c = 'r')
            
            ax2.plot(np.cumsum(np.ones((1,len(var))))/len(var),z_sorted,c = 'b', \
                       label = 'Cumulative count')
            
            ax2.spines[['left','right', 'top', 'bottom']].set_visible(False)
            
        elif Variable == 'T' :
            var = [(self.paths[i].W/sigma)**(1/4) for i in range(len(self.paths)) if self.paths[i].Abs_surf == False]
            xlabel = 'Température de radiance'
            
        cax = axs[0].scatter(var,z, s = 0.5,c = 'k', marker = 'o')
        
        axs[0].set_title('Absorbed by Atmosphere : %.2f ($W/m^2$)' %(self.W_atm*self.W_tot))
        
        # Surface absorbed
        
        if len([self.paths[i].Dist for i in range(len(self.paths)) \
                if self.paths[i].Abs_surf == True]) != 0 :
        
            z = [self.paths[i].Pos_abs[2] for i in range(len(self.paths)) if self.paths[i].Abs_surf == True]
            if Variable == 'W' :
                var = [self.paths[i].W for i in range(len(self.paths)) if self.paths[i].Abs_surf == True]
                xlabel = 'Weight ($W/m^2$)'

                ax2 = axs[1].twiny()            
                paired = list(zip(z, var/self.W_tot))
                paired_sorted = sorted(paired, key=lambda x: x[0])
                z_sorted, var_sorted = zip(*paired_sorted)

                ax2.plot(np.cumsum(var_sorted),z_sorted, c = 'r', \
                            label = 'Cumulative relative weight')

                ax2.plot(np.cumsum(np.ones((1,len(var))))/len(var),z_sorted,c = 'b', \
                           label = 'Cumulative count')

                ax2.spines[['left','right', 'top', 'bottom']].set_visible(False)

                ax2.legend()

            elif Variable == 'T' :
                var = [(self.paths[i].W/sigma)**(1/4) for i in range(len(self.paths)) if self.paths[i].Abs_surf == True]
                xlabel = 'Température de radiance'
            
        cax = axs[1].scatter(var,z, s = 0.5,c = 'k', marker = 'o', label = Variable)  
        
        axs[1].set_title('Absorbed by Surface : %.2f ($W/m^2$)' %(self.W_surf*self.W_tot))
        
        for ax in axs :
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Altitude of absorption (m)')
            ax.spines[['left','right', 'top', 'bottom']].set_visible(False)

        plt.savefig(save_title)
        
    def histogramme(self,ax,variable) :
        
        if variable == 'Dist':
            var = [self.paths[i].Dist for i in range(len(self.paths))]
        elif variable == 'z':
            var = [self.paths[i].Pos_abs[2] for i in range(len(self.paths))]
        elif variable == 'W':
            var = [self.paths[i].W for i in range(len(self.paths))]
            
        hist = ax.hist(variable, bins = np.arange(100)*100, histtype = 'step')

        ax.set_ylabel('Count')
        ax.set_xlabel('Distance to camera (m)')
        ax.spines[['left','right', 'top', 'bottom']].set_visible(False)
        
    def threeD_scat(self, fig):
        
        ax = fig.add_subplot(projection='3d')
        
        markers = ['o','+']
        boolis = [True, False]
        
        for j in range(2):
            
            x = self.get_coord('x', boolis[j])
            y = self.get_coord('y', boolis[j])
            z = self.get_coord('z', boolis[j])
            
            W = [self.paths[i].W for i in range(len(self.paths)) \
                if self.paths[i].Abs_surf == boolis[j]]

            cax = ax.scatter(x, y, z, c = W, marker = markers[j], cmap = 'viridis', vmin = 0, vmax = 30)
            
        fig.colorbar(cax, label = 'Relative importance in raddiance')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')
        ax.grid(False)
        
    def twoD_scat(self, ax, fig,save_title):
        
        markers = ['o','+']
        boolis = [True, False]
        
        for j in range(2):
        
            x = self.get_coord('x', boolis[j])
            y = self.get_coord('y', boolis[j])

            W = [self.paths[i].W for i in range(len(self.paths)) \
                if self.paths[i].Abs_surf == boolis[j]]
                        
            cax = ax.scatter(x, y, marker = markers[j], c = W, cmap = 'viridis')
            
        fig.colorbar(cax, label = 'Radiance ($W/m^2$)')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.spines[['left','right', 'top', 'bottom']].set_visible(False) 
        
        plt.savefig(save_title)
        
    def dist_weight_2(self, ax, Surface, color, name):
        
        ax2 = ax.twinx()
        
        Dist = [self.paths[i].Dist for i in range(len(self.paths)) \
            if self.paths[i].Abs_surf == Surface and self.paths[i].Space == False]

        W = [self.paths[i].W for i in range(len(self.paths)) \
            if self.paths[i].Abs_surf == Surface and self.paths[i].Space == False]

        cax = ax.plot(Dist, W, c = color, markersize = 1, marker = 'o', label = "Chemin "+name,linestyle='')

        paired = list(zip(Dist, W))
        paired_sorted = sorted(paired, key=lambda x: x[0])
        Dist_sorted, W = zip(*paired_sorted)    
        
        ax2.plot(Dist_sorted,np.cumsum(W)/self.W_tot, c = color, linestyle = 'solid',\
                label = "Cumulative relative weight " + name)
        
        Nbre_photon_abs = self.Nbre_photon - self.Nbre_space
        ax2.plot(Dist_sorted, np.cumsum(np.ones((1,len(W))))/Nbre_photon_abs,c = color, \
                   label = 'Cumulative count ' + name, linestyle = 'dashed')
        
        ax2.spines[['left','right', 'top', 'bottom']].set_visible(False)
        
        ax.set_ylabel('Weight ($W/m^2$)')
        ax.set_xlabel('Distance of absorption (m)')
        ax.spines[['left','right', 'top', 'bottom']].set_visible(False)
        
        ax2.set_ylim((0,1))
        
        ax2.legend()
        
    def dist_weight(self, axs, save_title):
        
        ax2 = axs[0].twinx()
        ax3 = axs[1].twinx()
        
        markers = ['o','+']
        boolis = [True, False]
        colors = ['green','blue']
        labels = ['Surface', 'Atm']
                
        for j in range(2):
            
            if len([self.paths[i].Dist for i in range(len(self.paths)) \
                if self.paths[i].Abs_surf == boolis[j]]) == 0 :
                
                j = 1
        
            Dist = [self.paths[i].Dist for i in range(len(self.paths)) \
                if self.paths[i].Abs_surf == boolis[j]]

            W = [self.paths[i].W for i in range(len(self.paths)) \
                if self.paths[i].Abs_surf == boolis[j]]
            
            Azi = [self.paths[i].Azimuth for i in range(len(self.paths)) \
                if self.paths[i].Abs_surf == boolis[j]]
            
                        
            cax = axs[0].scatter(Dist, W, marker = markers[j], c = 'k')
            
            paired = list(zip(Dist, W))
            paired_sorted = sorted(paired, key=lambda x: x[0])
            Dist_sorted, W = zip(*paired_sorted)
            
            
            ax2.plot(Dist_sorted,np.cumsum(W)/np.cumsum(W)[-1], c = colors[j], \
                        label = labels[j])
            
            if j == 0 :
                cax = axs[1].scatter(Dist, W, marker = markers[j], c = 'k')

                ax3.plot(Dist_sorted,np.cumsum(W)/np.cumsum(W)[-1], c = colors[j], \
                        label = labels[j])
                            
        W = [self.paths[i].W/self.W_tot for i in range(len(self.paths))]
        Dist = [self.paths[i].Dist for i in range(len(self.paths))]
        
        paired = list(zip(Dist, W))
        paired_sorted = sorted(paired, key=lambda x: x[0])
        Dist_sorted, W = zip(*paired_sorted)

        ax2.plot(Dist_sorted,np.cumsum(W), c = 'r', \
                    label = 'Cumulative weight total')

        ax2.spines[['left','right', 'top', 'bottom']].set_visible(False)
        ax3.spines[['left','right', 'top', 'bottom']].set_visible(False)

        ax2.legend() 
        
        for i in range(2):
            ax = axs[i]
            ax.set_xlabel('Distance from camera (m)')
            ax.set_ylabel('Weight')
            ax.spines[['left','right', 'top', 'bottom']].set_visible(False) 
        
        plt.savefig(save_title)        
                 
    def plotting_pcolor_cumul(self, ax, Mat_cumul,zs,rs, norm, cmap):
    
        cax = ax.pcolormesh(Mat_cumul, cmap = cmap, norm = norm)
    
    def figure_utile(self,type_name):

        fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (12,6), layout = 'constrained')

        save_title = '/home/barroisl/edstar/Simus/Output/Dist_W_15_15_10_'+type_name+'.png'

        set_c.dist_weight(axs, save_title)

        plt.close()

        fig, axs = plt.subplots(nrows = 1, ncols = 2,figsize = (12,6), layout = 'constrained')

        save_title = '/home/barroisl/edstar/Simus/Output/W_15_15_10_'+type_name+'.png'

        set_c.plot_atm_radiance_profile(axs,'W',save_title)

        plt.close()

        fig,ax = plt.subplots(figsize = (6,6), layout = 'constrained')

        save_title = '/home/barroisl/edstar/Simus/Output/W_xy_15_15_10_'+type_name+'.png'

        set_c.twoD_scat(ax, fig, save_title)

        plt.close()
    
    def plotting_rose(self,ax,grid_fact,surface,cmap,vmin,vmax,add_colorbar):
        
        # Structure of the diagramme
        
        compass_angles = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315])  # N, NE, E, SE, S, SW, W, NW
        compass_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        ax.set_xticks(compass_angles)         
        ax.set_xticklabels(compass_labels)    

        ax.set_theta_zero_location('N')       
        ax.set_theta_direction(-1)            
        ax.set_rticks([5])                     

        for R in range(1,5): 
            plotting_circle_labeled(R*grid_fact,ax)
            
        
        #Collision points
        
        bounds = np.arange(vmin, vmax, (vmax-vmin)*0.1)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        
        path_considered = [self.paths[i] for i in range(len(self.paths)) if \
                           self.paths[i].Abs_surf == surface and self.paths[i].Space == False]
        
        Dist = [path_considered[i].Pos_abs[2]-self.pos_camera[2] for i in range(len(path_considered))]
        W = [path_considered[i].W for i in range(len(path_considered))]
        Azi = [path_considered[i].Azimuth for i in range(len(path_considered))]
        
        cax = ax.scatter(Azi,Dist,c = W,cmap = cmap, norm = norm)
        
        if add_colorbar == True :
            
            cbar = plt.colorbar(cax, ax=ax,location='right', orientation='vertical', cmap = cmap, norm = norm)
            cbar.set_label('Weight $W.m^{-2}$')
            
    def plotting_collision_map(self,ax,surface,cmap,vmin,vmax,add_colorbar):
        
        #Collision points
        
        bounds = np.arange(vmin, vmax, (vmax-vmin)*0.1)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        
        path_considered = [self.paths[i] for i in range(len(self.paths)) if \
                           self.paths[i].Abs_surf == surface and self.paths[i].Space == False]
        
        x = [path_considered[i].Pos_abs[0] for i in range(len(path_considered))]
        y = [path_considered[i].Pos_abs[1] for i in range(len(path_considered))]
        W = [path_considered[i].W for i in range(len(path_considered))]
        
        cax = ax.scatter(x,y,c = W,s=1,cmap = cmap, norm = norm)
        
        if add_colorbar == True :
            
            cbar = plt.colorbar(cax, ax=ax,location='right', orientation='vertical', extend = 'max',cmap = cmap, norm = norm)
            cbar.set_label('Thermal flux $W.m^{-2}$')
            
    def plotting_collision_map_2(self,ax,color):    
        
        path_considered = [self.paths[i] for i in range(len(self.paths)) if \
                           self.paths[i].Abs_surf == True and self.paths[i].Space == False]
        
        x = [path_considered[i].Pos_abs[0] for i in range(len(path_considered))]
        y = [path_considered[i].Pos_abs[1] for i in range(len(path_considered))]
        W = [path_considered[i].W for i in range(len(path_considered))]
        
        cax = ax.scatter(x,y,c = color,s=1)
        
    def tab_lambda_z_flux(self,z,LW_int_edges):
    
        res = np.zeros((z.size-1,LW_int_edges.shape[0]-1)) # on prend en compte aussi les rayon arrivant en dessous de la caméra
        dz = z[1]-z[0]
    
        pathes = [self.paths[i] for i in range(len(self.paths)) if \
                      self.paths[i].Abs_atm == True]
                      
        for path in pathes :
            index_z = int(path.Pos_abs[2]//dz)
            if index_z >= (z.size-1) :
                index_z = int(len(res)-1)
            
            mask = (10000/(LW_int_edges[:, 1]) <= path.wlen/1000) & \
                (path.wlen/1000 <= 10000/(LW_int_edges[:, 0]))
            index_lambda = np.where(mask)[0]

            res[index_z,index_lambda] += path.W/self.Nbre_photon
                    
        return res
        
def plotting_flux_profile_all_cams(
            exp :str,
            res : str,
            n_cam : str,
            lr : str,
            force_save : bool = False,
            save_name : str = ' ',
            z : np.ndarray = np.arange(0,10e3,100),
            step :int = 1000):

    dz = z[1] - z[0]
    n_bins = z.size -1

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), layout='constrained')

    im = None 

    for j,atm in enumerate(["SMLSATM","SMLWATM"]):

        dire = f"/home/barroisl/Transect_MC_auto/Data/{exp}_{res}_{n_cam}_atms_{lr}_emis/{atm}/"
        print(dire)
        list_ = os.listdir(dire)

        res_ = np.zeros((z.size-1, len(list_)))
        alt_cams = np.zeros(len(list_))

        for d,dossier in tqdm(enumerate(list_)):
            dat = np.loadtxt(dire +f"/{dossier}/{dossier}_50_15_15.txt")
            cam = np.loadtxt(dire +f"/{dossier}/{dossier}_camera_tgt.txt")

            mc_set = MC_Set(dat,cam[:3])
            alt_cams[d] = cam[2]
            res_[:,d] = mc_set.energy_exchange(z)

        im = axs[j].pcolormesh(
                res_,
                norm=LogNorm(vmin=1e-7, vmax=1e2),
                cmap='binary',
                shading='auto')
        
        axs[j].scatter(np.arange(alt_cams.size), alt_cams/dz, color = 'dodgerblue', marker = "*")

        # --- Y ticks every 500 m ---
        if j == 0 :
            step_bins = int(step / dz)  # number of bins per 500 m
            ytick_positions = np.arange(0, n_bins + 1, step_bins)
            ytick_labels = [f"{int(z[i])} m" for i in ytick_positions]

            axs[j].set_yticks(ytick_positions)
            axs[j].set_yticklabels(ytick_labels, fontsize=12)
        else :
            axs[j].set_yticks([])
            axs[j].set_ylabel('')

        # --- X ticks: spectral intervals indexed 1 to 16 ---
        axs[j].set_xticks([])
        axs[j].set_xlabel(" ")

        axs[j].set_title(atm)
        axs[j].spines[['left','right', 'top', 'bottom']].set_visible(False)

    # --- Shared colorbar on the right, for the whole figure ---
    fig.colorbar(im, ax=axs, location='right', label='Flux ($W.m^{-2}$)')

    if force_save :
        plt.savefig(f"/home/barroisl/Transect_MC_auto/Output/{save_name}.png")

    plt.show()
    


def plotting_spectral_energy_exchange(z : np.ndarray,
                                      dossier :str,
                                     step : int,
                                     force_save : bool = False,
                                     save_name : str =' '):
    
    dz = z[1] - z[0]
    n_bins = z.size - 1

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), layout='constrained')

    im = None  # will hold the last mappable for the shared colorbar

    for j, atm in enumerate(atms[1:3]):
        if atm != "EMPTATM" and atm != "TUXUATM":
            dat = np.loadtxt(f"/home/barroisl/Transect_MC_auto/Data/lavey_30_100_atms_00_emis/{atm}/{dossier}/{dossier}_50_15_15.txt")
            cam = np.loadtxt(f"/home/barroisl/Transect_MC_auto/Data/lavey_30_100_atms_00_emis/{atm}/{dossier}/{dossier}_camera_tgt.txt")
            mc_set = MC_Set(dat, cam[:3])
            
            axs[j].hlines(cam[2]/dz,0,15,linestyle = "dashed", color = 'dodgerblue')

            prop_atm = Prop_atm(atm)
            tab = mc_set.tab_lambda_z_flux(z=z, LW_int_edges=prop_atm.LW_int_edges)

            im = axs[j].pcolormesh(
                tab,
                norm=LogNorm(vmin=1e-7, vmax=1e2),
                cmap='binary',
                shading='auto')

            # --- Y ticks every 500 m ---
            if j == 0 :
                step_bins = int(step / dz)  # number of bins per 500 m
                ytick_positions = np.arange(0, n_bins + 1, step_bins)
                ytick_labels = [f"{int(z[i])} m" for i in ytick_positions]

                axs[j].set_yticks(ytick_positions)
                axs[j].set_yticklabels(ytick_labels, fontsize=12)
            else :
                axs[j].set_yticks([])
                axs[j].set_ylabel('')

            # --- X ticks: spectral intervals indexed 1 to 16 ---
            n_spectral = tab.shape[1]
            xtick_positions = np.arange(n_spectral) + 0.5  # center of each column
            xtick_labels = [str(k) for k in range(1, n_spectral + 1)]

            axs[j].set_xticks(xtick_positions)
            axs[j].set_xticklabels(xtick_labels, fontsize=12)

            axs[j].set_title(atm)
            axs[j].spines[['left','right', 'top', 'bottom']].set_visible(False)

    # --- Shared colorbar on the right, for the whole figure ---
    fig.colorbar(im, ax=axs, location='right', label='Flux ($W.m^{-2}$)')
    
    if force_save :
        plt.savefig(f"/home/barroisl/Transect_MC_auto/Output/{save_name}.png")

    plt.show()
    
def plot_lambda_filter(
    Planck : bool,
        force_save : bool,
        save_name : str):
    
    path_dir = "/home/barroisl/Transect_MC_auto/Data/guiers_250_144_atms/"
    
    fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize = (12,6), layout = 'constrained')
    
    for i,atm in enumerate(atms):
        
        bins = np.linspace(4,40,240)
        
        if atm != "TUXUATM" : #and atm != "EMPTATM":
            
            data = np.loadtxt(path_dir + atm +'/11/11_50_15_15.txt')
            pos_camera = np.loadtxt(path_dir + atm +'/11/11_camera_tgt.txt')[:3]
            mc_set = MC_Set(data, pos_camera)
            
            if atm == "EMPTATM" : 
                lambdas = np.array(mc_set.distrib_lambda(False))*1e-3
                axs[0].hist(lambdas, bins = bins, histtype = 'step', color = 'k', \
                            density = True, label = 'Initial spectrum')
            
            lambdas_surf = np.array(mc_set.distrib_lambda(True))*1e-3
            axs[1].hist(lambdas_surf, bins = bins, histtype = 'step',linestyle = 'solid', \
                    color = dict_atm[atm][1],label = dict_atm[atm][0], density = False, alpha = 0.6)
    
    ### Planck
    if Planck :
        wavelength = np.linspace(4e-6,40e-6,2000)
        T = 275

        Planck = Planck_law(
            wavelength = wavelength, 
            temperature = T)
        axs[0].plot(wavelength*1e6,0.065*Planck/max(Planck),color = 'firebrick', label = "Planck distribution")
    
    for ax in axs :
        ax.set_yticks([])
        ax.set_xlabel('$\lambda$ [$\mu m$]')
        ax.spines[['left','right', 'top', 'bottom']].set_visible(False)
        ax.legend()
        
    axs[0].set_ylabel('Density')
    axs[1].set_ylabel('Count')
        
    if force_save :
    
        plt.savefig(f"/home/barroisl/Transect_MC_auto/Output/guiers_250_144_atms/{save_name}.jpg")
        
def plotting_spectral_dist(z : np.ndarray,
                         dossier :str,
                         force_save : bool = False,
                         save_name : str =' '):
    
    dz = z[1] - z[0]
    n_bins = z.size - 1
    
    bins = np.linspace(4,40,240)
    indices_h = [0,1,2,3]
    colors_ = ["lightblue","steelblue","dodgerblue","darkblue"]

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(18, 6), layout='constrained')

    im = None  # will hold the last mappable for the shared colorbar

    for j, atm in enumerate(atms[1:3]):
        if atm != "EMPTATM" and atm != "TUXUATM":
            dat = np.loadtxt(f"/home/barroisl/Transect_MC_auto/Data/lavey_30_100_atms_00_emis/{atm}/{dossier}/{dossier}_50_15_15.txt")
            cam = np.loadtxt(f"/home/barroisl/Transect_MC_auto/Data/lavey_30_100_atms_00_emis/{atm}/{dossier}/{dossier}_camera_tgt.txt")
            mc_set = MC_Set(dat, cam[:3])

            lambda_dist_lists = mc_set.lambda_exchange(z)
            for k,ind in enumerate(indices_h) :
                axs[j].hist(lambda_dist_lists[ind], bins = bins, histtype = 'step',linestyle = 'solid', \
                                    color = colors_[k],label = f"{z[k]} m", density = False, linewidth = 1)
                
            axs[j].set_title(atm)
            axs[j].spines[['left','right', 'top', 'bottom']].set_visible(False)
            
            
            data = np.loadtxt("/home/barroisl/Transect_MC_auto/Data/guiers_250_144_atms/EMPTATM/11/11_50_15_15.txt")
            pos_camera = np.loadtxt("/home/barroisl/Transect_MC_auto/Data/guiers_250_144_atms/EMPTATM/11/11_camera_tgt.txt")[:3]
            mc_set = MC_Set(data, pos_camera)

            lambdas = np.array(mc_set.distrib_lambda(False))*1e-3
            axs[j].hist(lambdas, bins = bins, histtype = 'step', color = 'k', \
                        density = False, label = 'Initial spectrum')
            
            axs[j].set_ylabel('Count')
            axs[j].set_yticks([])
            axs[j].set_xlabel("$\lambda$ [$\mu m$]")
            axs[j].legend()
    
    if force_save :
        plt.savefig(f"/home/barroisl/Transect_MC_auto/Output/{save_name}.png")

    plt.show()
        
######################### Class Porp_atm #########################
        
def extract_marker_positions(
    lignes : list[str],
    marker_list : list[str],
    offset = 0,
    )->dict:
    
    index = {}
    
    for i, ligne in enumerate(lignes):
        for marker in marker_list:
            stripped = ligne.strip()
            if marker in stripped and 'x-point' not in stripped:
                index[marker] = i + offset
            
    return index
            
def extract_between_markers(
    file_path: str,
    start_marker: str,
    end_marker: str,
    include_markers: bool = False,
    ) -> list[float]:
    
    extracted = []
    inside = False

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n")

            # Détection du marqueur de début
            if not inside and start_marker in stripped:
                inside = True
                if include_markers:
                    extracted.append(stripped)
                continue  # on ne garde pas la ligne de début si include_markers=False

            # Détection du marqueur de fin
            if inside and end_marker in stripped:
                if include_markers:
                    extracted.append(stripped)
                break  # on arrête la lecture dès le marqueur de fin

            # Collecte des lignes situées entre les deux marqueurs
            if inside:
                extracted.append(float(stripped))

    return extracted

def contains_string(
        lignes : list[str],
        test : str)->bool:  #, encoding="utf-8"):

    for ligne in lignes :
        if test in ligne :
            return True
    
    return False 

"""
dict_atm = {"SMLSATM" : ["Mid latitude summer", "red", "Reds", cm.Reds, ],
            "SMLWATM" : ["Mid latitude winter", "green", "Greens", cm.Greens],
            "SPOSATM" : ["Polar summer", "orange","Oranges",cm.Oranges],
            "SPOWATM" : ["Polar winter", "blue","Blues",cm.Blues],
            "STROATM" : ["Tropics", "purple","Purples",cm.Purples],
            "EMPTATM" : ["Transparent", "black","Greys",cm.Greys],
            "TUXUATM" : ["Uniforme", "yellow","YlGn", cm.YlGn]}
"""

dict_atm = {"EMPTATM" : ["Transparent", "black","Greys",cm.Greys],
            "SMLSATM" : ["Mid latitude summer", "red", "Reds", cm.Reds],
            "SMLWATM" : ["Mid latitude winter", "green", "Greens", cm.Greens],
            "TUXUATM" : ["Uniforme", "blue","Blues", cm.Blues]}

atms = list(dict_atm.keys())

class Prop_atm:
    
    """
    Cadre pour faciliter l'étude d'un profile atm au format ecRAD
    """
    
    def __init__(self,name):
        
        directory = "/home/barroisl/edstar/Simus/atms/"
        file_path = directory + "ecrad_opt_prop_"+name+".txt"
        marker_list = ["Number of levels", "Number of layers", "Ground temperature",
                       "Pressure","Temperature", "Altitude", "Nominal x(H2O)",
                      "Number of water vapor concentration values","LW emissivity",
                      "SW emissivity","Number of spectral intervals (LW)"]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lignes = [ligne.rstrip('\n') for ligne in f]
        
        index = extract_marker_positions(
                    lignes = lignes,
                    marker_list = marker_list)
        
        # Name 
        self.name = name
            
        # Index dictionnary
        self.d_index = index
            
        # Lignes of file
        self.lignes = lignes
        
        # Number of levels
        self.N_levels = int(lignes[index["Number of levels"]+1])
        
        # Number of layers
        self.N_layers = int(lignes[index["Number of layers"]+1])
        
        # Groud temperature
        self.Ts = float(lignes[index["Ground temperature"]+1])
        
        # Number of water vapor values
        self.N_wv = int(lignes[index["Number of water vapor concentration values"]+1])
        
        # LW emissivity
        self.LW_eps = float(lignes[index["LW emissivity"]+1])
        
        # SW emissivity
        self.SW_eps = float(lignes[index["SW emissivity"]+1])
        
        # Number of spectral intervals (LW)
        self.N_LW_int = int(lignes[index["Number of spectral intervals (LW)"]+1])
        
        if self.name != "EMPTATM":
            # Intervals of wave lengths
            marker_list_LW_int = []
            LW_int_edges = np.zeros((self.N_LW_int,2))
            for i in range(self.N_LW_int):
                if i+1 > 9 :
                    spaces = " "
                else :
                    spaces = "  "
                
                marker_list_LW_int.append(f"LW interval index:{spaces}{i+1}")
                
            index_LW_int_l = extract_marker_positions(
                        lignes = lignes,
                        marker_list = marker_list_LW_int
                        )
            index_LW_int_h = extract_marker_positions(
                        lignes = lignes,
                        marker_list = marker_list_LW_int
                        )
            
            for i in range(len(index_LW_int_h)):
                # Le wave number k et la longueur d'onde sont inversement proportionnels
                # Donc on inverse les deux indices de colonne de LW_int_edges
                #LW_int_edges[i,1] = (2*np.pi*1e7)/float(lignes[index_LW_int_l[marker_list_LW_int[i]]+2])
                #LW_int_edges[i,0] = (2*np.pi*1e7)/float(lignes[index_LW_int_h[marker_list_LW_int[i]]+4])
                LW_int_edges[i,0] = float(lignes[index_LW_int_l[marker_list_LW_int[i]]+2])
                LW_int_edges[i,1] = float(lignes[index_LW_int_l[marker_list_LW_int[i]]+4])
                
            self.LW_int_edges = LW_int_edges
            
        else :
            self.LW_int_edges = np.nan
        
        if self.name != "EMPTATM" and self.name != "TUXUATM":
            # Nominal profiles of absorption
            marker_list_LW_int = []
            for i in range(self.N_LW_int):
                if i+1 > 9 :
                    spaces = "  "
                else :
                    spaces = "   "

                marker_list_LW_int.append(f"Nominal absorption coefficient [m^-1] for LW interval:{spaces}{i+1} ; g-point:   1")

            index_LW_int = extract_marker_positions(
                        lignes = lignes,
                        marker_list = marker_list_LW_int)

            Abs_profiles = np.zeros((self.N_layers,self.N_LW_int))

            for i,LW_int in enumerate(marker_list_LW_int):
                Abs_profiles[:,i] = lignes[index_LW_int[LW_int]+1:index_LW_int[LW_int]+31]

            self.abs_profiles = Abs_profiles
        else :
            self.abs_profiles = np.nan
        
    def g_for_LW_int(self):
        
        d_g = {}
        
        spaces_g = "   "
        spaces_x = "  "
        spaces_LW = "   "
        for LW_int in tqdm(range(self.N_LW_int)):
            g_point = 1
            spaces_g = "   "
            if LW_int > 9 :
                spaces_LW = "  "
            string_test = "Water absorption coefficient [m^-1] for LW interval:"+spaces_LW+\
                       f"{LW_int+1} ; g-point:"+spaces_g+f"{g_point} ; x-point:"+spaces_x+f"{1}"
            while contains_string(
                lignes = self.lignes,
                test = string_test) == True :
                g_point +=1
                if g_point > 9 :
                    spaces_g = "  "
                string_test = "Water absorption coefficient [m^-1] for LW interval:"+spaces_LW+\
                       f"{LW_int+1} ; g-point:"+spaces_g+f"{g_point} ; x-point:"+spaces_x+f"{1}"

            d_g[LW_int+1] = g_point-1
            
        return d_g
            
    def profiles(self,
                variable : str)->np.ndarray:
        
        if variable == "T":
            starting_ligne = self.d_index["Temperature"]+1
            
        elif variable == "P":
            starting_ligne = self.d_index["Pressure"]+1
            
        elif variable == "x":
            starting_ligne = self.d_index["Nominal x(H2O)"]+1
            
        elif variable == "z":
            starting_ligne = self.d_index["Altitude"]+1
            
        return np.array(self.lignes[starting_ligne:starting_ligne+self.N_layers],\
                        dtype = np.float32)
        
    def profile_w_abs_coef(self,
                LW_interval : int,
                g_point : int,
                x_point : int)->np.ndarray:
        
        spaces_LW = "   "
        spaces_g = "   "
        spaces_x = "  "
        if LW_interval > 9 :
            spaces_LW = spaces_LW[:-1]
        if g_point > 9 :
            spaces_g = spaces_g[:-1]
        if x_point > 9 :
            spaces_x = spaces_x[:-1]
            
        marker_list = ["Water absorption coefficient [m^-1] for LW interval:"+spaces_LW+\
                       f"{LW_interval} ; g-point:"+spaces_g+f"{g_point} ; x-point:"+spaces_x+f"{x_point}"]

        index = extract_marker_positions(
                    lignes = self.lignes,
                    marker_list = marker_list)
        
        return np.array(self.lignes[index[marker_list[0]]+1:index[marker_list[0]]+1+self.N_layers],\
                        dtype = np.float32)
                        
    def w_g(self,LW_int_number : int)-> np.ndarray:
        spaces_LW = "  "
        if LW_int_number > 9 :
            spaces_LW = " "
        marker_list = [f'LW interval index:{spaces_LW}{LW_int_number}']

        index = extract_marker_positions(
                    lignes = self.lignes,
                    marker_list = marker_list)
                    
        N_quadrature_weights = int(self.lignes[index[marker_list[0]]+6])
                    
        return np.array(self.lignes[index[marker_list[0]]+8:index[marker_list[0]]+8+N_quadrature_weights],\
                        dtype = np.float32)
                        
    def Number_g_point(self,LW_int_number : int) -> int:
        
        spaces_LW = "  "
        if LW_int_number > 9 :
            spaces_LW = " "
        marker_list = [f'LW interval index:{spaces_LW}{LW_int_number}']

        index = extract_marker_positions(
                    lignes = self.lignes,
                    marker_list = marker_list)
                    
        N_quadrature_weights = int(self.lignes[index[marker_list[0]]+6])
        
        return N_quadrature_weights
        
        
    def k_g_nb(self,LW_int_number : int,g_point : int)-> np.ndarray:
                   
        # Extracting the k_eff profile for the given quadrature point and spectral interval
        spaces_LW = "   "
        spaces_g = "   "
        if LW_int_number > 9 :
            spaces_LW = spaces_LW[:-1]
        if g_point > 9 :
            spaces_g = spaces_g[:-1]
                
        marker_list = ["Nominal absorption coefficient [m^-1] for LW interval:"+spaces_LW+\
                           f"{LW_int_number} ; g-point:"+spaces_g+f"{g_point}"]
                           
        index = extract_marker_positions(
                    lignes = self.lignes,
                    marker_list = marker_list)

        return np.array(self.lignes[index[marker_list[0]]+1:index[marker_list[0]]+31],\
                        dtype = np.float32) 
    
            
    ######################## Dictionnary #################################
    
    def d_ka_LW(self):
        
        d_ka_LW = {}
        d_g = self.g_for_LW_int()
        
        for LW_int in tqdm(range(self.N_LW_int)):
            for x_point in range(self.N_wv):
                for g_point in range(d_g[LW_int+1]):
                    d_ka_LW[f"{int(LW_int)+1}_{int(g_point)+1}_{int(x_point)+1}"] = self.profile_w_abs_coef(
                                                                    LW_interval = LW_int+1,
                                                                    g_point = g_point+1,
                                                                    x_point = x_point+1)
        return d_ka_LW
                      
    ######################## Plotting ####################################
    
    def plotting_LW_abs_coeff_profile(self, 
                ax,
                LW_interval : int,
                g_point : int,
                x_point : int,
                y_lim):
        
        x = self.profile_w_abs_coef(
                LW_interval = LW_interval,
                g_point = g_point,
                x_point = x_point)
        y = self.profiles(variable = "z")
        
        ax.plot(x, y, color = dict_atm[self.name][1])
                
        label = f" - LW = {LW_interval};g = {g_point}; w = {x_point}"
        ax.set_title(dict_atm[self.name][0]+label)
        c
        ax.set_ylabel('Altitude (m)')
        ax.set_xlabel('$k_a$ $(m^{-1})$')
        ax.set_ylim(y_lim)
        
    def plotting_LW_abs_coeff_profile_interact(self, 
                LW_interval : int,
                g_point : int,
                x_point : int):
        
        plt.close()
        
        fig, ax = plt.subplots(figsize = (5,5), layout = 'constrained')

        x = d.item().get(f"{int(LW_interval)}_{int(g_point)}_{int(x_point)}")
        y = self.profiles(variable = "z")
        
        y_lim = (None,None)
        x_lim = (0,1e-5)
        
        ax.plot(x, y, color = dict_atm[self.name][1])
                
        label = f" - LW = {int(LW_interval)};g = {int(g_point)}; x = {int(x_point)}"
        ax.set_title(dict_atm[self.name][0]+label)
        ax.spines[['left','right', 'top', 'bottom']].set_visible(False)
        ax.set_ylabel('Altitude (m)')
        ax.set_xlabel('$k_a$ $(m^{-1})$')
        ax.set_ylim(y_lim)
        #ax.set_xlim(x_lim)
        
        plt.show()
        
    def tab_k_z_nb(self)->np.ndarray:

        tab_lambda_z = np.zeros((self.N_layers,self.N_LW_int))

        for nb in range(1,self.N_LW_int+1):
            # poids de chaque g-point par bande
            weights = self.w_g(nb)
            for g in range(1,self.Number_g_point(nb)+1):
                # profile de k_a en fonction de l'altitude pour un g et un nb
                nominal_k = self.k_g_nb(LW_int_number =nb ,g_point =g) 
                # profile de k_eff somme pondérée par le w_g
                tab_lambda_z[:,nb-1] += weights[g-1]*nominal_k

        return tab_lambda_z

    def plotting_2D_ka(self,axs,ax,colorbar : bool = False, up_label : bool = False, force_save  : bool= False, save_name : str = ' ',
                       x_lab : bool = False, y_lab : bool = False):

        alt_km = self.profiles(variable = "z")*1e-3
        bands = self.LW_int_edges
        arr_nb_z = self.tab_k_z_nb()

        im = ax.pcolormesh(
            np.arange(1, arr_nb_z.shape[1] + 1),  
            alt_km,                                  
            arr_nb_z,                             
            norm=LogNorm(vmin=1e-7, vmax=1e0),
            cmap='binary',
            shading='auto'
        )
        if colorbar :
            cbar = plt.colorbar(im, ax=axs)
            cbar.set_label('$k_a$ [m⁻¹]')

        if x_lab :
            ax.set_xlabel('Bande spectrale LW')
            ax.set_xticks(range(1, arr_nb_z.shape[1] + 1))
        else :
            ax.set_xticks([])
        if y_lab :
            ax.set_ylabel('Altitude [km]')
        else :
            ax.set_yticks([])

        if up_label :
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(range(1, arr_nb_z.shape[1] + 1))
            ax2.set_xticklabels([f"{b[0]}-{b[1]} µm" for b in bands], fontsize=15, rotation=45, ha='left')
            #ax2.set_xlabel('Intervalle [cm⁻¹]')

        if force_save :
            plt.savefig(save_name)
    """        
    def plot_ka_lambda(self, alt : float, force_save = False : bool, save_name = ' ' : str):
            
        alt_km = self.profiles(variable = "z")*1e-3
        bands = self.LW_int_edges
        arr_nb_z = self.tab_k_z_nb()
        
        fig, ax = plt.subplots(figsize=(10, 6), layout = 'constrained')
        
        lams = 10000 / ((bands[:,0] + b[:,1]) / 2)
    """
        
        

    def plot_ka_profile(self,force_save : bool = False, save_name : str = ' '):

        alt_km = self.profiles(variable = "z")*1e-3
        bands = self.LW_int_edges
        arr_nb_z = self.tab_k_z_nb()

        fig, ax = plt.subplots(figsize=(10, 6), layout = 'constrained')

        n_z,nb = arr_nb_z.shape

        colors = cm.rainbow(np.linspace(0, 1, nb))

        for i, col in enumerate(colors):
            b = bands[i - 1]
            k = arr_nb_z[:,i - 1]  # shape (n_layers,)
            lam = 10000 / ((b[0] + b[1]) / 2)
            ax.plot(np.maximum(k, 1e-12), alt_km, color=col, lw=2,
                    label=f"LW{i+1} ({b[0]}-{b[1]} cm⁻¹, ~{lam:.1f}µm)")

        ax.set_xlabel('$k_a$ [m⁻¹]')
        ax.set_ylabel('Altitude [km]')
        ax.set_xscale('log')  # log on x axis only
        ax.set_title(dict_atm[self.name][0])
        ax.set_ylim(0, 50)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines[['left','right', 'top', 'bottom']].set_visible(False)

        if force_save :
            plt.savefig(save_name)
        
        
############################ Plotting ################################## 

params = {'legend.fontsize': 'x-large',
          #'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

######################### Planck & luminence #########################################

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
    
def from_position_to_luminence_symbolique(
    abs_nd : np.ndarray,
    ds : xr.Dataset)-> np.ndarray | float :
    
    """
    [Inputs]
        - abs_nd : np.nadarray des positions des absorptions par la surface
        - ds avec une variable 'ts' de température de surface
    [Outputs]
        - sigma T^4 pour une simulation symbolique
    """
    
    sigma = 5.67e-8
    
    if abs_nd.ndim == 1:
        ts_interp = interpolate(ds,np.vstack((abs_nd[:2],abs_nd[:2])),var_name = 'ts')
        ts_interp = ts_interp[0]
        
    elif abs_nd.ndim == 2:
        ts_interp = interpolate(ds,abs_nd[:,:2],var_name = 'ts')
    
    return sigma*ts_interp**4
    
    
    
########################### Atm filter ################################# 

def plot_lambda_filter(
    Planck : bool,
        force_save : bool,
        save_name : str):
    
    path_dir = "/home/barroisl/Transect_MC_auto/Data/guiers_250_144_atms/"
    
    fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize = (12,6), layout = 'constrained')
    
    for i,atm in enumerate(atms):
        
        bins = np.linspace(4,60,240)
        
        if atm != "TUXUATM" : #and atm != "EMPTATM":
            
            data = np.loadtxt(path_dir + atm +'/11/11_50_15_15.txt')
            pos_camera = np.loadtxt(path_dir + atm +'/11/11_camera_tgt.txt')[:3]
            mc_set = MC_Set(data, pos_camera)
            
            if atm == "EMPTATM" : 
                lambdas = np.array(mc_set.distrib_lambda(False))*1e-3
                axs[0].hist(lambdas, bins = bins, histtype = 'step', color = 'k', \
                            density = True, label = 'Initial spectrum')
            
            lambdas_surf = np.array(mc_set.distrib_lambda(True))*1e-3
            axs[1].hist(lambdas_surf, bins = bins, histtype = 'step',linestyle = 'solid', \
                    color = dict_atm[atm][1],label = dict_atm[atm][0], density = False)
    
    ### Planck
    if Planck :
        wavelength = np.linspace(4e-6,60e-6,2000)
        T = 275

        Planck = Planck_law(
            wavelength = wavelength, 
            temperature = T)
        axs[0].plot(wavelength*1e6,0.065*Planck/max(Planck),color = 'firebrick', label = "Planck distribution")
    
    for ax in axs :
        ax.set_yticks([])
        ax.set_ylabel('Density')
        ax.set_xlabel('$\lambda$ $\mu m$')
        ax.spines[['left','right', 'top', 'bottom']].set_visible(False)
        ax.legend()
        
    if force_save :
    
        plt.savefig(f"/home/barroisl/Transect_MC_auto/Output/guiers_250_144_atms/{save_name}.jpg")
    
"""
plot_lambda_filter(Planck = True,
                    force_save = False,
                    save_name = "lambda_filter_guiers_11")
"""

############################### Cumulative absorption profile #################################

def plot_atm_density_abs_guiers(
        force_save : bool,
        save_name : str):
    
    path_dir = "/home/barroisl/Transect_MC_auto/Data/guiers_250_144_atms/"
    
    fig, ax = plt.subplots(figsize = (12,4), layout = 'constrained')
    
    for i,atm in enumerate(atms):
        
        if atm != "TUXUATM" : #and atm != "EMPTATM":
            
            data = np.loadtxt(path_dir + atm +'/11/11_50_15_15.txt')
            pos_camera = np.loadtxt(path_dir + atm +'/11/11_camera_tgt.txt')[:3]
            mc_set = MC_Set(data, pos_camera)
            
            mc_set.dist_profile(ax = ax, surface = True, color = dict_atm[atm][1], \
                                name = dict_atm[atm][0], linestyle = "dashed")
            mc_set.dist_profile(ax = ax, surface = False, color = dict_atm[atm][1], \
                                name = dict_atm[atm][0], linestyle = "solid")
            
    ax.legend()
    #axs[0].set_xlim((0,1000))
    #axs[1].set_xlim((0,30))
    
    ax.set_xscale('log')
    
    if force_save :
    
        plt.savefig(f"/home/barroisl/Transect_MC_auto/Output/fake_topo/{save_name}.jpg")
  
"""  
plot_atm_density_abs_guiers(force_save = True,
                    save_name = "dist_abs_profile_guiers_11")
"""

def plot_atm_density_abs_wlen_guiers(
        force_save : bool,
        save_name : str,
        surf : bool,
        atm : bool):
    
    path_dir = "/home/barroisl/Transect_MC_auto/Data/guiers_250_144_atms/"
    
    cmap = cm.autumn_r
    
    fig, axs = plt.subplots(xshare = True, nrows = len(atms[:-2]), ncols = 1, figsize = (12,16), layout = 'constrained')
    
    for j,atm in tqdm(enumerate(atms[:-2])):
        
        atm_set = Prop_atm(atm)

        mask = (atm_set.LW_int_edges[:, 0] > 3000) & (atm_set.LW_int_edges[:, 1] < 45000)        
        LW_ints = atm_set.LW_int_edges[mask]

        for i,LW_int in enumerate(LW_ints):

            data = np.loadtxt(path_dir + atm + '/11/11_50_15_15.txt')
            pos_camera = np.loadtxt(path_dir + atm +'/11/11_camera_tgt.txt')[:3]
            mc_set = MC_Set(data, pos_camera)
            
            if atm : 
                mc_set.density_dist_profile_wlen(ax = axs[j], surface = False, wlen_1 = LW_int[0], wlen_2 = LW_int[1],\
                                            color = cmap((LW_int[1]-min(LW_ints[:,0]))/(max(LW_ints[:,1])-min(LW_ints[:,0]))),\
                                            name = f"[{int(LW_int[0]*1e-3)};{int(LW_int[1]*1e-3)}] $\mu m$",\
                                            linestyle = 'solid')
            

            if surf :
                mc_set.density_dist_profile_wlen(ax = axs[j], surface = True, wlen_1 = LW_int[0], wlen_2 = LW_int[1],\
                                            color = cmap((LW_int[1]-min(LW_ints[:,0]))/(max(LW_ints[:,1])-min(LW_ints[:,0]))),\
                                            name = f"[{int(LW_int[0]*1e-3)};{int(LW_int[1]*1e-3)}] $\mu m$",\
                                            linestyle = 'dashed')
        
        if  surf :
            mc_set.density_dist_profile_wlen(ax = axs[j], surface = True, wlen_1 = min(LW_ints[:,0]), wlen_2 = max(LW_ints[:,1]),\
                                            color = 'k',\
                                            name = "All $\lambda$",\
                                            linestyle = 'dashed')
                                            
        if atm :
            mc_set.density_dist_profile_wlen(ax = axs[j], surface = False, wlen_1 = min(LW_ints[:,0]), wlen_2 = max(LW_ints[:,1]),\
                                            color = 'k',\
                                            name = "All $\lambda$",\
                                            linestyle = 'solid')
            
        #axs[j].set_xlim((0,300))
        axs[j].set_title(atm)
        
        axs[j].set_xscale('log')
        
    axs[0].legend()
    
    if force_save :
    
        plt.savefig(f"/home/barroisl/Transect_MC_auto/Output/fake_topo/{save_name}.jpg")
        
"""
plot_atm_density_abs_wlen_guiers(
        force_save = True,
        save_name = 'Abs_spectrale_atm_guiers',
        surf = True,
        atm = True)
"""

#################################### Fake canyonne DEM #############################


def plot_canyonne_DEM(force_save : bool,
                      save_name : str):
	fig, ax = plt.subplots(figsize = (12,6), layout = 'constrained')

	largeur = 100000
	SVF_geom = 0.5

	distances = 500+np.arange(1,10)*100
	hs = np.tan(np.arccos(SVF_geom))*distances

	for i,d in enumerate(distances):
	    
	    ax.hlines(y = hs[i], xmin = 0, xmax = 50000-d//2, linestyle = 'solid')
	    ax.vlines(x = 50000-d//2, ymin = 0, ymax = hs[i])
	    ax.hlines(y = 0, xmin = 50000-d//2, xmax = 50000+d//2, linestyle = 'solid')
	    ax.vlines(x = 50000+d//2, ymin = 0, ymax = hs[i])
	    ax.hlines(y = hs[i], xmin = 50000+d//2, xmax = 100000, linestyle = 'solid')
	    
	distances = np.arange(1,50)*10
	hs = np.tan(np.arccos(SVF_geom))*distances

	for i,d in enumerate(distances):
	    
	    ax.hlines(y = hs[i], xmin = 0, xmax = 50000-d//2, linestyle = 'solid')
	    ax.vlines(x = 50000-d//2, ymin = 0, ymax = hs[i])
	    ax.hlines(y = 0, xmin = 50000-d//2, xmax = 50000+d//2, linestyle = 'solid')
	    ax.vlines(x = 50000+d//2, ymin = 0, ymax = hs[i])
	    ax.hlines(y = hs[i], xmin = 50000+d//2, xmax = 100000, linestyle = 'solid')
	    
	    
	ax.set_xlim(50000-2000,50000+2000)
	ax.set_ylim(0,3000)
	ax.set_xlabel('x (m)')
	ax.set_ylabel('y (m)')
	ax.spines[['left','right', 'top', 'bottom']].set_visible(False)
	ax.set_aspect('equal')
	ax.set_title('Idealized vertical cliffs',fontsize = 20)
	
	if force_save :
	

	    plt.savefig(f"/home/barroisl/Transect_MC_auto/Output/fake_2/{save_name}.jpg")
	    
def altit_T(dem, T0, z0, lapse_rate = -0.0065):
    
    return T0+lapse_rate*(dem-z0)
    
def fraction_neigeuse(T,T0,delta):
    return np.exp((T-T0)/delta)/(np.exp((T-T0)/delta)+1)

def obj_mtls_dem_vhs_distributed(fic_ds,fic_obj,fic_mtls,T0=273,z0=1300,delta=50,lapse_rate = -0.0065):
    
    """
    [entrée]
    - fic_tif le chemin vers le fichier .tif
    - fic_obj le chemin de sortie du fichier .obj
    - fic_mtls le chemin de sortie du fichier .mtls
    - T0 la température en z0, ici au niveau de la mer : 25°C
    - z0 l'altitude auquelle on trouve T0, au niveau de la mer
    - delta la distance caractéristique de variation de la fraction neigeuse
    - le lapse rate de variation de la température dans les 11 km de la troposphere
    
    [sortie]
    - Aucune, écriture des fichier .obj et .mtls pour htrdr
    """
        
    ds = xr.open_dataset(fic_ds)
    zs = ds.elevation.values
    x,y = ds.x.values, ds.y.values
    
    #zs, src, transform = lect_tif(fic_tif)
        
    ts =  altit_T(zs, T0, z0, lapse_rate)
    
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
                d_ts[ts_arr].append( (j,i) )
                
    lignes_mtls =  []
    
    #cwd = os.getcwd() # répertoire actuel
    cwd='${HTRDR_ATMOSPHERE_SPK}'
    #cwd='/home/barroisl/edstar/Simus'
    #print("f")
    for ts_arr in sorted(d_ts): # liste triée des Ts arrondies
            
        # Nombre de facettes à la température ts_arr rangées dans le dictionnaire d_ts
        n_ts_arr = len(d_ts[ts_arr])
        
        # Pourcentage de facettes recouvertes de neige d'après fraction_neigeuses(zsnow,delta)
        # La hauteur pour laquelle la fraction neigeuse est de 0.5 est celle où la température est de 273 K
        n_ts_arr_snow = int(n_ts_arr*fraction_neigeuse(ts_arr,273,delta)) 

        # On définit des indices pris au hasard dans n_ts_arr au nombre de n_ts_arr_snow 
        # pour faire tourner le choix des facettes sans préférence dans l'ordre d'apparition
        # dans la liste d_ts[ts_arr] afin d'éviter toute séparation géographique non physique
        all_ints = set(range(0, n_ts_arr))
        indices_snow = random.sample(all_ints, n_ts_arr_snow)
        indices_sand = list(all_ints - set(indices_snow))
        
        # Les facettes sont triées aléatoirement & les facettes trop inclinées ne sont plus recouvertes de neige
        ts_arr_snow = [d_ts[ts_arr][k] for k in indices_snow] 
        ts_arr_sand = [d_ts[ts_arr][k] for k in indices_sand]
        
        if len(ts_arr_snow)  > 0 :
                
            mrumtl = 'snow'
            nom_mat = mrumtl + str(ts_arr).replace('.','_') # ex.'snow238_1'
            lignes_obj.append(f'usemtl air:{nom_mat}\n')

            # Boucles sur les facettes de température ts_arr d'indice inférieur à frac_neige*len(d_ts)
            # dans d_ts.sorted() --> Toutes les facettes recouvertes de neige
            #for j,i in d_ts[ts_arr][indices_snow]:# points avec la Ts arrondie ts_arr
            for j,i in ts_arr_snow :
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
        
        
        if len(ts_arr_sand)  > 0 :
        
            if ts_arr > (1500 - z0)*lapse_rate + T0:
                mrumtl = 'forest_green'
                
            elif ts_arr < (1500 - z0)*lapse_rate + T0 and ts_arr > (2000 - z0)*lapse_rate + T0 :
                mrumtl = 'grass'
                
            else :
                mrumtl = 'limestone'
                
            nom_mat = mrumtl + str(ts_arr).replace('.','_') # ex.'snow238_1'
            lignes_obj.append(f'usemtl air:{nom_mat}\n') 

            #for j,i in d_ts[ts_arr][indices_sand]:
            for j,i in ts_arr_sand :
                # points avec la Ts arrondie ts_arr
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
        
    lignes_mtls.append('air none\n')
    
    print("lignes_mtls OK : Nbre %d ; " %len(lignes_mtls), lignes_mtls[0])

    #On change les lignes de l\'.obj pour faire apparaître une fraction neigeuse
       
    with open(fic_obj,'w') as fid: # w: créer, défaut: encoding='UTF-8'
        fid.writelines(lignes_obj)
    with open(fic_mtls,'w') as fid:
        fid.writelines(lignes_mtls)
        
"""

!rm -f /home/barroisl/edstar/Simus/models/fake/0.5_*.obj
!rm -f /home/barroisl/edstar/Simus/materials/fake/0.5_*.mtls
    
for i,dist in enumerate(distances):

    fic_ds = f"/home/barroisl/Transect_MC_auto/topographie/fake/0.5/{0.5}_{dist}.nc"
    fic_obj = f"/home/barroisl/edstar/Simus/models/fake/{0.5}_{dist}.obj"
    fic_mtls = f"/home/barroisl/edstar/Simus/materials/fake/{0.5}_{dist}.mtls"
    obj_mtls_dem_vhs_distributed(fic_ds,fic_obj,fic_mtls,T0=273,z0=1300,delta=50,lapse_rate = -0.0065)
    

"""

def compute_SVF_htrdr_flux(n_dist : int,
                          svf : float):
    
    path_dir = f"/home/barroisl/Transect_MC_auto/Data/fake_2/"
    distances = os.listdir(path_dir+"SMLSATM/")
      
    dtype = np.dtype([
        ("distance",    np.float64),
        ("SMLSATM",     np.float64),
        ("SMLWATM",     np.float64),
        ("SPOSATM",     np.float64),
        ("SPOWATM",     np.float64),
        ("STROATM",     np.float64),
        ("EMPTATM",     np.float64),
        ("TUXUATM",     np.float64),

    ])
    
    flux_surf = np.zeros(n_dist, dtype=dtype)
    taux_surf = np.zeros(n_dist, dtype=dtype)
    flux_atm = np.zeros(n_dist, dtype=dtype)

    for i,key in enumerate(dict_atm.keys()):
        

        l_distances = []
        l_flux_surf = []
        l_taux_surf = []
        l_flux_atm = []

        for j,d in tqdm(enumerate(distances)):
            data = np.loadtxt(path_dir + key + '/' + d +'/_50_15_15.txt')
            pos_camera = np.array([50000,50000,0])
            mc_set = MC_Set(data, pos_camera)

            l_distances.append(int(d)/2)
            l_flux_surf.append(mc_set.W_surf*mc_set.W_tot/mc_set.Nbre_photon)
            l_taux_surf.append(mc_set.Nbre_surf/mc_set.Nbre_photon)
            l_flux_atm.append(mc_set.W_atm*mc_set.W_tot/mc_set.Nbre_photon)

        if key == 'SPOSATM' :
            flux_surf["distance"] = l_distances
            taux_surf["distance"] = l_distances
            flux_atm["distance"] = l_distances

        flux_surf[key] = l_flux_surf
        taux_surf[key] = l_taux_surf
        flux_atm[key] = l_flux_atm
        
    np.save(f"/home/barroisl/Transect_MC_auto/Output/fake_2/flux_surf_{svf}.npy",flux_surf)
    np.save(f"/home/barroisl/Transect_MC_auto/Output/fake_2/taux_surf_{svf}.npy",taux_surf)
    np.save(f"/home/barroisl/Transect_MC_auto/Output/fake_2/flux_atm_{svf}.npy",flux_atm)
    
    
"""
!rm -f /home/barroisl/Transect_MC_auto/Output/fake_2/flux_surf_0.5.npy
!rm -f /home/barroisl/Transect_MC_auto/Output/fake_2/taux_surf_0.5.npy
!rm -f /home/barroisl/Transect_MC_auto/Output/fake_2/flux_atm_0.5.npy


compute_SVF_htrdr_flux(n_dist = 58,
                      svf = 0.5)  
"""

def plot_svf_distance(
        svf : float,
        save_name : str,
        force_save = False,
    ):
    
    flux_surf = np.load(f"/home/barroisl/Transect_MC_auto/Output/fake_2/flux_surf_{svf}.npy", allow_pickle = True)
    taux_surf = np.load(f"/home/barroisl/Transect_MC_auto/Output/fake_2/taux_surf_{svf}.npy", allow_pickle = True)
    flux_atm = np.load(f"/home/barroisl/Transect_MC_auto/Output/fake_2/flux_atm_{svf}.npy", allow_pickle = True)

    fig,axs = plt.subplots(nrows = 2, ncols = 1, figsize = (12,6), layout = 'constrained')

    for i,atm in enumerate(atms):            
        if i == 8 :
            axs[0].plot([np.min(taux_surf['distance']),np.max(taux_surf['distance'])],[0.35,0.35],linestyle ='dashed', \
                        color = 'k',label = 'Dozier & Frew')
        axs[0].plot(taux_surf['distance'],(1-taux_surf[atm]), color = dict_atm[atm][1],\
                       label = dict_atm[atm][0]) #topo_params[:,-1]

        cax = axs[1].plot(flux_atm['distance'],flux_atm[atm], color = dict_atm[atm][1],\
                             label = dict_atm[atm][0])
            
    for ax in axs :
        ax.spines[['left','right', 'top', 'bottom']].set_visible(False)
        ax.set_xlabel('Distance to cliff')
        ax.set_ylabel('SVF')
        #ax.set_aspect('equal')
        
    axs[1].set_ylabel('Atm LW flux density ($W.m^{-2}$)')
    axs[0].set_xticks([])
    axs[0].set_xlabel('')
    #axs[0].set_ylim((0.7,1))
    #axs[1].set_ylim((300,None))
    axs[1].legend()

    dire = "/home/barroisl/Transect_MC_auto/"
    
    if force_save :

        plt.savefig(dire+f"Output/fake_2/{save_name}.jpg")

"""
plot_svf_distance(
        svf = 0.5,
         save_name = 'fake_topo_SVF_flux_0.5',
        force_save = True)
"""

################################# Exctracting informations out of htrdr product##################

dict_map = {"x" : 0,
           "y" : 1,
           "z" : 2,
           "svf" : 3,
           "svf_lslope" : 4,
           "aspect" : 5,
           "slope" : 6,
           "elevation" : 7,
           "flux" : 8,
           "f_atm" : 9,
           "f_surf" : 10,
           "f_refl" : 11,
           "Nbre_atm" : 12,
           "Nbre_surf" : 13,
           "Nbre_refl" : 14,
           "90_dist" : 15,
           "50_dist" : 16,
           "10_dist" : 17,
           "90_alt" : 18,
           "CN" : 19,
           "flux_90_dist" : 20,
           "flux_90_alt" : 21,
           }
           
######################### Extracting and ordening data ########################

def from_transect_to_pdt(
    fic_topo_param : str,
    dir_data : str,
    fic_cam_tgt : str,
    fic_res : str,
    ds : xr.Dataset,
    ):
    
    # Paramètres topographiques topocalc
    topo_params = np.loadtxt(fic_topo_param)
    a,b = topo_params.shape
    
    #Source xr.Dataset
    #ds = xr.open_dataset("/home/barroisl/Transect_MC_auto/s2m_simu/chartreuse_thomas.nc")
    
    # Cam_tgt file
    cam_tgt = np.loadtxt(fic_cam_tgt)
    
    list_dir = os.listdir(dir_data)
    
    res = np.zeros((len(list_dir),3+b+14))
        
    res[:,:3] = cam_tgt[:,:3]
    res[:,3:3+b] = topo_params
    
    # diagnostiques simulation htrdr
    L = []
    T_atm = []
    T_surf = []
    T_refl = []
    Nbre_atm = []
    Nbre_surf = []
    Nbre_refl = []
    Dist_90_quantile = []
    Dist_50_quantile = []
    Dist_10_quantile = []
    Alt_90_quantile = []
    Flux_s_CN = []
    Flux_90_quantile_dist = []
    Flux_90_quantile_alt = []
    
    for dossier in tqdm(list_dir):
        
        # Data des chemins htrdr
        data_path = np.loadtxt(dir_data + dossier + '/' + dossier +'_50_15_15.txt')
        pos_camera = np.loadtxt(dir_data + dossier + '/' + dossier +'_camera_tgt.txt')[:3]
        mc_set = MC_Set(data_path, pos_camera)
        
        L.append(mc_set.W_tot)
        T_atm.append(mc_set.W_atm)
        T_surf.append(mc_set.W_surf)
        T_refl.append(mc_set.W_reflechis_surf + mc_set.W_reflechis_atm)
        Nbre_surf.append(mc_set.Nbre_surf)
        Nbre_atm.append(mc_set.Nbre_atm)
        Nbre_refl.append(mc_set.Nbre_reflect_surf + mc_set.Nbre_reflect_surf)
        Dist_90_quantile.append(mc_set.quartil_cumul_collision("dist",90))
        Dist_50_quantile.append(mc_set.quartil_cumul_collision("dist",50))
        Dist_10_quantile.append(mc_set.quartil_cumul_collision("dist",10))
        Alt_90_quantile.append(mc_set.quartil_cumul_collision("alt",90))
        Flux_s_CN.append(mc_set.Pos_surf(ds = ds))
        Flux_90_quantile_dist.append(mc_set.quartil_cumul_flux("dist",90))
        Flux_90_quantile_alt.append(mc_set.quartil_cumul_flux("alt",90))
        
    res[:,3+b] = np.array(L)
    res[:,3+b+1] = np.array(T_atm)
    res[:,3+b+2] = np.array(T_surf)
    res[:,3+b+3] = np.array(T_refl)
    res[:,3+b+4] = np.array(Nbre_surf)
    res[:,3+b+5] = np.array(Nbre_atm)
    res[:,3+b+6] = np.array(Nbre_refl)
    res[:,3+b+7] = np.array(Dist_90_quantile)
    res[:,3+b+8] = np.array(Dist_50_quantile)
    res[:,3+b+9] = np.array(Dist_10_quantile)
    res[:,3+b+10] = np.array(Alt_90_quantile)
    res[:,3+b+11] = np.array(Flux_s_CN)
    res[:,3+b+12] = np.array(Flux_90_quantile_dist)
    res[:,3+b+13] = np.array(Flux_90_quantile_alt)

    np.savetxt(fic_res,res,delimiter = '\t',newline='\n')
    
"""
dire = "/home/barroisl/Transect_MC_auto/"
fic_topo_param = dire + "camera_tgt/topo_polygon_guiers.txt"
dir_data = dire + "Data/guiers_250_144_atms/TUXUATM/"
fic_cam_tgt = dire + "camera_tgt/polygon_guiers.txt"
fic_res = dire + "Output/guiers_250_144_atms/topo_flux_TUXUATM.txt"

from_transect_to_pdt(
    fic_topo_param,
    dir_data,
    fic_cam_tgt,
    fic_res)
"""
    

def d_abs_pos_flux(dir_data : str,
              dir_cam_tgt : str,
              dir_res_topo_flux : str,
          ):
    list_dir = os.listdir(dir_data)
    d_abs = {}
    
    cam_tgt = np.loadtxt(dir_cam_tgt)
    topo_flux = np.loadtxt(dir_res_topo_flux)

    for dossier in tqdm(list_dir):
        
        # Data des chemins htrdr
        data_path = np.loadtxt(dir_data + dossier + '/' + dossier +'_50_15_15.txt')
        pos_camera = np.loadtxt(dir_data + dossier + '/' + dossier +'_camera_tgt.txt')[:3]
        mc_set = MC_Set(data_path, pos_camera)

        pathes_s = [mc_set.paths[i] for i in range(len(mc_set.paths)) if \
             mc_set.paths[i].Abs_surf == True and mc_set.paths[i].Space == False]
        
        Pos = np.zeros((len(pathes_s),12))
        if len(pathes_s) > 0 :
            for i in range(len(pathes_s)):
                Pos[i,:2] = pathes_s[i].Pos_abs[:2]
                Pos[i,2] = pathes_s[i].W
                Pos[i,3:5] = cam_tgt[int(dossier)-1,:2]
                Pos[i,5:7] = cam_tgt[int(dossier)-1,3:5]
                Pos[i,7:10] = topo_flux[int(dossier)-1,15:18]
                Pos[i,10:12] = topo_flux[int(dossier)-1,18:20]
            
        d_abs[int(dossier)] = Pos
        
    return d_abs
    
def flux_npy(
        exp : str,
        res : str,
        n_cam : str,
        uniforme : str,
        atms = atms):
    
    directory = f"/home/barroisl/Transect_MC_auto/Data/{exp}_{res}_{n_cam}_atms_{uniforme}_emis/"

    Nbre_cam = int(n_cam)
    taux_surf = np.zeros((Nbre_cam,len(atms)))
    taux_atm = np.zeros((Nbre_cam,len(atms)))
    flux_surf = np.zeros((Nbre_cam,len(atms)))
    flux_atm = np.zeros((Nbre_cam,len(atms)))
    
    for i,atm in enumerate(atms):

        listdir = os.listdir(directory+atm)

        for dossier in tqdm(listdir) :

            dir_data = directory+atm+"/"
            data_path = np.loadtxt(dir_data + dossier + '/' + dossier +'_50_15_15.txt')
            pos_camera = np.loadtxt(dir_data + dossier + '/' + dossier +'_camera_tgt.txt')[:3]
            mc_set = MC_Set(data_path, pos_camera)
            flux_surf[int(dossier)-1,i] = mc_set.W_surf*mc_set.W_tot/mc_set.Nbre_photon
            taux_surf[int(dossier)-1,i] = mc_set.Nbre_surf/mc_set.Nbre_photon
            taux_atm[int(dossier)-1,i] = mc_set.Nbre_atm/mc_set.Nbre_photon
            flux_atm[int(dossier)-1,i] = mc_set.W_atm*mc_set.W_tot/mc_set.Nbre_photon

    np.save(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms_{uniforme}_emis/flux_surf.npy",flux_surf)
    np.save(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms_{uniforme}_emis/taux_surf.npy",taux_surf)
    np.save(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms_{uniforme}_emis/taux_atm.npy",taux_atm)
    np.save(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms_{uniforme}_emis/flux_atm.npy",flux_atm)
    
""" 
flux_npy(
        exp = "lavey",
        res = "30",
        n_cam = "100",
        uniforme = 'T',
        atms = atms)
"""
 
######################## Plotting symbolique ###################

def plot_MC_symoblique_test(
    ax,
    atm,
    topo_flux,
    dict_map,
    plot_reg,
    alpha,
    epsilon_surf,
    ):
        
    #c = (topo_flux[:,dict_map["CN"]] - topo_flux[:,dict_map["f_surf"]]*topo_flux[:,dict_map["flux"]]/(15*15*50))#\
        #/(topo_flux[:,dict_map["f_surf"]]*topo_flux[:,dict_map["flux"]]/(15*15*50))
    cax = ax.scatter(topo_flux[:,dict_map["f_surf"]]*topo_flux[:,dict_map["flux"]]/(15*15*50),\
                     epsilon_surf*topo_flux[:,dict_map["CN"]], c = dict_atm[atm][1], label = dict_atm[atm][0],\
                    alpha = alpha)#c = c, cmap = 'RdBu_r', vmin = -2, vmax = 2, marker = 'o', s = 15)
    ax.plot([0,200],[0,200],linestyle = 'dashed', color = 'k', linewidth = 1)

    """
    cmap = mpl.cm.RdBu_r
    bounds = [-2+0.2*i for i in range(21)]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax = ax, orientation='vertical',
                 location = 'right',extend = 'both',label = 'Relative bias to MC density flux')
    """

    if plot_reg == True :
        model = LinearRegression()
        Y = topo_flux[:,dict_map["CN"]]
        X = (topo_flux[:,dict_map["f_surf"]]*topo_flux[:,dict_map["flux"]]/(15*15*50))[np.isnan(Y) == False].reshape(-1, 1)
        Y = topo_flux[:,dict_map["CN"]][np.isnan(Y) == False]

        model.fit(X,Y)
        a = model.coef_,[0],      # pente
        b = model.intercept_   # ordonnée à l'origine
        X_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_fit = model.predict(X_fit)
        ax.plot(X_fit, y_fit, color='red', alpha = 0.5, linestyle = 'dashed') #, label=f"y={round(a[0][0],2)}*x {round(b,2)}")

    ax.set_ylabel('$\epsilon \sigma T^4$ ($W.m^{-2}$)')
    #ax.set_xlabel('Surface LW radiative flux density htrdr ($W.m^{-2}$)')
    ax.set_xlabel('')
    #ax.set_title('Densité de flux radiatif dû à la surface pour MC vs atm gris')
    ax.spines[['left','right', 'top', 'bottom']].set_visible(False)


"""
dire = "/home/barroisl/Transect_MC_auto/"

fig,axs = plt.subplots(sharex=True,nrows = 2, ncols = 1, figsize = (20,6))

for i,atm in enumerate(atms[:-1]): 
    fic_res = dire + "Output/guiers_250_144_atms/topo_flux_%s.txt" %atm
    topo_flux = np.loadtxt(fic_res)

    if i == 1 :
        plot_reg = False
    else :
        plot_reg = False

    plot_MC_symoblique_test(
        ax = axs[0],
        atm = atm,
        topo_flux = topo_flux,
        dict_map = dict_map,
        plot_reg = plot_reg,
        alpha = 0.6)

    axs[0].set_xticks([])
    axs[1].set_xticks(np.arange(0,70,5))
    axs[1].hist(topo_flux[:,dict_map["f_surf"]]*topo_flux[:,dict_map["flux"]]/(15*15*50), density = True,\
               alpha = 0.3, color = dict_atm[atm][1],histtype='step', align = 'mid',linewidth = 5)
    axs[1].set_yticks([])
    axs[1].spines[['left','right', 'top', 'bottom']].set_visible(False)
    axs[1].set_xlabel('Surface LW radiative flux density htrdr ($W.m^{-2}$)')
    axs[0].legend()

plt.savefig(dire + "Output/symbolique_hist.jpg")
"""

############################ Basic stats ###############################

def  plotting_basic_stats(        
        exp : str,
        res : str,
        n_cam : str,
        lapse_rate : bool,
        save_name : str,
        atm : str,
        force_save = False):
    
    if exp == "lavey":
        if lapse_rate :
            taux_surf = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms_65_emis/taux_surf.npy")
            taux_atm = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms_65_emis/taux_atm.npy")
            
        else :
            taux_surf = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms_00_emis/taux_surf.npy")
            taux_atm = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms_00_emis/taux_atm.npy")          
    else :
    
        taux_surf = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms/taux_surf.npy")
        taux_atm = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms/taux_atm.npy")
    
    n_cam,n_atm = taux_atm.shape
    if atm == "SMLSATM":
        index_atm = 1
    elif atm == "SMLWATM":
        index_atm = 2 
    elif atm == "TUXUATM" :
        index_atm = 3

    index_EMPTATM = 0
    
    fig,ax = plt.subplots(figsize = (12,4), layout = 'constrained')
    
    ax.fill_between(x = np.arange(n_cam),y1 = 100*(taux_atm[:,index_atm] + taux_surf[:,index_atm]),\
                    y2 = np.ones(n_cam)*100, color = 'black', label = 'Espace')
    ax.fill_between(x = np.arange(n_cam),y1 = 100*taux_surf[:,index_atm],\
                    y2 = 100*(taux_surf[:,index_atm]+taux_atm[:,index_atm]), color = 'skyblue', label = 'Atmosphère')
    ax.fill_between(x = np.arange(n_cam), y1 = np.zeros(n_cam),\
                    y2 = 100*taux_surf[:,index_atm],\
                    color = 'lightcoral', label = 'Surface')
    ax.plot(np.arange(n_cam), 100*taux_surf[:,index_EMPTATM], linewidth = 2, \
            color = 'lightcoral',label = 'Transparent atm surface')
    
    ax.set_xlabel("Index caméra")
    ax.set_ylabel("Origine des chemins (%)")
    ax.legend()
    
    ax.spines[['left','right', 'top', 'bottom']].set_visible(False)
    
    ax.set_title(f'{exp} : res = {res}m, n_cam = {n_cam}, atm = {dict_atm[atm][0]}', fontsize = 15)
    
    if force_save :
        if exp == "lavey" and lapse_rate :
            plt.savefig(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms_65_emis/{save_name}.png")
        elif exp == 'lavey' and not lapse_rate :
            plt.savefig(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms_00_emis/{save_name}.png")
        elif exp != "lavey" :
            plt.savefig(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms/{save_name}.png")
        
def view_factors(        
        exp : str,
        res : str,
        n_cam : str,
        lapse_rate : bool,
        save_name : str,
        force_save = False):
        
    if exp == "lavey":
        if lapse_rate :
            taux_surf = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms_65_emis/taux_surf.npy")
        else :
            taux_surf = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms_00_emis/taux_surf.npy")
            
    else :
    
        taux_surf = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms/taux_surf.npy")
        taux_atm = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms/taux_atm.npy")
        
    n_cam,n_atm = taux_atm.shape
    index_atm = 1 #SMLSATM
    
    fig,ax = plt.subplots(figsize = (12,4), layout = 'constrained')
    
    ax.plot(np.arange(n_cam), 1-(taux_surf[:,index_atm]+taux_atm[:,index_atm]), \
            color = 'black', label = 'Espace')
    ax.plot(np.arange(n_cam), taux_atm[:,index_atm], \
            color = 'skyblue', label = 'Atmosphere')
    ax.plot(np.arange(n_cam), taux_surf[:,index_atm], \
            color = 'lightcoral', label = 'Surface')
    ax.plot(np.arange(n_cam), taux_surf[:,5], \
            color = 'lightcoral', linestyle = 'dashed', label = 'Surface atm transparent')
    
    """
    ax.fill_between(x = np.arange(n_cam),y1 = 100*(1-(taux_surf[:,index_atm]+taux_atm[:,index_atm])),\
                    y2 = 100*(1-(taux_surf[:,index_atm])), color = 'skyblue', label = 'Atmosphère')
    ax.fill_between(x = np.arange(n_cam),y1 = 100*(1-(taux_surf[:,index_atm])),\
                    y2 = np.ones(n_cam)*100, color = 'lightcoral', label = 'Surface')
    """
    ax.set_xlabel("Index caméra")
    ax.set_ylabel("View factors")
    ax.legend()
    
    ax.spines[['left','right', 'top', 'bottom']].set_visible(False)
    
    ax.set_title(f'Guiers : res = {res}m, n_cam = {n_cam}, atm = SMLSATM', fontsize = 15)
    
    if force_save :
        plt.savefig(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms/{save_name}.png")

"""
plotting_basic_stats(        
        exp = "ecrins",
        res = "30",
        n_cam = "151",
        save_name = 'basic_stats',
        force_save = True)
"""
    
############################################### SVF and Atm fluxes ################################


def plotting_atms_SVF_and_flux(
        exp : str,
        res : str,
        n_cam : str,
        save_name : str,
        lapse_rate : bool,
        force_save = False):
    
    if exp == "lavey" and lapse_rate :
    
        flux_surf = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms_65_emis/flux_surf.npy")
        taux_surf = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms_65_emis/taux_surf.npy")
        taux_atm = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms_65_emis/taux_atm.npy")
        flux_atm = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms_65_emis/flux_atm.npy")

    elif exp == "lavey" and not lapse_rate :
        
        flux_surf = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms_00_emis/flux_surf.npy")
        taux_surf = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms_00_emis/taux_surf.npy")
        taux_atm = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms_00_emis/taux_atm.npy")
        flux_atm = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms_00_emis/flux_atm.npy")
        
    elif exp != "lavey" :
   
        flux_surf = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms/flux_surf.npy")
        taux_surf = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms/taux_surf.npy")
        taux_atm = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms/taux_atm.npy")
        flux_atm = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms/flux_atm.npy")  
        
    topo_params = np.loadtxt(f"/home/barroisl/Transect_MC_auto/camera_tgt/topo_polygon_{exp}_{res}_{n_cam}.txt")
    
    

    fig,axs = plt.subplots(nrows = 2, ncols = 1, figsize = (12,6), layout = 'constrained')
    
    dict_atm = {"SMLSATM" : ["Mid latitude summer", "red", "Reds", cm.Reds, ],
            "SMLWATM" : ["Mid latitude winter", "green", "Greens", cm.Greens],
            "SPOSATM" : ["Polar summer", "orange","Oranges",cm.Oranges],
            "SPOWATM" : ["Polar winter", "blue","Blues",cm.Blues],
            "STROATM" : ["Tropics", "purple","Purples",cm.Purples],
            "EMPTATM" : ["Transparent", "black","Greys",cm.Greys],
            "TUXUATM" : ["Uniforme", "yellow","Blues", cm.Blues]}
    atms = ["SMLSATM", "SMLWATM","SPOSATM","SPOWATM","STROATM","EMPTATM","TUXUATM"]
    for i,atm in enumerate(atms):
        
        if atm == "TUXUATM":
            fact = 0.5
        else :
            fact = 1
	

        axs[0].scatter(topo_params[:,0],(1-taux_surf[:,i]), marker = '*', s = 70, c = dict_atm[atm][1],\
                       label = dict_atm[atm][0], alpha = 0.7) #topo_params[:,-1]
        axs[0].plot([min(topo_params[:,0]),max(topo_params[:,0])],[min(topo_params[:,0]),max(topo_params[:,0])],linestyle ='dashed', color = 'k')
        cax = axs[1].scatter(topo_params[:,0],flux_atm[:,i]*fact, marker = 'o', s = 70, c = topo_params[:,-1],\
                             cmap = dict_atm[atm][2],label = dict_atm[atm][0], \
                             edgecolors='black', linewidths=0.3) #topo_params[:,-1]
    for ax in axs :
        ax.spines[['left','right', 'top', 'bottom']].set_visible(False)
        ax.set_xlabel('$SVF_{D&F}$')
        ax.set_ylabel('$SVF_{htrdr}$')
        #ax.set_aspect('equal')
        #ax.set_xlim((0.77,1))

    axs[1].set_ylabel('Atm LW radiative flux density ($W.m^{-2}$)')
    axs[0].set_xticks([])
    axs[0].set_xlabel('')
    axs[0].legend()

    
    # Colorbar indépendante du plot
    vmin = round(min(topo_params[:,-1])/10)*10
    vmax = round(max(topo_params[:,-1])/10)*10
    cmap = cm.binary   
    bounds = np.arange(vmin,vmax, 100)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)   
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])                     
    cbar = fig.colorbar(sm, ax=axs[1], orientation='vertical', boundaries = bounds,
                        fraction=0.046, pad=0.04)   
    cbar.set_label('Elevation (m)')     

    dire = "/home/barroisl/Transect_MC_auto/"
    
    if force_save == True :
    
        if exp == 'lavey' and lapse_rate :

            plt.savefig(dire+f"Output/{exp}_{res}_{n_cam}_atms_65_emis/{save_name}.jpg")
           
        elif exp == 'lavey' and not lapse_rate :
        
            plt.savefig(dire+f"Output/{exp}_{res}_{n_cam}_atms_00_emis/{save_name}.jpg")
            
        elif exp != "lavey" :
        
            plt.savefig(dire+f"Output/{exp}_{res}_{n_cam}_atms/{save_name}.jpg")
            
   
"""     
plotting_atms_SVF_and_flux(
        exp = "lavey",
        res = "30",
        n_cam = "100",
        save_name = "comp_svf_flux_lavey",
        lapse_rate = True,
        force_save = False)
"""

########################################### Modelisation simple ##################################

def correlation_htrdr_model(
        exp : str,
        res : str,
        n_cam : str,
        save_name : str,
        alpha : float,
        ds_chartreuse : xr.Dataset,
        lapse_rate : bool,
        atm_ : str,
        force_save = False,
        epsilon_surf = 0.98,
        epsilon_atm = 0.98,
        sigma = 5.67e-8):
    
    if exp == 'lavey' :
        if lapse_rate == True :
            grad = '65'
        else :
            grad = '00'
    
        flux_surf = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms_{grad}_emis/flux_surf.npy")
        taux_surf = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms_{grad}_emis/taux_surf.npy")
        taux_atm = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms_{grad}_emis/taux_atm.npy")
        flux_atm = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms_{grad}_emis/flux_atm.npy")

    else :
        flux_surf = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms/flux_surf.npy")
        taux_surf = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms/taux_surf.npy")
        taux_atm = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms/taux_atm.npy")
        flux_atm = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_{n_cam}_atms/flux_atm.npy")
        
    topo_params = np.loadtxt(f"/home/barroisl/Transect_MC_auto/camera_tgt/topo_polygon_{exp}_{res}_{n_cam}.txt")
    cam_tgt = np.loadtxt(f"/home/barroisl/Transect_MC_auto/camera_tgt/polygon_{exp}_{res}_{n_cam}.txt")

    fig,axs = plt.subplots(nrows = 2, ncols = 1, figsize = (12,6), layout = 'constrained')
    
    emissivities = {"EMPTATM" : 0,
                    "TUXUATM" : 0.98,
                    "SMLSATM" : 0.62,
                    "SMLWATM" : 0.62}

    for i,atm in enumerate(atms):
        
        if atm in atm_ :
            
            ########## Atm flux #############
        
            prop_atm = Prop_atm(atm)
            T_profile = prop_atm.profiles(variable = "T")
            z_profile = prop_atm.profiles(variable = "z")     
            T = np.interp(topo_params[:,-1], z_profile, T_profile) #-6.5 # pour prendre la température un poil plus haut
            LW_CN = epsilon_atm*sigma*T**4

            axs[1].scatter(flux_atm[:,i],emissivities[atm]*topo_params[:,0]*LW_CN, marker = 'o', s = 50, alpha = alpha,\
                                 c = topo_params[:,-1],cmap = dict_atm[atm][2], label = "$SVF_{D&F}$",\
                          edgecolors='black', linewidths=1)
            axs[1].scatter(flux_atm[:,i],taux_atm[:,i]*LW_CN, marker = 'o', s = 50, alpha = alpha,\
                                 c = topo_params[:,-1],cmap = dict_atm[atm][2], label = "$SVF_{htrdr}$")
            axs[1].plot([min(flux_atm[:,i]),max(flux_atm[:,i])],[min(flux_atm[:,i]),max(flux_atm[:,i])], color = 'k', linestyle = 'dashed')
            
            ########## Surf flux #############
            
            LW_CN_self = epsilon_surf*sigma*interpolate(
                ds = ds_chartreuse,
                points = cam_tgt[:,:2],
                var_name = "ts")**4
            
            
            axs[0].scatter(flux_surf[:,i],(1-topo_params[:,0])*LW_CN_self, marker = 'o', s = 50, alpha = alpha,\
                                 c = topo_params[:,-1],cmap = dict_atm[atm][2], label = "$SVF_{D&F}$",\
                          edgecolors='black', linewidths=1)  
            axs[0].scatter(flux_surf[:,i],taux_surf[:,i]*LW_CN_self, marker = 'o', s = 50, alpha = alpha,\
                                 c = topo_params[:,-1],cmap = dict_atm[atm][2], label = "$SVF_{htrdr}$")
            axs[0].plot([min(flux_surf[:,i]),max(flux_surf[:,i])],[min(flux_surf[:,i]),max(flux_surf[:,i])],color = 'k', linestyle = 'dashed')
            
    
    for ax in axs :
        ax.spines[['left','right', 'top', 'bottom']].set_visible(False)
        ax.set_ylabel('$LW_{s}$ ($W.m^{-2}$)')
        #ax.legend()
        #ax.set_aspect('equal')
        #ax.set_xlim((0.77,1))
        #ax.set_ylim((0,1600))
        
    axs[0].set_xlabel('$LW_{htrdr,surf}$ ($W.m^{-2}$)')
    axs[0].set_ylabel('$LW_{model,surf}$ ($W.m^{-2}$)')
    axs[0].set_title('$S_{urface}VF_{htrdr/D&F}$ $\epsilon \sigma T_{cam}^{4}$ ; $\epsilon_{surf}$ = %s' %epsilon_surf, fontsize = 15)
    
    axs[1].set_xlabel('$LW_{htrdr,atm}$ ($W.m^{-2}$)')
    axs[1].set_ylabel('$LW_{model,atm}$ ($W.m^{-2}$)')
    axs[1].set_title('$A_{tm}VF_{htrdr/D&F}$ $\epsilon \sigma T_{2m}^{4}$ ; $\epsilon_{atm}$ = %s' %epsilon_atm, fontsize = 15)
    
    # Colorbar indépendante du plot
    

    vmin = round(min(topo_params[:,-1])/10)*10
    vmax = round(max(topo_params[:,-1])/10)*10
    cmap = cm.binary   
    bounds = np.arange(vmin,vmax, 100)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)   
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])                     
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', boundaries = bounds,
                        fraction=0.046, pad=0.04)   
    cbar.set_label('Elevation (m)')     

    dire = "/home/barroisl/Transect_MC_auto/"
    
    if force_save == True :

        plt.savefig(dire+f"Output/{exp}_{res}_{n_cam}_atms_{grad}_emis/{save_name}.jpg")
        
"""      
ds_ecrins = xr.open_dataset("/home/barroisl/Transect_MC_auto/topographie/lavey_topo_params_30_00.nc")
ds_ecrins['ts'] = (['y','x'], altit_T(dem = ds_ecrins['zs'].values, 
                                        T0 = 273.15,
                                        z0 = 1300, 
                                        lapse_rate = -6.5e-3))


correlation_htrdr_model(
        exp = "lavey",
        res = "30",
        n_cam = "100",
        save_name = "htrdr_vs_model_fluxes_lavey_SMLWATM",
        alpha = 0.5,
        epsilon_atm = 0.89,
        epsilon_surf = 0.98,
        atm_ = ["SMLWATM","SMLSATM","TUXUATM","EMPTATM"],
        ds_chartreuse = ds_ecrins,
        lapse_rate = True,
        force_save = False)
"""


    
        
