#!/usr/bin/env python
# coding: utf-8

from Functions import *
from matplotlib import pylab
import math

#################################### Class ###############################################

params = {'legend.fontsize': 'x-large',
          #'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

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

#@numba.njit()
def get_dist(Pos_abs,pos_camera):
    return np.sqrt(np.sum((Pos_abs[:2]-pos_camera[:2])**2))

@nb.njit()
def get_Azimuth(Pos_abs,pos_camera):
    return math.atan2(Pos_abs[1]-pos_camera[1],Pos_abs[0]-pos_camera[0])

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
        self.Azimuth = get_Azimuth(self.Pos_abs,pos_camera)
        
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
        
        if type_ == "dist" :
                
            Dist = np.array([pathes[i].Dist for i in range(len(pathes))])
            res = np.percentile(Dist, quantil)
        
        elif type_ == "alt" :
            
            Alt = np.array([pathes[i].Pos_abs[2] for i in range(len(pathes))])
            res = np.percentile(Alt, quantil)         
        
        return res
    
    
    def quartil_cumul_flux(self,type_,quantil):
        
        pathes = [self.paths[i] for i in range(len(self.paths)) if \
             self.paths[i].Abs_surf == True and self.paths[i].Space == False]   
        
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
               
    def Pos_surf(self,
                ds : xr.Dataset):
        
        sigma = 5.67e-8
        pathes = [self.paths[i] for i in range(len(self.paths)) if \
             self.paths[i].Abs_surf == True and self.paths[i].Space == False]
        
        Pos = np.array([pathes[i].Pos_abs[:2] for i in range(len(pathes))])
        Ts = interpolate(
                ds = ds,
                points = Pos,
                var_name ='ts',
                x_dim = "x",
                y_dim = "y",
                )
         
        return np.sum(sigma*Ts**4)/self.Nbre_photon
    
# Plot methods ------------------------------------------------------------------------------------
    
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
        
########################### Calculating stuff #################################################

def from_transect_to_pdt(
    fic_topo_param : str,
    dir_data : str,
    fic_cam_tgt : str,
    fic_res : str
    ):
    
    # Paramètres topographiques topocalc
    topo_params = np.loadtxt(fic_topo_param)
    a,b = topo_params.shape
    
    #Source xr.Dataset
    ds = xr.open_dataset("/home/barroisl/Transect_MC_auto/s2m_simu/chartreuse_thomas.nc")
    
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

atms = ["SMLSATM","SMLWATM","SPOSATM","SPOWATM","STROATM","EMPTATM"]

def produit_transect_atms(
    exp : str,
    res : str,
    fic_topo_param = "/home/barroisl/Transect_MC_auto/camera_tgt/topo_polygon_guiers.txt",
    atms = atms):

    for i,atm in enumerate(atms):
    
        from_transect_to_pdt(
	    dire + "camera_tgt/topo_polygon_%s.txt" %exp,
	    dire + f"Data/{exp}_{res}_atms/{atm}/",
	    dire + f"camera_tgt/polygon_{exp}.txt",
	    dire + "Output/{exp}_{res}_atms/topo_flux_%s.txt" %atm)
	   

dict_atm = {"SMLSATM" : ["Mid latitude summer", "red", "Reds"],
            "SMLWATM" : ["Mid latitude winter", "green", "Greens"],
            "SPOSATM" : ["Polar summer", "orange","Oranges"],
            "SPOWATM" : ["Polar winter", "blue","Blues"],
            "STROATM" : ["Tropics", "purple","Purples"],
            "EMPTATM" : ["Transparent", "black","Greys"]}
            
def flux_params(
		exp : str,
		res : str,
		dire = "/home/barroisl/Transect_MC_auto/Data/",
		dict_atm = dict_atm):

	directory = f"/home/barroisl/Transect_MC_auto/Data/{exp}_{res}_200/"
	
	Nbre_cam = len(os.listdir(directory+"EMPTATM/1/"))
	taux_surf = np.zeros((Nbre_cam,len(atms)))
	flux_surf = np.zeros((Nbre_cam,len(atms)))
	flux_atm = np.zeros((Nbre_cam,len(atms)))
	for i,key in enumerate(dict_atm.keys()):
	    
	    listdir = os.listdir(directory+key)
	    print(key + " OK")
	    for dossier in tqdm.tqdm(listdir) :

                dir_data = directory+key+"/"
                data_path = np.loadtxt(dir_data + dossier + '/' + dossier +'_50_15_15.txt')
                pos_camera = np.loadtxt(dir_data + dossier + '/' + dossier +'_camera_tgt.txt')[:3]
                mc_set = MC_Set(data_path, pos_camera)
                flux_surf[int(dossier)-1,i] = mc_set.W_surf*mc_set.W_tot/mc_set.Nbre_photon
                taux_surf[int(dossier)-1,i] = mc_set.Nbre_surf/mc_set.Nbre_photon
                flux_atm[int(dossier)-1,i] = mc_set.W_atm*mc_set.W_tot/mc_set.Nbre_photon
                
	np.save(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_atms/flux_surf.npy",flux_surf)
	np.save(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_atms/taux_surf.npy",taux_surf)
	np.save(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_atms/flux_atm.npy",flux_atm)

########################### Plotting stuff #################################################

dict_map = {"x" : 0,
           "y" : 1,
           "z" : 2,
           "svf" : 3,
           "svf_lslope" : 4,
           "aspect" : 5,
           "slope" : 6,
           "elevation" : 7,
           "flux" : 8,
           "f_atl" : 9,
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

def plot_MC_symoblique_test(
    ax,
    atm,
    topo_flux,
    dict_map,
    plot_reg,
    ):
        
    #c = (topo_flux[:,dict_map["CN"]] - topo_flux[:,dict_map["f_surf"]]*topo_flux[:,dict_map["flux"]]/(15*15*50))#\
        #/(topo_flux[:,dict_map["f_surf"]]*topo_flux[:,dict_map["flux"]]/(15*15*50))
    cax = ax.scatter(topo_flux[:,dict_map["f_surf"]]*topo_flux[:,dict_map["flux"]]/(15*15*50),\
                     topo_flux[:,dict_map["CN"]], c = dict_atm[atm][1], label = dict_atm[atm][0])#c = c, cmap = 'RdBu_r', vmin = -2, vmax = 2, marker = 'o', s = 15)
    ax.plot([0,40],[0,40],linestyle = 'dashed', color = 'k', linewidth = 1)

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

    ax.set_ylabel('Surface LW radiative flux density $\sigma T^4$ ($W.m^{-2}$)')
    #ax.set_xlabel('Surface LW radiative flux density htrdr ($W.m^{-2}$)')
    ax.set_xlabel('')
    #ax.set_title('Densité de flux radiatif dû à la surface pour MC vs atm gris')
    ax.spines[['left','right', 'top', 'bottom']].set_visible(False)

def plotting_symbolique(
        exp : str,
        res : str,
        dire = "/home/barroisl/Transect_MC_auto/",
        atms = atms):
        
         
        fig,axs = plt.subplots(sharex=True,nrows = 2, ncols = 1, figsize = (20,6))

        for i,atm in enumerate(atms[:-1]): 
            fic_res = dire + f"Output/{exp}_{res}_atms/topo_flux_%s.txt" %atm
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
                plot_reg = plot_reg)
             
            axs[0].set_xticks([])
            axs[1].set_xticks(np.arange(0,45,5))
            axs[1].hist(topo_flux[:,dict_map["f_surf"]]*topo_flux[:,dict_map["flux"]]/(15*15*50), density = True,\
                    alpha = 0.3, color = dict_atm[atm][1],histtype='step', align = 'mid',linewidth = 5)
            axs[1].set_yticks([])
            axs[1].spines[['left','right', 'top', 'bottom']].set_visible(False)
            axs[1].set_xlabel('Surface LW radiative flux density htrdr ($W.m^{-2}$)')
            axs[0].legend()
            
            plt.savefig(dire + "Output/{exp}_{res}_atms/symbolique_hist.jpg")
            
from matplotlib import cm
import matplotlib.colors as colors
	    
def plotting_svf_atm_flux(
		exp : str,
		res : str,
		n_cam : str):

	flux_surf = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_atms/flux_surf.npy")[:200,:]
	taux_surf = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_atms/taux_surf.npy")[:200,:]
	flux_atm = np.load(f"/home/barroisl/Transect_MC_auto/Output/{exp}_{res}_atms/flux_atm.npy")[:200,:]

	topo_params = np.loadtxt(f"/home/barroisl/Transect_MC_auto/camera_tgt/topo_polygon_{exp}_{n_cam}.txt")

	print(topo_params.shape)
	print(taux_surf.shape)
	
	fig,axs = plt.subplots(nrows = 2, ncols = 1, figsize = (12,6), layout = 'constrained')
	atms = ["SMLSATM","SMLWATM","SPOSATM","SPOWATM","STROATM","EMPTATM"]
	for i,atm in enumerate(atms):
	    axs[0].scatter(topo_params[:,0],(1-taux_surf[:,i]), marker = '*', s = 70, c = dict_atm[atm][1],\
		           label = dict_atm[atm][0]) #topo_params[:,-1]
	    axs[0].plot([0.8,1],[0.8,1],linestyle ='dashed', color = 'k')
	    cax = axs[1].scatter(topo_params[:,0],flux_atm[:,i], marker = 'o', s = 70, c = topo_params[:,-1],\
		                 cmap = dict_atm[atm][2],label = dict_atm[atm][0], \
		                 edgecolors='black', linewidths=0.3) #topo_params[:,-1]
	for ax in axs :
	    ax.spines[['left','right', 'top', 'bottom']].set_visible(False)
	    ax.set_xlabel('$SVF_{D&F}$')
	    ax.set_ylabel('$SVF_{htrdr}$')
	    #ax.set_aspect('equal')
	    ax.set_xlim((0.77,1))

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

	#plt.savefig(dire+f"Output/{exp}_{res}_atms/comp_svf.jpg")
