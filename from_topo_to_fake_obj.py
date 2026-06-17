from Functions import *

###################### Crop a topo params xr.dataset to feed the obj_mtls_dem ########################

def from_tif_to_topo_params(
        fic_tif : str,
        fic_ds_cropped : str,
        fic_topo_param : str,
        lon_min,
        lon_max,
        lat_min,
        lat_max): #,
        #T0 = 273,
        #z0 = 1300,
        #lapse_rate = -6.5e-3):
        
        ds_latlon = riox.open_rasterio(fic_tif)
        
        ds_latlon_crop = select_spatial_extent(
            ds = ds_latlon,
            x_min = lon_min,
            x_max = lon_max,
            y_min = lat_min,
            y_max = lat_max,
            x_dim = "x",
            y_dim = "y")
            
        xx,yy = np.meshgrid(ds_latlon_crop.x.values,ds_latlon_crop.y.values)
        x_32632,y_32632 = convert_epsg_pts(xx,yy, epsg_src=4326, epsg_tgt=32632)
        
        ds_32632 = xr.Dataset(
            {'zs': (["y", "x"], ds_latlon_crop.values[0])},
            #'ts': (["y", "x"], altit_T(ds_latlon_crop.values[0], T0 = T0, z0 = z0, lapse_rate = lapse_rate))
            #},
            coords={"x": x_32632[0], "y": y_32632[:,0]},
        )
    
        ds_32632.to_netcdf(fic_ds_cropped)
        
        compute_dem_param(path_to_ds_s2m = fic_ds_cropped,
                      path_to_topo_params = fic_topo_param)

####################### Create .obj & .mtls from .tif #################################################

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
    zs,slope = ds.zs.values, ds.slope.values
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
                d_ts[ts_arr].append( (j,i,slope[j,i]) )
                
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
        
        #if len(ts_arr_snow)  > 0 :
        
        if 5 == 5 :
                
            mrumtl = 'snow'
            nom_mat = mrumtl + str(ts_arr).replace('.','_') # ex.'snow238_1'
            lignes_obj.append(f'usemtl air:{nom_mat}\n')

	    # Boucles sur les facettes de température ts_arr d'indice inférieur à frac_neige*len(d_ts)
	    # dans d_ts.sorted() --> Toutes les facettes recouvertes de neige
	    #for j,i in d_ts[ts_arr][indices_snow]:# points avec la Ts arrondie ts_arr
            for j,i,s in d_ts[ts_arr] : #ts_arr_snow :
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
        
        """
        
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
            for j,i,s in ts_arr_sand :
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
        """
        
    lignes_mtls.append('air none\n')
    
    print("lignes_mtls OK : Nbre %d ; " %len(lignes_mtls), lignes_mtls[0])

    #On change les lignes de l\'.obj pour faire apparaître une fraction neigeuse
       
    with open(fic_obj,'w') as fid: # w: créer, défaut: encoding='UTF-8'
        fid.writelines(lignes_obj)
    with open(fic_mtls,'w') as fid:
        fid.writelines(lignes_mtls)
        
######################### Core function ######################################

if len(sys.argv) != 7:
	print("Nbre d'argument non valide %d : " %len(sys.argv))
	print("Usage: python3 from_topo_to_fake_obj.py ecrins res lon_min lon_max lat_min lat_max")
	sys.exit(1)

from_tif_to_topo_params(
        fic_tif = "/home/barroisl/Transect_MC_auto/tifs/all_alps.tif",
        fic_ds_cropped = "/home/barroisl/Transect_MC_auto/ds_fake_T_real_topo/%s_cropped.nc" %sys.argv[1],
        fic_topo_param = f"/home/barroisl/Transect_MC_auto/topographie/{sys.argv[1]}_topo_params_{sys.argv[2]}.nc",
        lon_min = float(sys.argv[3])-0.1,
        lon_max = float(sys.argv[4])+0.1,
        lat_min = float(sys.argv[5])-0.1,
        lat_max = float(sys.argv[6])+0.1)
        
print("Topo params from .tif without temperature information OK")

obj_mtls_dem_vhs_distributed(
        fic_ds = f"/home/barroisl/Transect_MC_auto/topographie/{sys.argv[1]}_topo_params_{sys.argv[2]}.nc",
        fic_obj = "/home/barroisl/edstar/Simus/models/%s_65.obj" %sys.argv[1],
        fic_mtls = "/home/barroisl/edstar/Simus/materials/%s_65.mtls" %sys.argv[1],
        T0=273,
        z0=1300,
        lapse_rate = 6.5e-3)
        
print(".obj & .mtls from topo params lapse rate OK")
        
obj_mtls_dem_vhs_distributed(
        fic_ds = f"/home/barroisl/Transect_MC_auto/topographie/{sys.argv[1]}_topo_params_{sys.argv[2]}.nc",
        fic_obj = "/home/barroisl/edstar/Simus/models/%s_00.obj" %sys.argv[1],
        fic_mtls = "/home/barroisl/edstar/Simus/materials/%s_00.mtls" %sys.argv[1],
        T0=273,
        z0=1300,
        lapse_rate = 0.0)
        
print(".obj & .mtls from topo params no lapse rate OK")

"""

x1,y1 = convert_epsg_pts(sys.argv[3],sys.argv[5], epsg_src=4326, epsg_tgt=32632)
x2,y2 = convert_epsg_pts(sys.argv[4],sys.argv[5], epsg_src=4326, epsg_tgt=32632)
x3,y3 = convert_epsg_pts(sys.argv[4],sys.argv[6], epsg_src=4326, epsg_tgt=32632)
x4,y4 = convert_epsg_pts(sys.argv[3],sys.argv[6], epsg_src=4326, epsg_tgt=32632)

print("x1,x2,x3,x4",x1,x2,x3,x4)
print("y1,y2,y3,y4",y1,y2,y3,y4)

creating_polygon(
    path_to_poly = "/home/barroisl/Transect_MC_auto/polygon/%s.geojson" %sys.argv[1],
    x1 = round(x1),
    x2 = round(x2),
    x3 = round(x3),
    x4 = round(x4),
    y1 = round(y1),
    y2 = round(y2),
    y3 = round(y3),
    y4 = round(y4))
    
print(".geojson OK")

from_nc_to_polygon_cam_tgt(
    polygon = "/home/barroisl/Transect_MC_auto/polygon/%s.geojson" %sys.argv[1],
    ds_s2m_path = f"/home/barroisl/Transect_MC_auto/topographie/{sys.argv[1]}_topo_params_{sys.argv[2]}.nc",
    fic_res = "/home/barroisl/Transect_MC_auto/camera_tgt/polygon_%s.txt" %sys.argv[1],
    fic_ds_topo = f"/home/barroisl/Transect_MC_auto/topographie/{sys.argv[1]}_topo_params_{sys.argv[2]}.nc",
    fic_res_topo = "/home/barroisl/Transect_MC_auto/camera_tgt/topo_polygon_%s.txt" %sys.argv[1])

print("cam_tgt OK")

"""


