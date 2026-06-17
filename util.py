#!/usr/bin/env python3

import numpy as np
import rasterio.warp


def lect_tif(fic_tif):
	'''
	Lecture de .tif d'un modèle numérique de terrain (DEM = digital
	elevation model, par exemple EEA10).

	[Entrée]
	- fic_tif (str): nom de fichier .tif avec chemin

	[Sortie]
	- a_tif (np.array 2D de float32): orographie en m (lignes de haut
		en bas, selon les latitudes décroissantes)
	- crs (rasterio.crs.CRS): système de référence des coordonnées
		(coordinate reference system)
	- transform (affine.Affine): objet matrice de transformation affine 3x3
		indices (ic,il) -> coordonnées spatiales (x,y)
	'''
	# rasterio: import après (pip_install) python3 -m pip install -U ...
	import rasterio as rio
	with rio.open(fic_tif) as dataset: # (rasterio.io.DatasetReader)
		a_tif = dataset.read(1)
		crs = dataset.crs
		transform = dataset.transform
	return a_tif,crs,transform


def latlon_tif(zone,a_tif,crs,transform):
	'''
	Limitation à une zone et coordonnées d'un modèle numérique de terrain.

	[Entrée]
        - zone (dict): longitudes et latitudes minimum et maximum,
            par exemple dict(lonmin=0.02,lonmax=0.21,latmin=42.79,latmax=42.94)
	- a_tif (np.array 2D de float32): orographie (si issue de lect_tif:
		lignes de haut en bas, selon les latitudes décroissantes) 
        - crs (rasterio.crs.CRS): système de référence des coordonnées (par
            exemple 'EPSG:3035', ou issu de lect_tif)
        - transform (affine.Affine): objet matrice de transformation affine 3x3
            ic,il->x,y (par ex. issu de lect_tif)

	[Sortie]
	- a_tif_zone (np.array 2D de float32): orographie sur la zone
	- a_lat,a_lon (np.array 2D de float64 de même taille que a_tif):
		latitudes et longitudes (en °) de la grille de a_tif_zone
	'''

	# il et ic min et max à partir de "zone"
	icmin,icmax,ilmin,ilmax = indcrs_zone(zone,crs,transform)
	# -> indices de a_tif dans a_tif_zone
	a_tif_zone = a_tif[ilmin:ilmax+1:,icmin:icmax+1:]

	# indices de lignes/colonnes
	nl,nc = a_tif.shape # -> par exemple nl=30000 nc=40000
	a_il1D,a_ic1D = np.arange(nl),np.arange(nc) # indices en 1D
	a_il1D = a_il1D[ilmin:ilmax+1:]
	a_ic1D = a_ic1D[icmin:icmax+1:]
	a_ic,a_il = np.meshgrid(a_ic1D,a_il1D) # indices en 2D

	# (lignes de a_tif de haut en bas selon lat. décroissantes: pas gênant)
	# indices -> coordonnées lat/lon (np.array de float)
	# (dtype par défaut car np.float32 à précision ~1m insuffisant)
	a_lat,a_lon = indcrs_latlon(a_ic,a_il,crs,transform)

	return a_tif_zone,a_lat,a_lon


def tif_interp(fic_tif,lats,lons):
	'''
	Interpolation sur la grille de coordonnées données (pas forcément
	rectangulaire) d'un .tif d'un modèle numérique de terrain
	(DEM = digital elevation model, par exemple EEA10).

	[Entrée]
	- fic_tif (str): nom de fichier .tif avec chemin
	- lats,lons (np.array 2D de float): latitudes et longitudes (en °)
		de la grille sur laquelle on interpole

	[Sortie]
	- z (np.array 2D): orographie sur la grille lats,lons
	'''
	# (d'après $HOME/mesonh/tif_surfex.py)
	# données utiles de fic_tif
	a_tif,crs,transform = lect_tif(fic_tif)
	# données limitées à la zone utile
	zone = dict(lonmin=lons.min(),lonmax=lons.max(),
		latmin=lats.min(),latmax=lats.max())
	z_zone,lat_zone,lon_zone = latlon_tif(zone,a_tif,crs,transform)
	del a_tif # pour moins de mémoire
	# interpolation de grille non rectangulaire (avec LinearNDInterpolator)
	z = chp_interpgri(z_zone,lat_zone,lon_zone,
		lats,lons,rayon=50,marge=3) # rayon > ~3*résolution(EEA10:10m)
	return z


def indcrs_latlon(ic,il,crs,transform,dtype=np.float64):
    '''
    Latitude et longitude des indices de colonne et de ligne, dans un tableau
    avec le géoréférencement donné.

    [IN]
        - ic,il (int ou liste ou np.array de int, les deux de mêmes dimensions):
            indices de colonne et de ligne dans le tableau 2D (nc,nl)
        - crs (rasterio.crs.CRS): système de référence des coordonnées (par
            exemple 'EPSG:3035', ou issu de lect_tif)
        - transform (affine.Affine): objet matrice de transformation affine 3x3
            ic,il->x,y (par ex. issu de lect_tif)
        - dtype: type de données voulu pour les latitudes et longitudes
            (si dtype=np.float32: précision ~1m)

    [OUT]
        - lat,lon (float ou liste ou np.array de dtype): latitude et longitude
            en ° (avec ic,il=0,0 correspondant à lat,lon du centre du 1er pixel)
    '''

    # -> np.array
    if not isinstance(ic,np.ndarray):
        if not isinstance(ic,list): # ic,il scalaires -> listes
            ic,il = [ic],[il]
        ic,il = np.array(ic),np.array(il)
    dims = ic.shape # (=il.shape)

    crslonlat = 'EPSG:4326' # WSG8
    # mis au milieu du pixel (transform: ic,il -> x,y en haut à droite du pixel)
    ic,il = ic + 0.5, il + 0.5
    x,y = transform * (ic,il) # -> 2 np.array de dimensions "dims"
    # x,y -> listes lon et lat
    lon,lat = rasterio.warp.transform(crs,crslonlat,x.flatten(),y.flatten())
    # liste -> np.array 2D de mêmes dimensions que ic et il
    lon = np.array(lon,dtype=dtype).reshape(dims)
    lat = np.array(lat,dtype=dtype).reshape(dims)

    if lon.ndim == 1: # liste
        lon,lat = list(lon),list(lat)
        if len(lon) == 1: # scalaire
            lon,lat = lon[0],lat[0]

    return lat,lon


def latlon_indcrs(lat,lon,crs,transform):
    '''
    Indices (non arrondis) de colonne et de ligne des coordonnées données,
    dans un tableau avec le géoréférencement donné.

    [IN]
        - lat,lon (floats, ou listes de float de même longueur): latitude et
            longitude en °
        - crs (rasterio.crs.CRS): système de référence des coordonnées (par
            exemple 'EPSG:3035', ou issu de lect_tif)
        - transform (affine.Affine): objet matrice de transformation affine 3x3
            ic,il->x,y (par ex. issu de lect_tif)

    [OUT]
        - ic,il (liste de float): indices de colonne et de ligne non arrondis,
            à arrondir pour utilisation comme indices [il,ic] d'un tableau 2D
            (avec lat,lon du centre du 1er pixel correspondant à ic,il=0,0)
    '''

    if not isinstance(lat,list): # lat,lon scalaires -> listes
        lat,lon = [lat],[lon]

    crslonlat = 'EPSG:4326' # WSG8
    l_x,l_y = rasterio.warp.transform(crslonlat,crs,
            lon,lat) # -> liste des x, des y (de même taille)
    l_xy = [(l_x[i],l_y[i]) for i in range(len(l_x))] # liste des (x,y)
    # "~": transformation inverse (cf. https://github.com/rasterio/affine)
    # ~transform: x,y -> ic,il du pixel qui a x,y en haut à droite
    l_icl = [~transform * (x,y) for x,y in l_xy] # liste des (ic,il)
    # -0.5 pour ic,il du pixel qui a x,y au centre
    ic = [icl[0] - 0.5 for icl in l_icl]
    il = [icl[1] - 0.5 for icl in l_icl]

    if len(ic) == 1: # scalaire
        ic,il = ic[0],il[0]

    return ic,il


def indcrs_zone(zone,crs,transform):
    '''
    Indices extrêmes de colonne et de ligne pour couvrir la zone de coordonnées
    donnée, dans un tableau avec le géoréférencement donné.

    [IN]
        - zone (dict): longitudes et latitudes minimum et maximum,
            par exemple dict(lonmin=0.02,lonmax=0.21,latmin=42.79,latmax=42.94)
        - crs (rasterio.crs.CRS): système de référence des coordonnées (par
            exemple 'EPSG:3035', ou issu de lect_tif)
        - transform (affine.Affine): objet matrice de transformation affine 3x3
            ic,il->x,y (par ex. issu de lect_tif)

    [OUT]
        - icmin,icmax (int): indices de colonne minimum et maximum
        - ilmin,ilmax (int): indices de ligne minimum et maximum
    '''

    # les (lon,lat) des coins, en tournant dans le sens anti-trigonométrique
    l_lonlat = [ (zone['lonmin'],zone['latmax']), # en haut à gauche
            (zone['lonmax'],zone['latmax']), # en haut à droite
            (zone['lonmax'],zone['latmin']), # en bas à droite
            (zone['lonmin'],zone['latmin']) ] # en bas à gauche

    l_lon = [lon for lon,lat in l_lonlat]
    l_lat = [lat for lon,lat in l_lonlat]
    l_ic,l_il = latlon_indcrs(l_lat,l_lon,crs,transform)
    # (int est arrondi au-dessous)
    icmin,icmax = int(min(l_ic)) - 1, int(max(l_ic)) + 2
    ilmin,ilmax = int(min(l_il)) - 1, int(max(l_il)) + 2

    return icmin,icmax,ilmin,ilmax


def ijdebfin(lons,lats,l_carte,marge=2,nbpass=2):
    '''
    Indices de tableaux de longitudes et latitudes pour inclure la zone donnée.

    [IN]
        - lons,lats (ma.array ou np.array): latitudes et longitudes 2D (nj,ni)
            (lons variant surtout avec i et croissant en i, de même lats en j)
        - l_carte (liste de float): zone voulue [lonmin, lonmax, latmin, latmax]
        - marge (int): distance minimale (en indices) depuis le bord de la zone
        - nbpass (int): nombre de passages, pour résultat plus proche
            de la zone voulue

    [OUT]
        ideb,ifin,jdeb,jfin (int): indices (depuis 0) débordant de la zone
            de la valeur "marge"
    '''

    nj,ni = lons.shape # (=lats.shape aussi) nj lignes, ni colonnes
    lonmin,lonmax,latmin,latmax = l_carte
    ideb,ifin = 0, ni - 1
    jdeb,jfin = 0, nj - 1

    for ipass in range(nbpass):
        # seulement sur la zone obtenue au passage précédent
        lons2 = lons[jdeb : jfin + 1, :]
        lats2 = lats[:, ideb : ifin + 1]

        latsmin = lats2.min(axis=1)# min sur les ni colonnes -> par ligne (nj,)
        latsmax = lats2.max(axis=1)
        lonsmin = lons2.min(axis=0)# min sur les nj lignes -> par colonne (ni,)
        lonsmax = lons2.max(axis=0)

        # jdeb: 1er j où un lats[j,i] >=latmin
        #jdeb = next(i for i,v in enumerate(latsmax)# (les (indice,valeur))
        #        if v >= latmin) # générateur: calcul arrêté au premier trouvé
        # -> 5 fois plus lent que np.argmax (numpy mieux optimisé?)
        jdeb = np.argmax(latsmax >= latmin) # 1er indice True (=max du booléen)
        # jfin: 1er indice en sens inverse [::-1] où lats[j,i] <=latmax
        jfin = np.argmax(latsmin[::-1] <= latmax)
        jfin = nj - 1 - jfin # -> indice j dans latsmin
        # ideb: 1er i où un lons[j,i] >=lonmin
        ideb = np.argmax(lonsmax >= lonmin)
        # ifin: 1er indice en sens inverse [::-1] où lons[j,i] <=lonmax
        ifin = np.argmax(lonsmin[::-1] <= lonmax)
        ifin = ni - 1 - ifin # -> indice i dans lonsmin

        # à la limite dans la zone -> débordant de la zone de 1
        ideb,ifin = max(ideb - 1, 0), min(ifin + 1, ni - 1)
        jdeb,jfin = max(jdeb - 1, 0), min(jfin + 1, nj - 1)

    # déborder de la zone de "marge" en plus
    ideb,ifin = max(ideb - marge, 0), min(ifin + marge, ni - 1)
    jdeb,jfin = max(jdeb - marge, 0), min(jfin + marge, nj - 1)

    #print("lonmin,lonmax,latmin,latmax",l_carte)
    #print("longitudes ideb <=",lons[:,ideb].max()) # -> <lonmin
    #print("longitudes ifin >=",lons[:,ifin].min()) # -> >lonmax
    #print("latitudes jdeb <=",lats[jdeb,:].max()) # -> <latmin
    #print("latitudes jfin >=",lats[jfin,:].min()) # -> >latmax
    return ideb,ifin,jdeb,jfin


def ind_proche(lats_chp,lons_chp,lats_gri,lons_gri,rayon=50):
    '''
    Recherche de point le plus proche dans une grille lat/lon 2D.

    [IN]
        - lats_chp,lons_chp (np.array): latitudes et longitudes
            (en °) d'une grille horizontale 2D (nj,ni)
            (de dimensions lignes,colonnes ~ latitudes,longitudes)
            - ou bien tableaux 1D, resp. (nj,) et (ni,), pour une grille
                rectangulaire (par exemple, du relief SRTM par srtm_spline)
            - ou bien tableaux 2D (nj,ni) (par exemple, de .nc Meso-NH)
        - lats_gri,lons_gri (np.array 2D, ou float): latitude et longitude
            (en °) des points pour lesquels on cherche le point le plus proche
            dans la grille lats/lons_chp
        - rayon (float): distance de cut-off de la recherche de point le plus
            proche, en m (a priori, au moins 4 fois la résolution de la grille
            lats/lons_chp)

    [OUT]
        - a_i,a_j (np.array de mêmes dimensions que lats/lons_gri, ou float):
            dans la grille lats/lons_chp, les indices de colonne et de ligne
            des points les plus proches des points de lats/lons_gri
    '''

    import pyresample

    if lats_chp.ndim == 1: # mis en 2D si pas déjà 2D (nj lignes, ni colonnes)
        # lons_chp (ni,) et lats_chp (nj,) -> 2 np.array (nj,ni)
        lons_chp,lats_chp = np.meshgrid(lons_chp,lats_chp) # (fait une copie)

    # pour pyresample.geometry il faut des np.array
    if not isinstance(lats_gri,np.ndarray): # lats/lons_gri est un scalaire
        scal = True
        lats_gri = np.array([[lats_gri]])
        lons_gri = np.array([[lons_gri]])
    else:
        scal = False

    #nj,ni = lats_chp.shape # nombre de lignes et colonnes de la grille cible
    nl,nc = lats_gri.shape # nombre de lignes et colonnes des points à chercher

    # recherche de point le plus proche dans la grille de lats/lons_chp
    # (cf. stackoverflow.com/questions/40009528/)
    grid = pyresample.geometry.GridDefinition(lons=lons_chp,lats=lats_chp)
    swath = pyresample.geometry.SwathDefinition(lons=lons_gri,lats=lats_gri)
    _,_,index_array,distance_array = pyresample.kd_tree.get_neighbour_info(
            source_geo_def=grid,target_geo_def=swath,
            radius_of_influence=rayon,neighbours=1) # 1: voisin le + proche
    #print(grid.shape,index_array.shape) # -> (nj,ni) (nl*nc,)
    # index_array: indice de la grille de a_chp (nj,ni) aplatie 1D (nj*ni,)
    #  -> index_array_2d: les 2 indices en 2D de a_chp (nj,ni),
    # pour chaque point de lats/lons_gri: dim. (nl,nc) aplatie 1D (nl*nc,)
    index_array_2d = np.unravel_index(index_array,grid.shape)
    # le point le plus proche de lats/lons_gri[il,ic]
    #  est lats/lons_chp[a_j[il,ic],a_i[il,ic]]
    a_j = index_array_2d[0].reshape((nl,nc))
    a_i = index_array_2d[1].reshape((nl,nc))

    if scal: # lats/lons_gri était un scalaire
        a_i = a_i[0,0]
        a_j = a_j[0,0]

    return a_i,a_j


def chp_interpgri(a_chp,lats_chp,lons_chp,lats_gri,lons_gri,marge=1,rayon=50,
        nbpass=0):
    '''
    Interpolation d'un champ horizontal en lat/lon sur la grille lat/lon donnée
    (pas forcément rectangulaire).

    [IN]
        - a_chp (np.array): tableau 2D (nj,ni) des valeurs du champ horizontal,
            de dimensions lignes,colonnes ~ latitudes,longitudes
            (par exemple relief SRTM par chp_srtm)
        - lats_chp,lons_chp (np.array): latitudes et longitudes (°) de la
            grille de a_chp (croissantes selon les indices si nbpass!=0)
            - ou bien tableaux 1D, resp. (nj,) et (ni,), pour une grille
                rectangulaire (par exemple, du relief SRTM par srtm_spline)
            - ou bien tableaux 2D (nj,ni) (par exemple, de .nc Meso-NH)
        - lats_gri,lons_gri (np.array): tableaux 2D des latitudes et longitudes
            (en °) de la grille sur laquelle on interpole (par exemple, le sol
            vu en chaque point d'une image de caméra, avec en 1e dimension les
            lignes allant de haut en bas)
        - marge (int): marge sur les indices de lats_gri et lons_gri, pour
            l'interpolation (défaut 1, suffisant pour une grille lats_lons_gri
            bien régulière)
        - rayon (float): (si nbpass=0) distance de cut-off de la recherche de
            point le plus proche dans la grille de a_chp, en m (a priori,
            au moins 3 fois sa résolution)
        - nbpass (int): nombre de passages, pour recherche de la zone
            d'interpolation par ijdebfin (défaut 0: zone d'interpolation par
            recherche de point le plus proche dans la grille de a_chp)

    [OUT]
        - a_gri (np.array): tableau 2D des valeurs du champ aux points de la
            grille sur laquelle on interpole, de mêmes dimensions que lats_gri
            et lons_gri (et par exemple avec les lignes allant de même de haut
            en bas)
    '''

    # lats,lons mis en mêmes dimensions que a_chp (nj lignes, ni colonnes)
    if lats_chp.ndim == 1: # si pas déjà 2D (nj,ni)
        # lons_chp (ni,) et lats_chp (nj,) -> 2 np.array (nj,ni)
        lons_chp,lats_chp = np.meshgrid(lons_chp,lats_chp) # (fait une copie)

    nj,ni = lats_chp.shape # nombre de lignes et colonnes du champ a_chp
    nl,nc = lats_gri.shape # nb de lignes/col de la grille sur laq. on interpole
    a_gri = np.empty((nl,nc))

    if nbpass == 0:
        # le point le plus proche de lats/lons_gri[il,ic]
        #  est lats/lons_chp[a_j[il,ic],a_i[il,ic]]
        a_i,a_j = ind_proche(lats_chp,lons_chp,lats_gri,lons_gri,rayon=rayon)

    from scipy.interpolate import LinearNDInterpolator

    # pour chaque point [il,ic] de la grille lats/lons_gri vers où on interpole
    for ic in range(nc):
        if nc > 1 and (ic % 100 == 0 or ic == nc - 1):
            print(f"colonne {ic}") # pour suivre la progression
        for il in range(nl):
            lat,lon = lats_gri[il,ic],lons_gri[il,ic]
            if np.isnan(lat) or np.isnan(lon):
                a_gri[il,ic] = np.nan
                continue # passer au point suivant

            # [jdeb:jfin+1,ideb:ifin+1] zone d'interp. du champ pour ce point
            if nbpass == 0:
                ideb = max(a_i[il,ic] - marge, 0)
                ifin = min(a_i[il,ic] + marge, ni - 1)
                jdeb = max(a_j[il,ic] - marge, 0)
                jfin = min(a_j[il,ic] + marge, nj - 1)
            else:
                ideb,ifin,jdeb,jfin = ijdebfin(lons_chp,lats_chp,
                        [lon,lon,lat,lat],marge=marge,nbpass=nbpass)

            # valeurs du champ entre lesquelles interpoler: 3 tableaux 1D
            x1D = lats_chp[jdeb : jfin + 1, ideb : ifin + 1].flatten()
            y1D = lons_chp[jdeb : jfin + 1, ideb : ifin + 1].flatten()
            z1D = a_chp[jdeb : jfin + 1, ideb : ifin + 1].flatten()
            #points = np.array(list(zip(x1D,y1D))) # équivalent, pas plus rapide
            points = np.empty((len(x1D),2))
            points[:,0] = x1D
            points[:,1] = y1D

            # interpolation de a_chp vers lats/lons_gri
            #from scipy.interpolate import SmoothBivariateSpline
            # s=0 pour interpolation, ordre 1 (bilinéaire) comme srtm_spline
            #spl_chp = SmoothBivariateSpline(x1D,y1D,z1D,kx=1,ky=1,s=0)
            #a_gri[il,ic] = spl_chp.ev(lat,lon)# .ev: interpolation (NaN -> NaN)
            # -> parfois valeur aberrante: plutôt LinearNDInterpolator
            #  (cf. gist.github.com/ev-br/8544371b40f414b7eaf3fe6217209bff)
            interp_chp = LinearNDInterpolator(points,z1D)
            a_gri[il,ic] = interp_chp(lat,lon)

    return a_gri


