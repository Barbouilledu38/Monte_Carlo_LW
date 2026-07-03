#!/usr/bin/env python3

import os

import numpy as np
import netCDF4 as nc

import util


def lect_ncMNH(fic,l_var):
	'''
	Met dans un dictionnaire les variables indiquées d'un fichier netCDF
	*SEG*.nc (issu de MESONH) *SEG*dg.nc (de DIAG) ou PGD*.nc (de PREP_PGD).
	
	[Entrée]
	- fic (str): nom du fichier avec le chemin
	- l_var (liste de str): les noms de variable à lire

	[Sortie]
	- d_var (dict): {
		nom de variable (str): (np.array) les valeurs,
			avec masqués et Fill_Value mis à NaN
			correspondant
		}
	'''
	# (d'après read_var de $SRC_MESONH/src/LIB/Python/read_MNHfile.py)

	d_var = {}
	variables = nc.Dataset(fic,'r').variables

	for var in l_var:
		variable = variables[var]
		# [...]: netCDF4._netCDF4.Variable -> np.ma.array
		d_var[var] = variable[...] # par ex. (1, 74, 402, 502)

		# (NHALO=)1 élément enlevé au début / à la fin des dim spatiales
		#  -> par ex. (1, 72, 400, 500)
		# noms_dim: tuple, par ex.: ('time', 'level', 'nj_u', 'ni_u')
		noms_dim = variable.dimensions
		for idim in range(len(noms_dim)): # indice de la dimension
			if len(noms_dim) != 0 and noms_dim[idim] in (
				'level','level_w',
				'ni','ni_u','ni_v','nj','nj_u','nj_v'):
				# enlever 0 et i_fin dans la dim d'indice idim
				i_fin = d_var[var].shape[idim] - 1
				d_var[var] = np.delete(d_var[var],
					[0,i_fin],axis=idim)

		# np.ma.array -> np.array, avec masqué -> NaN
		d_var[var] = np.where(~d_var[var].mask, d_var[var], np.nan)
		# Fill_Value (dans read_BACKUPfile de read_MNHfile.py) -> NaN
		d_var[var] = np.where( (d_var[var] != -99999.0)
			& (d_var[var] != 999.0), d_var[var], np.nan)

		# dimension 'time' de lgr 1 enlevée -> par ex. (72, 400, 500)
		d_var[var] = np.squeeze(d_var[var])

	if 'time' in variables: # pas pour PGD
		# (ici MESONH ou DIAG: .calendar existe dans fic)
		time,t_units,t_calendar = time_attr(fic)
		# time: np.ma.array de dimensions "()" (pas (1,))
		# -> par num2date, 'date' est datetime (pas un tableau)
		d_var['time'] = time
		d_var['date'] = nc.num2date(time,
			# par exemple 'seconds since 2022-01-14 00:00:00 +0:00'
			units=t_units,
			calendar=t_calendar)

	return d_var


def time_attr(fic):
	'''
	Lit la valeur et les attributs utiles de la variable netCDF "time"
	dans un fichier netCDF qui la contient (par exemple *SEG*.nc issu
	de MESONH, ou fichier netCDF du type voulu pour les2htcp).

	[Entrée]
	- fic (str): nom du fichier avec le chemin

	[Sortie]
	- time (float, ou en secondes int): temps (par exemple 75600)
	- t_units (str): indiquant unité et date de départ (par exemple
		"seconds since 2022-01-14 00:00:00 +0:00"),
	- t_calendar (str): calendrier (par exemple "standard")
	'''
	variable_time = nc.Dataset(fic,'r').variables['time']
	# netCDF4._netCDF4.Variable
	# -> [0]: np.ma.array de dimensions "()", [...]: de dimensions (1,)
	time = float(variable_time[...]) # -> float
	t_units = variable_time.units
	if t_units[:7] == "seconds": # -> arrondi à la seconde près
		time = round(time) # (int)
	t_calendar = ("standard"
		if 'calendar' not in variable_time.ncattrs()
		else variable_time.calendar)
	return time,t_units,t_calendar


def ss_nan(zs,dist=1):
	'''
	Enlève les NaN du np.array 2D "zs" par moyenne sur distance
	"dist" autour (dist=1 suffit pour des NaN isolés).
	'''
	# (fait au départ pour pas de NaN dans ZS, et donc dans alt et alt_w
	# comme c'est nécessaire pour RectBivariateSpline de coupe oblique_proj)
	nb_nan = np.isnan(zs).sum()
	if nb_nan == 0: # si pas de NaN, pas besoin de le faire
		return
	jmax,imax = zs.shape
	for j in range(jmax):
		for i in range(imax):
			if np.isnan(zs[j,i]):
				zs[j,i] = np.nanmean(zs[ # moyenne hors NaN
					max(0,j - dist):min(jmax,j + dist + 1),
					max(0,i - dist):min(imax,i + dist + 1)])
				print("i,j",i,j,": NaN ->",zs[j,i])
	nb_nan2 = np.isnan(zs).sum()
	print(nb_nan,"NaN ->",nb_nan2,"NaN")
	if nb_nan2 != 0: # doit être ramené à 0
		print("ss_nan: augmenter dist=")
	return


def xyz_les(fic,l_carte=[],fic_z=None):
	'''
	Orographie d'un fichier netCDF issu de MESONH ou de PREP_PGD au
	centre des cases horizontales (position des variables Meso-NH
	'ni' et 'nj'), limitée éventuellement à une zone, et ses coordonnées
	x et y adaptées à ncles (entrée de les2htcp pour -c de
	htrdr-atmosphere).

	[Entrée]
	- fic (str): nom de fichier *SEG*.nc issu de MESONH ou PGD*.nc issu
		de PREP_PGD (avec le chemin)
	- l_carte (liste de float): [lonmin,lonmax,latmin,latmax],
		zone voulue dans fic (défaut []: pas de limitation à une zone)
	- fic_z (str): nom de fichier .npy (issu de np.save) de l'orographie
		des centres des cases déjà calculée avec lcarte=[] à partir
		d'un fichier .tif (par exemple EEA10), ou nom de fichier .tif
		pour en tirer directement l'orographie (défaut None: orographie
		de fic, où il peut y avoir un filtrage &NAM_ZSFILTER)

	[Sortie]
	- x,y (np.array 1D): coordonnées x et y des centres des cases (en m)
	- zs_xy (np.array 2D): altitude de l'orographie sur grille x,y (en m)
	- ijdebfin (quadruplet de int): (ideb,ifin,jdeb,jfin) indices extrêmes
		de la zone voulue (+ marge de 2 indices) dans les dimensions
		ni et nj de variables Meso-NH (comme 'latitude' et 'longitude')
		qui comportent ces dimensions
	'''

	# x,y et lats/lons au milieu des cases, avec 0 au bord de la maille
	x,y,lons,lats = xy_les(fic,centre=True)# 50m -> (500,) (400,) (400,500)
	dx,dy = x[1] - x[0], y[1] - y[0] # 50m -> 50

	# ZS: surface_altitude (m) (nj, ni)
	d_var = lect_ncMNH(fic,['ZS'])
	zs_xy = d_var['ZS'] # 2D (nj,ni), 50m -> (400,500)

	# i,j min/max à l'intérieur de l_carte (tout si l_carte=[])
	ijdebfin = ( # indices dans lons et lats débordant de l_carte (de 2)
		util.ijdebfin(lons,lats,l_carte,marge=2) if len(l_carte) != 0
		else (0, len(x) - 1, 0, len(y) - 1) ) # extrêmes dans x,y
	ideb,ifin,jdeb,jfin = ijdebfin
	# grille horizontale limitée à l_carte
	x = x[ideb : ifin + 1]
	y = y[jdeb : jfin + 1]
	zs_xy = zs_xy[jdeb : jfin + 1, ideb : ifin + 1]
	lats = lats[jdeb : jfin + 1, ideb : ifin + 1]
	lons = lons[jdeb : jfin + 1, ideb : ifin + 1]

	# pour ncles: coordonnées x,y centres des cases avec bords partant de 0
	# par ex. x,y=25,25 au bord inférieur gauche de la maille 50m
	x = x - x[0] + dx / 2
	y = y - y[0] + dy / 2

	if fic_z is None: # vérifier zs
		# pas de NaN dans zs (ni donc dans alt dans ncles)
		# par moyenne sur dist=1 autour (NaN isolés)
		ss_nan(zs_xy,dist=1) # -> il y avait 0 NaN
	elif fic_z[-4:] != '.tif': # orographie lue dans un .npy
		zs_xy = np.load(fic_z)
		# .npy avec lcarte=[] -> limiter à l_carte
		zs_xy = zs_xy[jdeb : jfin + 1, ideb : ifin + 1]
	else: # orographie refaite par interpolation du .tif sur des coordonnées
		# lat/lon sur la grille de x,y voulue (centres de cases)
		# (utilise LinearNDInterpolator)
		zs_xy = util.tif_interp(fic_z,lats,lons)

	return x,y,zs_xy,ijdebfin


def xyzbord_les(fic,fic_z=None):
	'''
	Orographie d'un fichier netCDF issu de MESONH ou de PREP_PGD au
	bord des cases horizontales (position des variables Meso-NH 'ni_u' et
	'nj_v'), et ses coordonnées x et y adaptées à ncles sur tout le domaine
	Meso-NH (entrée de les2htcp pour -c de htrdr-atmosphere).

	[Entrée]
	- fic (str): nom de fichier *SEG*.nc issu de MESONH ou PGD*.nc issu
		de PREP_PGD (avec le chemin)
	- fic_z (str): nom de fichier .npy (issu de np.save) de l'orographie
		des centres des cases déjà calculée avec lcarte=[] à partir
		d'un fichier .tif (par exemple EEA10), ou nom de fichier .tif
		pour en tirer directement l'orographie (défaut None: orographie
		de fic, où il peut y avoir un filtrage &NAM_ZSFILTER)

	[Sortie]
	- x,y (np.array 1D): coordonnées x et y des bords des cases (en m),
		partant de 0 au bord de la maille
	- zs_xy (np.array 2D): altitude de l'orographie sur grille x,y (en m)
	'''

	# x,y et lats/lons au bord des cases, avec 0 au bord de la maille
	x,y,lons,lats = xy_les(fic,centre=False)# 50m -> (501,) (401,) (401,501)
	dx,dy = x[1] - x[0], y[1] - y[0] # 50m -> 50

	if fic_z is None:
		variables = nc.Dataset(fic,'r').variables
		# ZS: surface_altitude (m) (nj, ni)
		# pour interpoler vers x et y, centres autour des bords voulus:
		# tout garder (1er et dernier indices) -> grille 50m: (402,502)
		# (masked_invalid: masqué -> NaN)
		zs = np.array(np.ma.masked_invalid(variables['ZS'][...]))
		# vérifier zs: pas de NaN dans zs (ni donc dans alt dans ncles)
		# par moyenne sur dist=1 autour (NaN isolés)
		ss_nan(zs,dist=1) # -> il y avait 0 NaN
		# ni=x+dx/2 nj=y+dy/2 correspondant à x,y qui partent de 0
		# mais ici doivent être autour de x,y: 1er ni,nj en plus
		ni = np.insert(x + dx / 2, 0, -dx / 2)# -dx/2 mis avant indice 0
		nj = np.insert(y + dy / 2, 0, -dy / 2)
		# -> nj: (402,), ni: (502,)

		# interpolation d'une grille rectangulaire
		from scipy.interpolate import RectBivariateSpline
		# s=0 (défaut) pour interpolation ensuite par .ev
		# kx,ky: d'ordre 1 = bilinéaire (cf. oblique_proj de
		#  $SRC_MESONH/src/LIB/Python/misc_functions.py)
		# -> objet BivariateSpline
		spl_zs = RectBivariateSpline(nj,ni,zs,kx=1,ky=1)
		# mis en 2D: x (501,) et y (401,) -> 2 np.array (401,501)
		x2D,y2D = np.meshgrid(x,y)
		zs_xy = spl_zs.ev(y2D,x2D)

	elif fic_z[-4:] != '.tif': # orographie lue dans un .npy
		zs_xy = np.load(fic_z)

	else: # orographie par interpolation du .tif sur des coordonnées lat/lon
		# sur la grille de x,y voulue (bords de cases)
		# (utilise LinearNDInterpolator)
		zs_xy = util.tif_interp(fic_z,lats,lons)

	return x,y,zs_xy


def xy_les(fic,centre=True):
	'''
	Coordonnées x, y et latitudes/longitudes dans un fichier netCDF issu
	de MESONH ou de PREP_PGD, au centre ou au bord des cases, avec les
	x et y adaptés à ncles (entrée de les2htcp pour -c de htrdr-atmosphere).

	[Entrée]
	- fic (str): nom de fichier *SEG*.nc issu de MESONH ou PGD*.nc issu
		de PREP_PGD (avec le chemin)
	- centre (bool): x,y des centres des cases (défaut), sinon
		des bords des cases
	
	[Sortie]
	- x,y (np.array 1D): coordonnées x et y des centres ou des bords
		des cases (en m), en partant de 0 au bord de la maille
	- lons,lats (np.array 2D): longitudes et latitudes correpondantes (en °)
	'''

	# ni: projection_x_coordinate (m) (ni)
	# nj: projection_y_coordinate (m) (nj)
	# XHAT: projection_x_coordinate (m) (ni_u)
	# YHAT: projection_y_coordinate (m) (nj_v)
	#  ni_u=XHAT=ni-dx/2, nj_v=YHAT=nj-dy/2, ni_v=ni, nj_u=nj
	# latitude: latitude (degrees_north) (nj, ni)
	# longitude: longitude (degrees_east) (nj, ni)
	# latitude_f: latitude_at_f_location (degrees_north) (nj_v, ni_u)
	#  = latitude(ni-dx/2,nj-dy/2)
	#  ~(50m: différence ~10^-5°~1m) latitude_v = latitude(ni,nj-dy/2)
	#  < latitude(ni,nj)
	# longitude_f: longitude_at_f_location (degrees_east) (nj_v, ni_u)
	#  = longitude(ni-dx/2,nj-dy/2)
	#  ~(50m: différence ~10^-5°~1m) longitude_u = longitude(ni-dx/2,nj)
	#  < longitude(ni,nj)

	variables = nc.Dataset(fic,'r').variables
	# ici partir de XHAT,YHAT car ni,nj masqués dans PGD
	# [...]: netCDF4._netCDF4.Variable -> np.ma.array
	# np.array: -> np.array sans masque
	xhat = np.array(variables['XHAT'][...]) # par ex. 50m -> (502,)
	yhat = np.array(variables['YHAT'][...]) # (402,)

	if centre:
		# on veut x,y sur les centres de cases, en nj,ni
		# -> enlever le 1er (NHALO=1) indice et le dernier
		# (comme dans lect_ncMNH)
		xhat = xhat[1:-1] # (500,)
		# -> xhat sur 25km par 50 de xhat.min()=62375 à xhat.min()+24950
		yhat = yhat[1:-1] # (400,)
		# -> yhat sur 20km par 50 de yhat.min()=39875 à yhat.min()+19950
		dx,dy = xhat[1] - xhat[0], yhat[1] - yhat[0] # 50m -> 50
		# pour ncles: x,y centres des cases avec bords partant de 0
		# par ex. x,y=25,25 au bord inférieur gauche de la maille 50m
		x = xhat - xhat[0] + dx / 2
		y = yhat - yhat[0] + dy / 2

		# lon/lat des centres de cases, en ni,nj
		lons = np.array(variables['longitude'][...]) # np.ma.->np.array
		lats = np.array(variables['latitude'][...])
		# comme pour xhat,yhat: enlever le 1er indice et le dernier
		lons = lons[1:-1,1:-1] # -> 50m: (400,500)
		lats = lats[1:-1,1:-1]

	else:
		# on veut x,y sur les bords de cases, en ni_u,nj_v
		# -> enlever le 1er (NHALO=1) indice, garder le dernier
		xhat = xhat[1:] # (501,)
		yhat = yhat[1:] # (401,)
		# pour ncles: x,y bords des cases partant de 0
		x = xhat - xhat[0]
		y = yhat - yhat[0]

		# lon/lat sur les bords de cases, en ni_u,nj_v
		lons = np.array(variables['longitude_f'][...])# np.ma.->np.array
		lats = np.array(variables['latitude_f'][...])
		# comme pour xhat,yhat: enlever le 1er indice, garder le dernier
		lons = lons[1:,1:] # -> 50m: (401,501)
		lats = lats[1:,1:]

	return x,y,lons,lats


def ij_imbr(fic):
	'''
	Indices extrêmes, dans le modèle père Meso-NH, des bords des cases
	qui ne sont pas nécessaires pour le relief triangulé imbriqué (.obj)
	car la case est aussi décrite dans le modèle fils imbriqué.

	[Entrée]
	- fic (str): nom de fichier *SEG*.nc issu de MESONH (avec le chemin)
		du modèle fils (ou sans modèle fils, nom du fichier
		PGD*.nc correspondant issu de PREP_PGD)

	[Sortie]
	- imin,imax,jmin,jmax (int): indices dans la grille des bords des
		cases horizontales (position des variables Meso-NH 'ni_u'
		et 'nj_v') du modèle père
	'''
	# domaine père = domaine 1, domaine fils = domaine 2

	# ni: projection_x_coordinate (m) (ni)
	# nj: projection_y_coordinate (m) (nj)
	#  (ni_u=XHAT=ni-dx/2, nj_v=YHAT=nj-dy/2, ni_v=ni, nj_u=nj)
	# XOR: "Horizontal position of this mesh relative to its father"
	# YOR: "Vertical position of this mesh relative to its father"
	# DXRATIO: "Resolution ratio between this mesh and its father in
	#  x-direction"
	# DYRATIO: "Resolution ratio between this mesh and its father in
	#  y-direction"
	# =IXOR IYOR IDXRATIO IDYRATIO dans &NAM_INIFILE (PREP_PGD pour créer 2)
	# IXOR/IYOR: pt de départ dans 1 (à l'intérieur de 2) du domaine pour 2
	# IDXRATIO,IDYRATIO: 5 pour résolution divisée par 5 pour 2
	d_var = lect_ncMNH(fic,['ni_u','nj_v','XOR','YOR','DXRATIO','DYRATIO'])
	# np.array de int de dim '()' -> int
	iorig,jorig = int(d_var['XOR']),int(d_var['YOR']) # -> 250 160
	rx,ry = int(d_var['DXRATIO']),int(d_var['DYRATIO']) # -> 5 5

	# taille de la grille des bords des cases: 1 de plus que les centres
	nxhat,nyhat = len(d_var['ni_u']) + 1, len(d_var['nj_v']) + 1 #-> 501 401
	# taille dans 1 du domaine pour 2 (grille des centres des cases)
	# =IXSIZE IYSIZE dans &NAM_INIFILE (PREP_PGD pour créer 2):
	xsize = int((nxhat - 1) / rx) # (int/int=float -> int) -> 100
	ysize = int((nyhat - 1) / ry) # -> 80

	#fic1 = fic.replace('.2.','.1.') # par exemple: nom 1 à partir de 2
	#d_var1 = lect_ncMNH(fic1,['latitude_f','longitude_f'])
	#d_var2 = lect_ncMNH(fic,['latitude_f','longitude_f'])
	#latf1,latf2 = d_var1['latitude_f'],d_var2['latitude_f']
	#lonf1,lonf2 = d_var1['longitude_f'],d_var2['longitude_f']
	#print(lonf1[jorig-1,iorig-1],lonf2[0,0])
	#print(latf1[jorig-1,iorig-1],latf2[0,0])
	# -> sur la grille des bords des cases x,y = XHAT,YHAT (nj_v,ni_u),
	#  point X/YHAT1[jorig-1,iorig-1] = point X/YHAT2[0,0]
	# indices min non nécessaires dans 1 car case aussi décrite dans 2:
	imin,jmin = iorig - 1, jorig - 1

	#d = 1 # >=1
	#print(lonf1[jmin+ysize - d, imin+xsize - d],
	#	lonf2[nyhat-1 - d*ry, nxhat-1 - d*ry])
	#print(latf1[jmin+ysize - d, imin+xsize - d],
	#	latf2[nyhat-1 - d*ry, nxhat-1 - d*rx])
	# -> point X/YHAT1[jmin+ysize - d, imin+xsize - d]
	#     = point X/YHAT2[nyhat-1 - d*ry, nxhat-1 - d*rx]:
	# indices max non nécessaires dans 1 car case aussi décrite dans 2:
	imax,jmax = imin + xsize - 1, jmin + ysize - 1

	return imin,imax,jmin,jmax


def obj_mtls(l_fic,fic_obj,fic_mtls,l_fic_dg=None,l_fic_z=None,mrumtl='snow',
	extn=None):
	'''
	Crée le .obj (relief triangulé indiquant les matériaux) pour -g et
	le .mtls (matériaux avec Ts de chacun, format htrdr-materials) pour -M
	de htrdr-atmosphere, à partir de fichiers netCDF issus de modèles
	imbriqués Meso-NH (et éventuellement de fichiers issus de DIAG),
	dans les coordonnées x,y de la maille père des mailles imbriquées
	(à adapter ensuite par adapt_obj au .nc LES d'entrée de les2htcp si
	l_fic[0] n'a pas le même coin sud-ouest).

	[Entrée]
	- l_fic (n-uplet de str): noms de fichier *SEG*.nc issus de MESONH
		(avec le chemin) pour les modèles imbriqués (par exemple 1,2,3
		= 250m 50m 10m), ou (possible si pas le premier fichier)
		sans modèle avec un nom de fichier PGD*.nc issu de PREP_PGD
	- fic_obj (str): nom de fichier à créer en format htrdr-obj
	- fic_mtls (str): nom de fichier à créer en format htrdr-materials
	- l_fic_dg (n-uplet de str ou None de même longueur que l_fic, ou None):
		noms des fichiers *SEG*dg.nc issus de DIAG où lire 'TS' (défaut
		None: lire 'TSRAD' dans le fichier correspondant de l_fic)
		(fichier de l_fic_dg non utilisé s'il correspond dans l_fic à
		un PGD*.nc: alors Ts par interpolation du modèle père)
	- l_fic_z: (n-uplet de str de même longueur que l_fic) noms des
		fichiers .npy (issus de np.save) de l'orographie des bords des
		cases déjà calculée à partir d'un fichier .tif (par exemple
		EEA10), ou nom de fichier .tif pour en tirer directement
		l'orographie (défaut None: orographie interpolée à partir de
		fic, où il peut y avoir un filtrage &NAM_ZSFILTER)
	- mrumtl (str): ici un seul fichier de réflectivité (matériau réel)
		materials/{mrumtl}.mrumtl (format mrumtl), pour tous les
		matériaux (fictifs) de fic_mtls qui ont chacun une température
	- extn (None ou float ou n-uplet): si défaut None pas d'extension du
		premier l_fic, sinon extension définie par (lim seul si float)
		- lim (float): distance en km du centre au bord de l'extension
		- alt0 (float, absent: 0): altitude en m à l'infini
		- dimin (float, absent: 1/4 dist.max lim-bord et <=20):
			distance en km de diminution par 2 de l'écart à altitude
			= alt0 et à T_alt0 = T_alt0 moyen du bord du domaine
		- ndimin (int, absent: ~1 par 4 mailles): nombre de points
			pour atteindre la diminution par 2
	'''
	
	nfic = len(l_fic)
	if l_fic_dg is None:
		l_fic_dg = (None,) * nfic
	if not isinstance(l_fic_z,(tuple,list)): # -> toujours n-uplet de str
		l_fic_z = (l_fic_z,) * nfic
		
	# quels PGD dans l_fic: nécessaire en premier car ci-dessous
	#  xc,yc selon PGD ific et ific+1, et décalage de x,y selon ific-1
	b_pgd = {} # {indice ific dans l_fic: (bool) est un PGD}
	for ific in range(nfic):
		# PGD si pas de ZTOP haut du modèle (PGD est seulement le sol)
		b_pgd[ific] = 'ZTOP' not in nc.Dataset(
			l_fic[ific],'r').variables

	# calcul des coordonnées
	# centres des cases: xc-cte=ni(=ni_v), yc-cte=nj(=nj_u)
	# bords des cases: x-cte=XHAT=ni_u=ni-dx/2, y-cte=YHAT=nj_v=nj-dy/2
	# maille du 1er modèle (1er de l_fic): x,y part de 0, xc,yc de dx/2 dy/2
	x,y = {},{} # {indice ific dans l_fic: bords 1D des cases horizontales}
	zs = {} # {indice ific dans l_fic: altit. 2D des bords des cases horiz.}
	xc,yc = {},{} # {indice ific dans l_fic: centres 1D des cases horiz.}
	mask = {} # {indice ific dans l_fic: (bool 2D) indices non nécessaires}

	for ific in range(nfic):

		#print("xyz", ific + 1)
		# points au bord des cases, pour relief triangulé
		# -> par exemple sur maille 50m: x (501,) y (401,), zs (401,501)
		x[ific],y[ific],zs[ific] = xyzbord_les(
			l_fic[ific],fic_z=l_fic_z[ific])
		mask[ific] = np.full( (len(y[ific]),len(x[ific])),
			False) # np.array de bool, aucun True pour l'instant
		if ific > 0: # imbriqué depuis le père ific-1
			# indices min/max non nécessaires dans ific-1
			# car case décrite dans ific
			imin,imax,jmin,jmax = ij_imbr(l_fic[ific])
			# True pour les indices non nécessaires dans ific-1
			mask[ific-1][jmin : jmax + 1, imin : imax + 1] = True

			# x,y[ific] mis dans le système x,y[ific-1] du père:
			# x,y[ific] [0,0] était =0,0 et c'est le même point
			#  que x,y[ific-1] [j1min,i1min]
			x[ific] = x[ific] + x[ific-1][imin]
			y[ific] = y[ific] + y[ific-1][jmin]

		# points au milieu des cases, pour Ts:
		# si b_pgd[i], Ts sera interpolé sur xc,yc de i à partir de i-1
		# -> xc,yc[ific] nécessaires si b_pgd[ific] ou [ific+1]
		if b_pgd[ific] or b_pgd[min(ific + 1, nfic - 1)]:
			# -> par exemple sur maille 50m: xc (500,), yc (400,)
			xc[ific],yc[ific],_,_ = xyz_les(# (tout le domaine ific)
				l_fic[ific],l_carte=[],fic_z=None)
			if ific > 0: # imbriqué depuis le père ific-1
				# xc=x+dx/2 yc=y+dy/2 -> même décalage que x
				xc[ific] = xc[ific] + x[ific-1][imin]
				yc[ific] = yc[ific] + y[ific-1][jmin]

	# calcul de Ts, soit issu de DIAG:
	# TS: "TS (K)" (K) (nj, ni)
	#  = (doc Surfex: Diagnostic model output fields) surface temperature
	# soit issu de MESONH:
	# TSRAD: "radiative surface temperature" (K) (nj, ni)
	# et TS-TSRAD sur grille 50m: min,max = -0.911 0.006 (points isolés)
	ts = {} # {indice ific dans l_fic: Ts 2D des centres des cases horiz.}

	for ific in range(nfic):

		if not b_pgd[ific]: # est un *SEG*.nc issu d'un modèle MESONH
			#print("Ts", ific + 1, "SEG")
			# -> par exemple sur maille 50m: ts (400,500)
			if l_fic_dg[ific] is None:
				ts[ific] = lect_ncMNH(l_fic[ific],['TSRAD']
					)['TSRAD']
			else:
				ts[ific] = lect_ncMNH(l_fic_dg[ific],['TS']
					)['TS']

		else: # est PGD sans Ts: interpoler ts[ific-1] sur xc,yc[ific]
			#print("Ts", ific + 1, "PGD")
			# interpolation d'une grille rectangulaire
			from scipy.interpolate import RectBivariateSpline
			# s=0 (défaut) pour interpolation ensuite par .ev
			# kx,ky: d'ordre 1 = bilinéaire (cf. oblique_proj de
			#  $SRC_MESONH/src/LIB/Python/misc_functions.py)
			# -> objet BivariateSpline
			spl_ts = RectBivariateSpline(
				yc[ific-1],xc[ific-1],ts[ific-1], # PGD: ific>0
				kx=1,ky=1)
			# mis en 2D: xc (500,), yc (400,) -> 2 de dim. (400,500)
			xc2D,yc2D = np.meshgrid(xc[ific],yc[ific])
			# extrapolé au bord (xc,yc ific-1 moins étendu que ific)
			ts[ific] = spl_ts.ev(yc2D,xc2D)

	if extn is not None:
		# bord du domaine l_fic[0] -> grille d'extension à lim km vers
		#  alt=alt0, écart diminué 1/2 sur dimin km en ndimin points

		if not isinstance(extn,(list,tuple)):
			extn = (extn,) # float -> toujours n-uplet (qui a len)
		# (200,0,10,10) -> kargs=dict(lim=200,alt0=0,dimin=10,ndimin=10)
		kargs = {}
		KEYS = ['lim','alt0','dimin','ndimin']
		for ik in range(len(KEYS)):
			if ik <= len(extn) - 1:
				kargs[KEYS[ik]] = extn[ik]

		# x[-1],y[-1],zs[-1],ts[-1]: grilles 2D [jl,ib]
		# (ib indice sur bord de domaine x[0],y[0], jl vers l'extérieur)
		x[-1],y[-1],zs[-1],ts[-1] = xyzt_ext(
			x[0],y[0],zs[0],ts[0],**kargs)
		mask[-1] = np.full(x[-1].shape,False) # extension: rien masqué

	# liste de str (terminés par \n) pour créer .obj/.mtls par writelines
	lignes_obj = []
	lignes_mtls = []

	# .obj 1e partie: liste de tous les v (vertex/sommet = point de bord
	# de case) définis par leurs x,y,z

	# numérotation des sommets
	num = 0 # numéro du sommet dans le .obj
	d_num = {} # { (indice ific dans l_fic, j, i de ific) : n° de sommet}
	range_nfic = range(nfic) if extn is None else range(-1,nfic)
	for ific in range_nfic:
		#print("v", ific + 1)
		nli,ncol = zs[ific].shape # nb de lignes (en j), de colonnes (i)
		for j in range(nli):
			for i in range(ncol):
				# la case i,j utilisera aussi les sommets
				# (en x,y) i+1,j i,j+1 i+1,j+1: sommet i,j utile
				# si case i,j i-1,j i,j-1 ou i-1,j-1 utilisée
				if (not mask[ific][j,i] or not mask[ific][j,i-1]
					or not mask[ific][j-1,i]
					or not mask[ific][j-1,i-1]):
					# x,y,z arrondi à 3 décimales
					# 2D pour x[-1],y[-1]
					x1 = round(x[ific][i] if x[ific].ndim
						== 1 else x[ific][j,i], 3)
					y1 = round(y[ific][j] if y[ific].ndim
						== 1 else y[ific][j,i], 3)
					zs1 = round(zs[ific][j,i],3)
					ligne = f'v {x1} {y1} {zs1}\n'
					lignes_obj.append(ligne)
					num += 1 # numéro commençant à 1
					d_num[(ific,j,i)] = num
	lignes_obj.append('\n') # ligne vide

	# Ts sur chaque milieu de case
	d_ts = {} # {Ts arrondi (= 1 matériau): liste des (indice ific, j,i)}
	for ific in range_nfic:
		nli,ncol = zs[ific].shape # nb de lignes (en j), de colonnes (i)
		# -1: autant de y utiles que de coord. j de Ts (50m: 401->400)
		for j in range(nli - 1):
			# -1: autant de x utiles que de coord i de Ts (501->500)
			for i in range(ncol - 1):
				if not mask[ific][j,i]:
					# Ts en K, arrondi à 0.1
					ts_arr = round(ts[ific][j,i],1)
					if ts_arr not in d_ts:
						d_ts[ts_arr] = []
					d_ts[ts_arr].append( (ific,j,i) )
	#print(len(d_ts),min(list(d_ts)),max(list(d_ts))) # -> 404 236.8 277.9

	# .obj 2e partie: pour chaque matériau, liste des f (faces) définis
	# par leurs v
	# et .mtls: liste des noms de matériaux avec leur Ts

	cwd = os.getcwd() # répertoire actuel
	#print("f")
	for ts_arr in sorted(d_ts): # liste triée des Ts arrondies
		nom_mat = mrumtl + str(ts_arr).replace('.','_') # ex.'snow238_1'
		lignes_obj.append(f'usemtl air:{nom_mat}\n')
		for ific,j,i in d_ts[ts_arr]:# points avec la Ts arrondie ts_arr
			# 2 triangles en ht à dr de (j,i), sens aig. montre (cf.
			#  htrdr-Atmosphere-Starter-Pack-0.8.0/models/plane.obj)
			lignes_obj.append( # en haut à droite du point j,i
				f'f {d_num[(ific,j,i)]} ' # j,i
				f'{d_num[(ific,j+1,i)]} ' # + haut que j,i
				f'{d_num[(ific,j,i+1)]}\n') # + à droite que j,i
			lignes_obj.append( # même case, plus en haut à droite
				f'f {d_num[(ific,j,i+1)]} ' # + à droite que j,i
				f'{d_num[(ific,j+1,i)]} ' # + haut que j,i
				f'{d_num[(ific,j+1,i+1)]}\n')# + haut + à droite
		lignes_obj.append('\n') # ligne vide
		lignes_mtls.append(f'{nom_mat} '
			f'{cwd}/materials/{mrumtl}.mrumtl {ts_arr}\n')
	lignes_mtls.append('air none\n')

	with open(fic_obj,'w') as fid: # w: créer, défaut: encoding='UTF-8'
		fid.writelines(lignes_obj)
	with open(fic_mtls,'w') as fid:
		fid.writelines(lignes_mtls)
	return
	

def xyzt_ext(x,y,z,t,lim,alt0=0,dimin=None,ndimin=None):
	'''
	Extension de la triangulation .obj de obj_mtls au-delà du bord d'un
	domaine, par exponentielle en facteur de l'altitude et de Tsurf
	jusqu'à la distance limite donnée.

	[Entrée]
	- x,y (np.array 1D): bords en m (ncol) et (nli,) des cases horizontales
	- z (np.array 2D): altitude en m (nli,ncol) des bords des cases horiz.
	- t (np.array 2D): Tsurf en K (nli-1,ncol-1) des centres des cases
		horizontales (à partir de dx/2,dy/2 si x,y part de 0)
	- lim (float): distance en km du centre au bord de l'extension
	- alt0 (float): altitude en m à l'infini
	- dimin (float): distance en km de diminution par 2 de l'écart
		à altitude = alt0 et à T_alt0 = T_alt0 moyen du bord du domaine
		(défaut None: 1/4 de dist. max entre cercle lim et bord et <=20)
	- ndimin (int): nombre de points pour atteindre la diminution par 2
		(défaut None: ~1 point par 4 mailles)

	[Sortie]
	- xe,ye,ze (np.array 2D): points des bords des cases de l'extension,
		(nl,nb) avec nb points au bord du domaine et nl points avant lim
	- te (np.array 2D): Tsurf (nl-1,nb-1) des centres de ces cases
	'''
	# variation de température dans la troposphère OACI (K/m)
	therm_L = -0.0065

	# centre et distance min/max au centre de la grille rectangulaire x,y
	x_cen,y_cen = (x[-1] + x[0]) / 2, (y[-1] + y[0]) / 2
	dmin = min( (x[-1] - x[0]) / 2, (y[-1] - y[0]) / 2 )
	dmax = np.sqrt( ((x[-1] - x[0]) / 2) ** 2 + ((y[-1] - y[0]) / 2) ** 2 )
	#print(dmin,dmax) # -> 50000 ~90139 (domaine 250m = 100x150km)
	# le cercle limite de l'extension doit être au-delà du bord de la grille
	lim = max(lim,int(dmax / 1000 + 1)) # lim > dmax en km arrondi au-dessus
	if dimin is None:
		# 20km, ou 1/4 de distance max entre cercle lim et bord si <20km
		dimin = min(20, (lim - dmin / 1000) / 4)
	if ndimin is None:
		# sur les 1ers dimin km, points à résoln ~ 4*résoln de la grille
		resol = (x[1] - x[0]) / 1000 # résolution de la grille en km
		ndimin = int(dimin / (4 * resol))

	nli,ncol = z.shape # nb de lignes (en j), de colonnes (en i) -> 401,601
	# ts (400,600): ts[j,i] est dx/2,dy/2 en haut à dr. de x[i],y[j],z[j,i]

	# Bord du domaine, sens des aiguilles d'une montre depuis en bas à dr.

	xb,yb,zb = [],[],[] # x,y,z des segments du bord du domaine x,y
	tb,ztb = [],[] # Tsurf et z: milieu de case à l'intérieur à dr. de xb,yb

	# bord gauche, vers le haut: i=0, j croissant sans le dernier
	xb += [x[0]] * (nli - 1)
	yb += list(y[:-1])
	zb += list(z[:-1,0])
	# milieu de case à l'intérieur à dr en suivant le bord = en haut à dr
	tb +=  list(t[:,0])
	# à partir des 4 points de bord [j,i],[j+1,i],[j,i+1],[j+1,i+1]
	ztb += list((z[:-1,0] + z[1:,0] + z[:-1,1] + z[1:,1]) / 4)

	# bord haut, vers la droite: j=-1, i croissant sans le dernier
	xb += list(x[:-1])
	yb += [y[-1]] * (ncol - 1)
	zb += list(z[-1,:-1])
	# milieu de case à l'intérieur à dr en suivant le bord = en bas à dr
	tb += list(t[-1,:])
	# à partir des 4 points de bord [j,i],[j-1,i],[j,i+1],[j-1,i+1]
	ztb += list((z[-1,:-1] + z[-2,:-1] + z[-1,1:] + z[-2,1:]) / 4)

	# bord droit, vers le bas: i=-1, j décroissant sans le dernier
	xb += [x[-1]] * (nli - 1)
	yb += list(y[-1:0:-1])
	zb += list(z[-1:0:-1,-1])
	# milieu de case à l'intérieur à dr en suivant le bord = en bas à gauche
	tb += list(t[-1::-1,-1])
	# à partir des 4 points de bord [j,i],[j-1,i],[j,i-1],[j-1,i-1]
	ztb += list((z[-1:0:-1,-1] + z[-2::-1,-1]
		+ z[-1:0:-1,-2] + z[-2::-1,-2]) / 4)

	# bord bas, vers la gauche: j=0, i décroissant sans le dernier
	xb += list(x[-1:0:-1])
	yb += [y[0]] * (ncol - 1)
	zb += list(z[0,-1:0:-1])
	# milieu de case à l'intérieur à dr en suivant le bord = en haut à g.
	tb += list(t[0,-1::-1])
	# à partir des 4 points de bord [j,i],[j+1,i],[j,i-1],[j+1,i-1]
	ztb += list((z[0,-1:0:-1] + z[1,-1:0:-1]
		+ z[0,-2::-1] + z[1,-2::-1]) / 4)

	# dernier point = premier point
	xb += [x[0]]
	yb += [y[0]]
	zb += [z[0,0]]
	# milieu de case à l'intérieur à dr en suivant le bord = en haut à dr
	tb += [t[0,0]]
	# à partir des 4 points de bord [j,i],[j+1,i],[j,i+1],[j+1,i+1]
	ztb += [(z[0,0] + z[1,0] + z[0,1] + z[1,1]) / 4]
	#print(np.array(zb).mean(),np.array(ztb).mean()) # -> ~848 et 850m

	# nb: nombre de points du bord du domaine (bords de case: indices ib)
	nb = len(xb) # xb,yb,zb,tb,ztb tous (nb,) (-> 2001=400+600+400+600+1)
	# T à alt0 selon atmosphère standard, moyennée sur le bord
	tb0_moy = np.array([
		tb[ib] - therm_L * (ztb[ib] - alt0) # L<0: tb0>tb si ztb>alt0
		for ib in range(nb) ]).mean()

	# Extension du domaine

	# nl: nombre de points vers l'extérieur, jusqu'à lim km (indices jl)
	# (d distance au centre) longueur du segment selon jl = d/djl(d)
	#  ~ largeur du segment selon ib à d[jl] = angle*d -> d/d[0]=exp(rl*jl)
	# (d[0] = bord du domaine) d[0]+dimin=d[ndimin]=exp(rl*ndimin)*d[0] ->rl
	#  et lim=d[nl]=exp(rl*nl)*d[0] ->nl: en général plus grand à d[0]=dmin
	rl = np.log((dmin + dimin * 1000) / dmin) / ndimin # à ib où d[0]=dmin
	nl = int(np.log(lim * 1000 / dmin) / rl + 1) # arrondi au-dessus
	nl = nl + 1 # dernier indice en partant de 0 -> nombre de points

	xe,ye,ze = np.empty((nl,nb)),np.empty((nl,nb)),np.empty((nl,nb))
	te = np.empty((nl - 1, nb - 1))
	for ib in range(nb): # indice ib en suivant le bord du domaine
 		# distance du centre au bord du domaine
		db = np.sqrt( (xb[ib] - x_cen) ** 2 + (yb[ib] - y_cen) ** 2 )
		# d/d[jl=0]=exp(rl*jl), d(xe,ye) de db à lim pour jl=0 à nl-1:
		#  lim/db=d[nl-1]/d[0]=exp(rl*(nl-1))
		rl = np.log(lim * 1000 / db) / (nl - 1) # à cet ib (nl connu)
		# diminution exponentielle de z-alt0 selon la distance au bord:
		#  (ze[jl]-alt0)/(zb-alt0)=exp(-rd*(d[jl]-d[0]))
		# 1/2=exp(-rd*(d[ndimin]-d[0])) avec d/d[0]=exp(rl*jl) -> rd
		rd = np.log(2) / (db * np.exp(rl * ndimin) - db)
		if ib != nb - 1: # pour te
			# T à alt0 mer selon atm. standard, au bord du domaine
			tb0 = tb[ib] - therm_L * (ztb[ib] - alt0) # L<0
		for jl in range(nl):
			# xe,ye: jl fait varier la distance au centre
			d = db * np.exp(rl * jl)
			xe[jl,ib] = x_cen + (xb[ib] - x_cen) * d / db
			ye[jl,ib] = y_cen + (yb[ib] - y_cen) * d / db
			# ze en exp
			ze[jl,ib] = alt0 + (zb[ib] - alt0) * np.exp(
				-rd * (d - db))
			if ib != nb - 1 and jl != nl - 1: # te selon tb et ztb
				# même diminution de te0-tb0_moy que de z-alt0
				te0 = tb0_moy + (tb0 - tb0_moy) * np.exp(
					-rd * (d - db))
				# et même diminution de zte-alt0
				zte = alt0 + (ztb[ib] - alt0) * np.exp(
					-rd * (d - db))
				te[jl,ib] = te0 + therm_L * (zte - alt0)

	return xe,ye,ze,te


