from Functions_lw_analysis.py import *

############################ Base function ###################

def plot_pos_abs(
    fig,
    ax,
    index,
    ds_topo : xr.Dataset,
    ds : xr.Dataset,
    data : np.ndarray,
    min_ : int,
    max_ : int,
    cmap : str,
    label : str,
    title : str,
    xlim,
    ylim,
    option : str = "NER",
    ):
    
    bounds = np.linspace(min_,max_,10)
    norm_t = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    
    #Plotting topo contour
    ds_topo.zs.plot.contourf(ax = ax, add_colorbar = False, cmap = 'terrain', \
                                   levels = np.arange(0,2000,100), alpha = 0.5)
        
    cax = ax.scatter(data[:,0],data[:,1],c = data[:,2]-,cmap = cmap,\
                        vmin = min_, vmax = max_, s=5)
    
    cax = ax.scatter(data[:,0],data[:,1],c = data[:,2],cmap = cmap,\
                        vmin = min_, vmax = max_, s=5)
    
    #Flux realisation colorbar
    #fig.colorbar(mappable = cax, ax = ax, norm = norm_t, orientation = 'vertical',location='right',\
    #             cmap = cmap, extend = 'both',label = label, shrink = 0.5)
    
    #Arrow of caméra target
    start_coord = (data[0,3],data[0,4])
    end_coord = (data[0,5],data[0,6])
    ax.add_patch(FancyArrowPatch(start_coord, end_coord,
                                 arrowstyle='->',
                                 edgecolor='r',
                                 facecolor='r',
                                 mutation_scale=10))
    
    #Circle of quantile of horizontal distribution of absorption event by surface
    thetas = np.linspace(0,2*np.pi,1000)
    Rs = [data[0,7],data[0,8],data[0,9]]
    cs = ['lightblue','dodgerblue','darkblue']
    quantiles = [90,50,10]
    for i in range(len(Rs)) :
        xs = Rs[i]*np.cos(thetas) + data[0,3] 
        ys = Rs[i]*np.sin(thetas) + data[0,4] 
        ax.plot(xs, ys, color = cs[i], label = f"Quantil {quantiles[i]} $\simeq$ {int(Rs[i])} m")
       
    xs = 2e3*np.cos(thetas) + data[0,3] 
    ys = 2e3*np.sin(thetas) + data[0,4] 
    ax.plot(xs, ys, color = 'r', linestyle = 'dashed', label = "Lamare & al 2020 = 2000 m")
    
    #Fancy camera position
    ax.scatter(data[:,3],data[:,4],c = 'dodgerblue', marker = '*')

    ax.legend()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title)
    

############### Plotting interactive for different atms ############

def plot_pos_abs_func(index):
    # Find the closest altitude center from the table to the input z
    closest_index = min(d_f_EMPTATM.item().keys(), key=lambda k: abs(k - index))
    
    data_EMPTATM = d_f_EMPTATM.item().get(closest_index)
    data_SMLSATM = d_f_SMLSATM.item().get(closest_index)
    data_SMLWATM = d_f_SMLSATM.item().get(closest_index)
    data_TUXUATM = d_f_TUXUATM.item().get(closest_index)
    
    
    #ds_topo
    ds_topo = xr.open_dataset("/home/barroisl/Transect_MC_auto/topographie/lavey_topo_params_30.nc")
        
    fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (15,15), layout = 'constrained')
    
    plot_pos_abs(
        fig = fig,
        ax = axs[0,1],
        index = closest_index,
        ds_topo = ds_topo,
        #ds_SW = ds_SW.sel(time='2025-01-01T12:00:00.000000000')
        data = data_EMPTATM,
        min_ = 200,
        max_ = 300,
        cmap = 'hot',
        label = 'flux $W.m^{-2}$',
        title = dict_atm["EMPTATM"][0],
        xlim = (275000,295000),
        ylim = (4.965e6,4.990e6)
        )
    
    plot_pos_abs(
        fig = fig,
        ax = axs[0,0],
        index = closest_index,
        ds_topo = ds_topo,
        #ds_SW = ds_SW.sel(time='2025-01-01T12:00:00.000000000')
        data = data_SMLSATM,
        min_ = 200,
        max_ = 300,
        cmap = 'hot',
        label = 'flux $W.m^{-2}$',
        title = dict_atm["SMLSATM"][0],
        xlim = (275000,295000),
        ylim = (4.965e6,4.990e6)
        )
    
    plot_pos_abs(
        fig = fig,
        ax = axs[1,0],
        index = closest_index,
        ds_topo = ds_topo,
        #ds_SW = ds_SW.sel(time='2025-01-01T12:00:00.000000000')
        data = data_EMPTATM,
        min_ = 200,
        max_ = 300,
        cmap = 'hot',
        label = 'flux $W.m^{-2}$',
        title = dict_atm["SMLWATM"][0],
        xlim = (275000,295000),
        ylim = (4.965e6,4.990e6)
        )
    
    plot_pos_abs(
        fig = fig,
        ax = axs[1,1],
        index = closest_index,
        ds_topo = ds_topo,
        #ds_SW = ds_SW.sel(time='2025-01-01T12:00:00.000000000')
        data = data_TUXUATM,
        min_ = 200,
        max_ = 300,
        cmap = 'hot',
        label = 'flux $W.m^{-2}$',
        title = dict_atm["TUXUATM"][0],
        xlim = (275000,295000),
        ylim = (4.965e6,4.990e6)
        )
    
    # Colorbar indépendante du plot
    vmin = 200
    vmax = 300
    pas = 5
    cmap = cm.hot  
    bounds = np.arange(vmin,vmax+pas, pas)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)   
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])                     
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', boundaries = bounds,
                        fraction=0.046, pad=0.04)   
    cbar.set_label('Luminence énergétique ($W.m^{-2}$)')     
    
    plt.show()
    
    
"""
atm = "EMPTATM"

d_f_EMPTATM =np.load(f"/home/barroisl/Transect_MC_auto/Output/lavey_30_100_atms_65/{atm}/pos_dict_{atm}.npy",allow_pickle=True)

atm = "SMLSATM"

d_f_SMLSATM =np.load(f"/home/barroisl/Transect_MC_auto/Output/lavey_30_100_atms_65/{atm}/pos_dict_{atm}.npy",allow_pickle=True)

atm = "SMLWATM"

d_f_SMLWATM =np.load(f"/home/barroisl/Transect_MC_auto/Output/lavey_30_100_atms_65/{atm}/pos_dict_{atm}.npy",allow_pickle=True)

atm = "TUXUATM"

d_f_TUXUATM =np.load(f"/home/barroisl/Transect_MC_auto/Output/lavey_30_100_atms_65/{atm}/pos_dict_{atm}.npy",allow_pickle=True)

index = FloatSlider(min=1, max=100, step=1, value=0.0,
                     description='ligne', continuous_update=False)

interact(plot_pos_abs_func, index=index)
"""
    
    
################ Plotting interactive for camera grid (guiers) ##############


def plot_pos_abs_func_ij(i,j):
    # Find the closest camera index from the table to the inputs i,j
    closest_index = (i-1)*12 +j
    data = d_f.item().get(closest_index)
    
    #ds_topo
    ds_topo = xr.open_dataset("/home/barroisl/Transect_MC_auto/topographie/topo_params_250m.nc")
    #ds_SW
    ds_SW = xr.open_dataset("/home/barroisl/Downscall_output/GIF/Chartreuse/ds_SW_shadow.netcdf")
    
    fig, ax = plt.subplots(figsize = (9,9), layout = 'constrained')
    plot_pos_abs(
        fig = fig,
        ax = ax,
        index = closest_index,
        ds_topo = ds_topo,
        #ds_SW = ds_SW.sel(time='2025-01-01T12:00:00.000000000')
        data = data,
        min_ = 280,
        max_ = 400,
        cmap = 'hot',
        label = 'flux $W.m^{-2}$',
        title = ' ',
        xlim = (234000,258000),
        ylim = (5020000,5038000)
        )
    plt.show()
    
    
"""
d_f =np.load("/home/barroisl/Transect_MC_auto/Output/guiers_250/pos_dict.npy",allow_pickle=True)

i = FloatSlider(min=1, max=12, step=1, value=0.0,
                     description='ligne', continuous_update=False)
j = FloatSlider(min=1, max=12, step=1, value=0.0,
                     description='colonne', continuous_update=False)

interact(plot_pos_abs_func_ij,i=i,j=j)
"""

############################ Plot interactive symbolique ##################

def altit_T(dem, T0, z0, lapse_rate = -0.0065):
    
    return T0+lapse_rate*(dem-z0)

def plot_pos_abs_symbolique(
    fig,
    ax,
    index,
    ds_topo : xr.Dataset,
    ds : xr.Dataset,
    data : np.ndarray,
    min_ : int,
    max_ : int,
    cmap : str,
    label : str,
    title : str,
    xlim,
    ylim,
    option : str = "NER",
    ):
    
    bounds = np.linspace(min_,max_,10)
    norm_t = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    
    #Plotting topo contour
    ds_topo.zs.plot.contourf(ax = ax, add_colorbar = False, cmap = 'terrain', \
                                   levels = np.arange(0,2000,100), alpha = 0.5)
    
    #ds_topo['ts'] = (['y','x'], altit_T(ds_topo['zs'].values, T0 = 273, z0 = 1300, lapse_rate = -6.5e-3))
    
    Symb_LW_abs = from_position_to_luminence_symbolique(
        abs_nd = data[1:,:3],
        ds = ds)

    #Plotting htrdr diagnostics
    if option == "RER" :
        cax = ax.scatter(data[1:,0],data[1:,1],c = Symb_LW_abs,cmap = cmap,\
                        vmin = min_, vmax = max_, s=5)
    
    if option == "NER" :
        
        Symb_LW_cam = from_position_to_luminence_symbolique(
            abs_nd = data[0,:3],
            ds = ds)
        
        cax = ax.scatter(data[1:,0],data[1:,1],c = Symb_LW_abs-Symb_LW_cam,cmap = cmap,\
                        vmin = min_, vmax = max_, s=5)
    
    #Flux realisation colorbar
    #fig.colorbar(mappable = cax, ax = ax, norm = norm_t, orientation = 'vertical',location='right',\
    #             cmap = cmap, extend = 'both',label = label, shrink = 0.5)
    
    #Arrow of caméra target
    start_coord = (data[0,3],data[0,4])
    end_coord = (data[0,5],data[0,6])
    ax.add_patch(FancyArrowPatch(start_coord, end_coord,
                                 arrowstyle='->',
                                 edgecolor='r',
                                 facecolor='r',
                                 mutation_scale=10))
    
    #Circle of quantile of horizontal distribution of absorption event by surface
    thetas = np.linspace(0,2*np.pi,1000)
    Rs = [data[0,7],data[0,8],data[0,9]]
    cs = ['lightblue','dodgerblue','darkblue']
    quantiles = [90,50,10]
    for i in range(len(Rs)) :
        xs = Rs[i]*np.cos(thetas) + data[0,3] 
        ys = Rs[i]*np.sin(thetas) + data[0,4] 
        ax.plot(xs, ys, color = cs[i], label = f"Quantil {quantiles[i]} $\simeq$ {int(Rs[i])} m")
       
    xs = 2e3*np.cos(thetas) + data[0,3] 
    ys = 2e3*np.sin(thetas) + data[0,4] 
    ax.plot(xs, ys, color = 'r', linestyle = 'dashed', label = "Lamare & al 2020 = 2000 m")
    
    #Fancy camera position
    ax.scatter(data[:,3],data[:,4],c = 'dodgerblue', marker = '*')

    ax.legend()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title)
    
def plot_pos_abs_func_symb(index):
    
    # Find the closest altitude center from the table to the input z
    closest_index = min(d_f_.item().keys(), key=lambda k: abs(k - index))
    
    data = d_f_.item().get(closest_index)
    
    #ds_topo
    ds_topo = xr.open_dataset("/home/barroisl/Transect_MC_auto/topographie/topo_params_250m.nc")
    
    ds_chartreuse = xr.open_dataset("/home/barroisl/Transect_MC_auto/s2m_simu/chartreuse_thomas.nc")
        
    fig, ax = plt.subplots( figsize = (9,9), layout = 'constrained')
    
    plot_pos_abs_symbolique(
        fig = fig,
        ax = ax,
        index = closest_index,
        ds_topo = ds_topo,
        ds = ds_chartreuse,
        data = data,
        min_ = -25,
        max_ = 25,
        cmap = 'RdBu_r',
        label = '$W.m^{-2}$',
        title = 'Symbolique',
        xlim = (234000,258000),
        ylim = (5020000,5038000),
        #xlim = (275000,295000),
        #ylim = (4.965e6,4.990e6),
        option = 'NER'
        )
    
    # Colorbar indépendante du plot
    vmin = -25
    vmax = 25
    pas = 5
    cmap = cm.RdBu_r 
    bounds = np.arange(vmin,vmax+pas, pas)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)   
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])                     
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', boundaries = bounds,
                        fraction=0.046, pad=0.04)   
    cbar.set_label('Luminence énergétique ($W.m^{-2}$)')     
    
    plt.show()
    
    
"""
d_f_ =np.load("/home/barroisl/Transect_MC_auto/Output/guiers_250/pos_dict.npy",allow_pickle=True)

index = FloatSlider(min=1, max=144, step=1, value=0.0,
                     description='ligne', continuous_update=False)

interact(plot_pos_abs_func_symb, index = index)
"""

"""
atm = 'SMLSATM'
d_f_ =np.load(f"/home/barroisl/Transect_MC_auto/Output/lavey_30_100_atms_65/{atm}/pos_dict_{atm}.npy",allow_pickle=True)

index = FloatSlider(min=1, max=100, step=1, value=0.0,
                     description='ligne', continuous_update=False)
                     
interact(plot_pos_abs_func_symb, index = index)
"""


