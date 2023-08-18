__all__ = [

    'fit_trace',
    'plot_result'

]

from jax import jit

def fit_trace(RV, phases, CCF, phase_start, phase_end, inclination, transit, vsys_range=[-40, 10], vorb_range=[100, 400], max_lw=20, drv=2, cpu_cores=20, num_warmup=200, num_samples=600):

    import numpy as np
    import pdb
    import tayph.system_parameters as sp

    from jax.scipy.optimize import minimize
    
    # We need to import numpyro first, though we use it last
    import numpyro
    from numpyro.infer import MCMC, NUTS
    from numpyro import distributions as dist
    from jax.random import PRNGKey, split
    from numpyro.infer import init_to_sample
    
    # Set the number of cores on your machine for parallelism:
    cpu_cores = cpu_cores
    numpyro.set_host_device_count(cpu_cores)
    from jax import numpy as jnp
    from jax import jit, config
    config.update('jax_enable_x64', True)

    import arviz
    #from corner import corner, overplot_lines, overplot_points
    
    # Jax has its own scipy module which uses autodiffed gradients
    #from scipy.optimize import minimize
    #from jax.scipy.optimize import minimize as jminimize


    phase_bins = phases
    #phase_bins[phase_bins>0.5] -=1 # makes sure the axis is continuous

    RV_without_correlations = RV[::drv] #- 0.5*drv
    CCF_stacked_without_correlations = CCF[:, ::drv]

    # This would be for plotting, not needed
    #vmin = np.nanmean(CCF_stacked_without_correlations)-5*np.nanstd(CCF_stacked_without_correlations)
    #vmax = np.nanmean(CCF_stacked_without_correlations)+5*np.nanstd(CCF_stacked_without_correlations)
    
    CCF_binned_without_NaNs = -jnp.array(CCF_stacked_without_correlations[:, np.argwhere(np.isnan(CCF_stacked_without_correlations[0])==False)])
    
    # binned = without correlations and without NaNs
    CCF_binned = CCF_binned_without_NaNs[:,:,0]
    
    RV_binned = jnp.array(RV_without_correlations[np.argwhere(np.isnan(CCF_stacked_without_correlations[0])==False)])
    RV_binned = np.reshape(RV_binned, (len(RV_binned),))
    
    CCF_binned_err = jnp.tile(jnp.nanstd(CCF_binned, axis=1), (len(RV_binned),1)).T

    # make it two-dimensional
    RV2D = jnp.tile(RV_binned,(len(CCF_binned),1))
    #RV2D_with_NaNs = jnp.tile(RV_with_NaNs,(len(CCF_binned_with_NaNs),1))
    #pdb.set_trace()

    phi_binned_real = phase_bins[np.isfinite(CCF_binned[:,0])]
    RV2D_binned_real = RV2D[np.isfinite(CCF_binned[:,0])]
    CCF_binned_real = CCF_binned[np.isfinite(CCF_binned[:,0])]
    CCF_binned_err_real = CCF_binned_err[np.isfinite(CCF_binned[:,0])]

    npro_data = CCF_binned_real
    npro_err = CCF_binned_err_real
    
    f_start = -jnp.min(CCF_binned_real) * 1e4
    print(f_start)
    transit_binned = transit - 1.
    
    def numpyro_model():
        # flux_beg,flux_end,lw,v_start,v_end,C
        flux_beg = numpyro.sample(r'$F_{\rm begin}$', dist.Uniform(low=f_start/1000, high=f_start*1000))
        flux_end = numpyro.sample(r'$F_{\rm end}$', dist.Uniform(low=f_start/1000, high=f_start*1000))
        lw  = numpyro.sample(r'$\sigma_{\rm w}$', dist.Uniform(low=1, high=max_lw)) 
        vorb1 = numpyro.sample(r'$v_{\rm orb, 1}$', dist.Uniform(low=vorb_range[0], high=vorb_range[1]))
        vorb2 = numpyro.sample(r'$v_{\rm orb, 2}$', dist.Uniform(low=vorb_range[0], high=vorb_range[1]))
        vsys = numpyro.sample(r'$v_{\rm sys}$', dist.Uniform(low=vsys_range[0], high=vsys_range[1]))
        
        constant = numpyro.sample(r'C', dist.Uniform(low=-3, high=3))

        beta = 1. 
        
        # Normally distributed likelihood
        numpyro.sample(
            "obs", dist.Normal(
                loc=trace_model_binned([flux_beg,flux_end,lw,vorb1,vorb2,vsys,constant],RV2D_binned_real.T,phi_binned_real,transit_binned,phase_start,phase_end,inclination), 
                scale=beta * jnp.array(npro_err)
            ), obs=jnp.array(npro_data)
        )
        return 0
    
        # Random numbers in jax are generated like this:
    
    
    rng_seed = 42
    rng_keys = split(
        PRNGKey(rng_seed), 
        cpu_cores
    )

    # Define a sampler, using here the No U-Turn Sampler (NUTS)
    # with a dense mass matrix:
    sampler = NUTS(
        numpyro_model, 
        dense_mass=True,
        target_accept_prob = 0.8,
        init_strategy = init_to_sample(),
        max_tree_depth=10
    )

    # Monte Carlo sampling for a number of steps and parallel chains: 
    mcmc = MCMC(
        sampler, 
        num_warmup=num_warmup, 
        num_samples=num_samples, 
        num_chains=cpu_cores
    )

    # Run the MCMC
    mcmc.run(rng_keys)


    # mcmc.print_summary()
    # arviz converts a numpyro MCMC object to an `InferenceData` object based on xarray:
    arviz_result = arviz.from_numpyro(mcmc)

    varnames = [r'$F_{\rm begin}$', r'$F_{\rm end}$', r'$\sigma_{\rm w}$',r'$v_{\rm orb, 1}$',r'$v_{\rm orb, 2}$',r'$v_{\rm sys}$', r'C']
    arr = np.zeros(np.shape((arviz_result['posterior']).to_array()))
    
    for i, v in enumerate(varnames):
        arr[i] = arviz_result['posterior'][v].to_numpy()

    return arviz_result, arr



@jit
def trace_model_binned(p,x,phi,transit,phase_start,phase_end,inclination):
    import numpy as np
    import jax.numpy as jnp

    flux_beg,flux_end,lw,vorb0,vorb1,vsys, C = p
    
    end = phase_start
    beg = phase_end 
    
    x0 = vsys + vorb1*jnp.sin(2.0*np.pi*phi)*jnp.sin(jnp.radians(inclination))
    x1 = vsys + vorb0*jnp.sin(2.0*np.pi*phi)*jnp.sin(jnp.radians(inclination))
    
    amp1 = -(flux_end - flux_beg) / (end - beg)
    amp0 = -(flux_beg*end - flux_end*beg) / (end - beg)    

    amplitude = transit * (amp0 + amp1*phi) / 1e4
    
    # Step function to switch between vorbs
    k = 1000
    f = 0.5*jnp.tanh(k*(phi)) + 0.5
    f_inv = -0.5*jnp.tanh(k*(phi)) + 0.5
        
    CCF_model = amplitude * f * jnp.exp(-0.5 * (x - x1)**2 / lw**2) + C /1e6 + amplitude * f_inv * jnp.exp(-0.5 * (x - x0)**2 / lw**2)
    


    return(CCF_model.T)





def model(params,x,phases,transit,phase_start,phase_end,inclination):
    import numpy as np

    flux_beg,flux_end,lw,vorb0,vorb1,vsys, C = params

    end = phase_start
    beg = phase_end 

    x0 = vsys + vorb1*np.sin(2.0*np.pi*phases)*np.sin(np.radians(inclination))
    x1 = vsys + vorb0*np.sin(2.0*np.pi*phases)*np.sin(np.radians(inclination))

    amp1 = -(flux_end - flux_beg) / (end - beg)
    amp0 = -(flux_beg*end - flux_end*beg) / (end - beg)    

    amplitude = (transit-1) * (amp0 + amp1*phases) / 1e4

    # Step function to switch between vorbs
    k = 1000
    f = 0.5*np.tanh(k*(phases)) + 0.5
    f_inv = -0.5*np.tanh(k*(phases)) + 0.5

    CCF_model = amplitude * f * np.exp(-0.5 * (x.T - x1)**2 / lw**2) + C /1e6 + amplitude * f_inv * np.exp(-0.5 * (x.T - x0)**2 / lw**2)

    return CCF_model.T


def plot_result(arr, CCF, RV, phases, transit, phase_start, phase_end, inclination, xlims=(-125,75), scaling=4):

    import numpy as np
    import matplotlib.pyplot as plt
    from corner import corner
    import matplotlib.patheffects as PathEffects
    #plt.rcParams.update({'font.size': 16})
    plt.rc('font', size=12)          # controls default text sizes
    #plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
    plt.rc('legend', fontsize=14)    # legend fontsize
    #plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["ytick.major.width"] = 2
    plt.rcParams["xtick.major.width"] = 2
    plt.rcParams["xtick.major.size"] = 5
    plt.rcParams["ytick.major.size"] = 5
    plt.rcParams["xtick.minor.size"] = 3.5
    plt.rcParams["ytick.minor.size"] = 3.5
    #print(plt.rcParams.keys())
    plt.rcParams["ytick.minor.width"] = 1
    plt.rcParams["xtick.minor.width"] = 1
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['legend.facecolor'] = 'white'
    plt.rcParams['legend.edgecolor'] = 'white'
    plt.rcParams['legend.framealpha'] = 1

    varnames = [r'$F_{\rm begin}$', r'$F_{\rm end}$', r'$\sigma_{\rm w}$',r'$v_{\rm orb, 1}$',r'$v_{\rm orb, 2}$',r'$v_{\rm sys}$', r'C']

    medians = np.median(arr, axis=(1,2))
    stds = np.std(arr, axis=(1,2))


    fig = corner(
        arr.T, 
        quiet=True,
        labels=varnames,
        truths = medians,
        show_titles=True,
        truth_color='red',
        title_kwargs={"fontsize": 16},
    )

    # set NaNs in CCF to 0 so the plot looks a bit nicer ;)
    CCF_binned_with_NaNs = CCF
    CCF_binned_with_NaNs[np.isnan(CCF_binned_with_NaNs)] = 0.0


    vmin = np.nanmean(CCF_binned_with_NaNs)-scaling*np.nanstd(CCF_binned_with_NaNs)
    vmax = np.nanmean(CCF_binned_with_NaNs)+scaling*np.nanstd(CCF_binned_with_NaNs)


    left, bottom, width, height = [0.70, 0.75, 0.28, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.pcolormesh(RV,phases,-CCF,shading='auto',vmin=vmin,vmax=vmax,linewidth=0,rasterized=True) # binned CCF with NaNs

    ax2.set_xlim(xlims)
    ax2.set_ylabel('Orbital phase')
    ax2.set_xticks([])

    params = [medians[i] for i in range(len(medians))]
    left, bottom, width, height = [0.70, 0.55, 0.28, 0.2]
    ax3 = fig.add_axes([left, bottom, width, height])

    RV2D = np.tile(RV, len(phases))
    RV2D = np.reshape(RV2D, (len(phases), len(RV)))

    model = model(
                    params=params,
                    x=RV2D,
                    phases=phases,
                    transit=transit,
                    phase_start=phase_start,
                    phase_end=phase_end,
                    inclination=inclination
    )
    ax3.pcolormesh(RV,phases, model, shading='auto',vmin=vmin,vmax=vmax,linewidth=0,rasterized=True)

    ax3.set_xlabel(r'Radial velocity [km s$^{-1}$]')
    ax3.set_xlim(xlims)
    ax3.set_ylabel('Orbital phase')

    fig.patch.set_facecolor('white')