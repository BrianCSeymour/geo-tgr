import jax.numpy as jnp
import jax 
from jax import grad, vmap
import numpy as np
from jimgw.waveform import RippleIMRPhenomD
from jimgw.detector import H1, L1, V1
jax.config.update("jax_enable_x64", True)

import scipy.interpolate as interp
import scipy.integrate as integ
import scipy.linalg as sla
import pycbc.conversions

waveform = RippleIMRPhenomD(f_ref=20)

import astropy.units as u
from astropy import constants as const

Ms = (u.Msun * const.G / const.c**3 ).si.value

# ------------ Fisher Stuff --------

def read_mag(freq, fileName):
    f_tf, mag_tf = np.loadtxt(fileName, unpack=True)
    
    idx = jnp.where(f_tf>0)
    mag_tf = mag_tf[idx]
    f_tf = f_tf[idx]

    mag_func = interp.interp1d(jnp.log(f_tf), jnp.log(mag_tf), kind='linear', bounds_error=False, fill_value=np.inf)
    mag_out = jnp.exp(mag_func(jnp.log(freq)))
    return mag_out

def innprod(hf1, hf2, psd, freqs):
    # prod = 2. * integ.simps( (jnp.conj(hf1) * hf2 + hf1 * jnp.conj(hf2)) / psd , freqs)
    prod = 2. * jax.scipy.integrate.trapezoid( (jnp.conj(hf1) * hf2 + hf1 * jnp.conj(hf2)) / psd , freqs)
    return prod

def fishslow(freqs, dh, par, idx_par, psd, log_flag):

    n_pt = len(freqs)
    n_dof = len(idx_par)

    dh_arr = jnp.zeros([n_dof, n_pt], dtype=np.complex128)

    for idx in idx_par:
        dh_arr = dh_arr.at[idx_par[idx],:].set(dh[idx])
        # dh_arr[idx_par[idx],:] = dh[idx]

        if log_flag[idx]:
            # dh_arr[idx_par[idx]] *= par[idx]
            dh_arr = dh_arr.at[idx_par[idx],:].set(dh_arr[idx_par[idx]] * par[idx])
            

        gamma = jnp.zeros([n_dof, n_dof], dtype=np.float64)

    for i in range(n_dof):
        for j in range(i, n_dof):
            # gamma[i, j] = jnp.real(innprod(dh_arr[i, :], dh_arr[j, :], psd, freqs))
            gamma = gamma.at[i, j].set(np.real(innprod(dh_arr[i, :], dh_arr[j, :], psd, freqs)))

    for i in range(n_dof):
        for j in range(i):
            # gamma[i, j] = jnp.conj(gamma[j, i])
            gamma = gamma.at[i, j].set(jnp.conj(gamma[j, i]))

    return gamma

@jax.jit
def fish(freqs, dh, par, idx_par, psd, log_flag):
    n_pt = len(freqs)
    n_dof = len(idx_par)

    dh_arr = jnp.zeros([n_dof, n_pt], dtype=jnp.complex128)

    # Convert idx_par to a list for static looping
    idx_list = list(idx_par.keys())
    for idx in idx_list:
        idx_position = idx_par[idx]
        dh_arr = dh_arr.at[idx_position, :].set(dh[idx])

        # Use jax.lax.cond for conditional multiplication
        print(idx_par[idx],log_flag[idx])
        dh_arr = dh_arr.at[idx_position, :].set(
            jax.lax.cond(
                log_flag[idx],
                lambda x: x * par[idx],
                lambda x: x,
                dh_arr[idx_position, :]
            )
        )

    gamma = jnp.zeros([n_dof, n_dof], dtype=jnp.float64)

    # Use static loops
    for i in range(n_dof):
        for j in range(i, n_dof):
            gamma = gamma.at[i, j].set(
                jnp.real(innprod(dh_arr[i, :], dh_arr[j, :], psd, freqs))
            )
        for j in range(i):
            gamma = gamma.at[i, j].set(jnp.conj(gamma[j, i]))

    return gamma

def get_cov_param_rel(fish, idx_par):
    cov = sla.inv(fish)
    cov_param = {}
    for i in idx_par:
        cov_param[i] = jnp.sqrt(cov[idx_par[i], idx_par[i]])
    
    return cov_param

# ------------ wavefrom derivs --------

def get_h_H1_slow(x, f):
    # not measuring spins nor gmst/epoch since these are tc
    x['s1_x'] = jnp.array(0.); x['s1_y'] = jnp.array(0.); x['s1_z'] = jnp.array(0.)
    x['s2_x'] = jnp.array(0.); x['s2_y'] = jnp.array(0.); x['s2_z'] = jnp.array(0.)
    x['gmst'] = jnp.array(0.); x['epoch'] = jnp.array(0.)
    
    ff = jnp.array([f])
    h_sky = waveform(ff, x)
    align_time = jnp.exp(-1j*2*jnp.pi*ff*(x['epoch']+x['t_c']))
    signal = H1.fd_response(ff, h_sky, x) * align_time
    return signal[0]

def get_h_L1_slow(x, f):
    x['s1_x'] = jnp.array(0.); x['s1_y'] = jnp.array(0.); x['s1_z'] = jnp.array(0.)
    x['s2_x'] = jnp.array(0.); x['s2_y'] = jnp.array(0.); x['s2_z'] = jnp.array(0.)
    x['gmst'] = jnp.array(0.); x['epoch'] = jnp.array(0.)
    
    ff = jnp.array([f])
    h_sky = waveform(ff, x)
    align_time = jnp.exp(-1j*2*jnp.pi*ff*(x['epoch']+x['t_c']))
    signal = L1.fd_response(ff, h_sky, x) * align_time
    return signal[0]

def get_h_V1_slow(x, f):
    x['s1_x'] = jnp.array(0.); x['s1_y'] = jnp.array(0.); x['s1_z'] = jnp.array(0.)
    x['s2_x'] = jnp.array(0.); x['s2_y'] = jnp.array(0.); x['s2_z'] = jnp.array(0.)
    x['gmst'] = jnp.array(0.); x['epoch'] = jnp.array(0.)
    
    ff = jnp.array([f])
    h_sky = waveform(ff, x)
    align_time = jnp.exp(-1j*2*jnp.pi*ff*(x['epoch']+x['t_c']))
    signal = V1.fd_response(ff, h_sky, x) * align_time
    return signal[0]

# @jax.jit
def get_dh_H1(x,f):
    ur = vmap(grad(lambda x,f : get_h_H1_slow(x,f).real), in_axes=(None, 0))(x,f)
    ui = vmap(grad(lambda x,f : get_h_H1_slow(x,f).imag), in_axes=(None, 0))(x,f)
    dh = {key: ur.get(key, 0) + 1j*ui.get(key, 0) for key in x}
    return dh

# @jax.jit
def get_dh_L1(x,f):
    ur = vmap(grad(lambda x,f : get_h_L1_slow(x,f).real), in_axes=(None, 0))(x,f)
    ui = vmap(grad(lambda x,f : get_h_L1_slow(x,f).imag), in_axes=(None, 0))(x,f)
    dh = {key: ur.get(key, 0) + 1j*ui.get(key, 0) for key in x}
    return dh

# @jax.jit
def get_dh_V1(x,f):
    ur = vmap(grad(lambda x,f : get_h_V1_slow(x,f).real), in_axes=(None, 0))(x,f)
    ui = vmap(grad(lambda x,f : get_h_V1_slow(x,f).imag), in_axes=(None, 0))(x,f)
    dh = {key: ur.get(key, 0) + 1j*ui.get(key, 0) for key in x}
    return dh

# get_h_H1 = jax.jit(vmap(lambda x,f : get_h_H1_slow(x,f), in_axes=(None, 0)))
# get_h_L1 = jax.jit(vmap(lambda x,f : get_h_L1_slow(x,f), in_axes=(None, 0)))
# get_h_V1 = jax.jit(vmap(lambda x,f : get_h_V1_slow(x,f), in_axes=(None, 0)))

get_h_H1 = vmap(lambda x,f : get_h_H1_slow(x,f), in_axes=(None, 0))
get_h_L1 = vmap(lambda x,f : get_h_L1_slow(x,f), in_axes=(None, 0))
get_h_V1 = vmap(lambda x,f : get_h_V1_slow(x,f), in_axes=(None, 0))



def get_FI(freqs, red_param, idx_par, psd, log_flag):
    dh_H1  = get_dh_H1(red_param, freqs)
    dh_L1  = get_dh_L1(red_param, freqs)
    dh_V1  = get_dh_V1(red_param, freqs)
    
    fi_H1 = fish(freqs, dh_H1, red_param, idx_par, psd, log_flag)
    fi_L1 = fish(freqs, dh_L1, red_param, idx_par, psd, log_flag)
    fi_V1 = fish(freqs, dh_V1, red_param, idx_par, psd, log_flag)
    return fi_H1, fi_L1, fi_V1

def get_snrs(freqs, red_param, psd):
    h_H1   = get_h_H1(red_param, freqs)
    h_L1   = get_h_L1(red_param, freqs)
    h_V1   = get_h_V1(red_param, freqs)

    snr_H1 = jnp.real(innprod(h_H1, h_H1, psd, freqs)**(1/2))
    snr_L1 = jnp.real(innprod(h_L1, h_L1, psd, freqs)**(1/2))
    snr_V1 = jnp.real(innprod(h_V1, h_V1, psd, freqs)**(1/2))

    return snr_H1, snr_L1, snr_V1, (snr_H1**2 + snr_L1**2 + snr_V1**2)**(1/2)

def compute_bias(dh, dh_nvnl, psd, freqs, idx_par):
    res = { key : jnp.real(innprod(dh[key], dh_nvnl, psd, freqs)) for key in dh.keys()}
    # bias = jnp.zeros(len(idx_par))
    # for idx in idx_par:
    #     bias[idx_par[idx]] = res[idx]
    bias = [res[key] for key, index in sorted(idx_par.items(), key=lambda item: item[1])]

    bias = jnp.real(jnp.array(bias))
    return bias

def get_FI_ppe(freqs, red_param, idx_par, psd, log_flag, k):
    dh_H1  = get_dh_H1(red_param, freqs)
    dh_L1  = get_dh_L1(red_param, freqs)
    dh_V1  = get_dh_V1(red_param, freqs)
    
    h_H1   = get_h_H1(red_param, freqs)
    h_L1   = get_h_L1(red_param, freqs)
    h_V1   = get_h_V1(red_param, freqs)

    dpsi_ppe = get_dpsi_ppe(freqs, red_param, k)

    ########## alot more work needs to be done. I think I need to calculate the value for ppe first then recompute the FI matrix at that point 
    dh_H1["phi_k"] = 1j*dpsi_ppe*h_H1
    dh_L1["phi_k"] = 1j*dpsi_ppe*h_L1
    dh_V1["phi_k"] = 1j*dpsi_ppe*h_V1

    fi_H1 = fish(freqs, dh_H1, red_param, idx_par, psd, log_flag)
    fi_L1 = fish(freqs, dh_L1, red_param, idx_par, psd, log_flag)
    fi_V1 = fish(freqs, dh_V1, red_param, idx_par, psd, log_flag)


    return fi_H1, fi_L1, fi_V1

def get_dpsi_ppe_inner(freqs, par, k):

    Mc = par["M_c"]
    η = par["eta"]
    
    M = pycbc.conversions.mtotal_from_mchirp_eta(Mc,η)*Ms
    phi0 = 1
    phi1 = 0
    phi2 = 3715/756 + 55/9*η
    phi3 = -16*np.pi
    phi4 =  15293365/508032+27145 *η /503+ 3085 *η**2 / 72
    phi5 = ((38645 * np.pi / 756) - (65 * np.pi * η / 9))
    pi = np.pi
    gamma_e = np.euler_gamma
    eta = η
    phi6 =  (11583231236531 / 4694215680) - (6848 * gamma_e / 21) - (640 * pi**2 / 3) + (-15737765635 / 3048192 + 2255 * pi**2 / 12) * eta + 76055 * eta**2 / 1728 -     127825 * eta**3 / 1296 
    phi7 = (77096675 * pi / 254016) + (378515 * pi * eta / 1512) - (74045 * pi * eta**2 / 756) 
    # k = kargs["k"]
    # δφ_k = par['dphi_k']
    δφ_k = 1
    
    # these come from eq B3 of 1508.07253
    if k == -2:
        dpsi = 3 / 128 / η * (np.pi * freqs * M)**(-5/3) * δφ_k * (np.pi * freqs * M)**(k/3)
    elif k == -1:
        dpsi = 3 / 128 / η * (np.pi * freqs * M)**(-5/3) * δφ_k * (np.pi * freqs * M)**(k/3)
    elif k == 0:
        dpsi = 3 / 128 / η * (np.pi * freqs * M)**(-5/3) * phi0 * δφ_k * (np.pi * freqs * M)**(k/3)
    elif k == 1:
        dpsi = 3 / 128 / η * (np.pi * freqs * M)**(-5/3) * δφ_k * (np.pi * freqs * M)**(k/3)
    elif k == 2:
        dpsi = 3 / 128 / η * (np.pi * freqs * M)**(-5/3) * phi2 * δφ_k * (np.pi * freqs * M)**(k/3)
    elif k == 3:
        dpsi = 3 / 128 / η * (np.pi * freqs * M)**(-5/3) * phi3 * δφ_k * (np.pi * freqs * M)**(k/3)
    elif k == 4:
        dpsi = 3 / 128 / η * (np.pi * freqs * M)**(-5/3) * phi4 * δφ_k * (np.pi * freqs * M)**(k/3)
    elif k == 5:
        dpsi = 3 / 128 / η * (np.pi * freqs * M)**(-5/3) * phi5 * δφ_k * (np.pi * freqs * M)**(k/3)
    elif k == 6:
        dpsi = 3 / 128 / η * (np.pi * freqs * M)**(-5/3) * phi6 * δφ_k * (np.pi * freqs * M)**(k/3)
    elif k == 7:
        dpsi = 3 / 128 / η * (np.pi * freqs * M)**(-5/3) * phi7 * δφ_k * (np.pi * freqs * M)**(k/3)
    else:
        print(k)
        print("power error defn")
    
    return dpsi

def get_dpsi_ppe(freqs, par, k):
    fend = 0.04257918562317578 # to match eob file 
    fstart = 0.004432985313285457
    Mc = par["M_c"]
    eta = par["eta"]
    fend = fend/pycbc.conversions.mtotal_from_mchirp_eta(Mc,eta)/Ms
    fstart = fstart/pycbc.conversions.mtotal_from_mchirp_eta(Mc,eta)/Ms

    dpsi = get_dpsi_ppe_inner(freqs, par, k)
    # dpsi[freqs>fend] = get_dpsi_ppe_inner(fend, par) # numpy version

    dpsi = jnp.where(freqs>fend, get_dpsi_ppe_inner(fend, par, k), dpsi) # jax version
    # dpsi = dpsi.at[freqs>fend].set(get_dpsi_ppe_inner(fend, par, k)) 
    return dpsi

