import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dts
from matplotlib import colors
from matplotlib.pyplot import cm
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from astral import Observer
from astral.sun import noon
from scipy.signal import correlate, correlation_lags
import aerosol.functions as af
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.special import erf

# All constants are SI base units
E=1.602E-19           # elementary charge
E_0=8.85418781e-12    # permittivity of vacuum
K_B=1.381e-23         # Boltzmann constant 
R=8.3413              # gas constant

# THAB mobility (cm2V-1S-1)
Z_THA = 0.97

def diam2mob(dp,temp=293.15,pres=101325.0,ne=1,gas="air"):
    """ 
    Convert electrical mobility diameter to electrical mobility in gas

    Parameters
    ----------

    dp : float
        particle diameter(s),
        unit : nm
    temp : float
        ambient temperature
        default 20 C 
        unit: K
    pres : float
        ambient pressure,
        default 1 atm 
        unit: Pa
    ne : int
        number and polarity of charges on the aerosol particle
        default 1
    gas : str
        air (default) or nitrogen

    Returns
    -------

    float
        particle electrical mobility, 
        unit: cm2 s-1 V-1

    """

    cc = af.slipcorr(dp*1e-9,temp,pres,gas=gas) # dataframe
    mu = af.gas_viscosity(temp,gas=gas) # series

    Zp = (ne*E*cc)/(3.*np.pi*mu*dp*1e-9)*1e4

    return Zp

def mob2diam(Zp,temp=293.15,pres=101325.,ne=1, tol=1e-3, maxiter=100, gas="air"):
    """
    Convert electrical mobility to electrical mobility diameter in gas

    Parameters
    ----------

    Zp : float
        particle electrical mobility or mobilities, 
        unit: cm2 s-1 V-1
    temp : float
        ambient temperature, 
        unit: K
    pres : float
        ambient pressure, 
        unit: Pa
    ne : integer
        number and polarity of elementary charges on the aerosol particle
    gas : str
        air (default) or nitrogen
    
    Returns
    -------

    float
        particle diameter, unit: m
    
    """
    
    ne = np.abs(ne)
    Zp = np.abs(Zp)

    def minimize_this(dp,Z):
        return np.abs(diam2mob(dp,temp,pres,ne,gas)-Z)

    # Initial guessing
    if (Zp>0.1):
        dp0=1.0
    elif (Zp>0.001):
        dp0=10.0
    elif (Zp>=0.001):
        dp0=50.0
    elif (Zp>=0.0001):
        dp0=150.0
    elif (Zp>=0.00001):
        dp0=1000.0
    else:
        dp0=10000.0

    # Optimization using Nelder Mead method
    diam = minimize(minimize_this, 
        dp0, 
        args=(Zp,), 
        tol=tol, 
        method='Nelder-Mead',
        options={"maxiter":maxiter})

    if not diam.success:
        return np.nan
    else:
        return diam.x[0]

def tubeloss(diam, flowrate, tubelength, temp=293.15, pres=101325.):
    """
    Calculate diffusional particle losses to walls of
    straight cylindrical tube assuming a laminar flow regime

    Parameters
    ----------
    
    diam : float or series of length m
        Particle diameters for which to calculate the
        losses, unit: m
    flowrate : float or series of length n
        unit: L/min
    tubelength : float
        Length of the cylindrical tube
        unit: m
    temp : float or series of length n
        temperature
        unit: K
    pres : float or series of lenght n
        air pressure
        unit: Pa

    Returns
    -------

    float or dataframe of shape (n,m)
        Fraction of particles passing through.
        Each column represents diameter and each
        each row represents different temperature
        pressure and flowrate value
        
    """

    float_input=af.is_input_float([diam,flowrate,temp,pres])

    temp=pd.Series(temp)
    pres=pd.Series(pres)
    diam=pd.Series(diam)
    flowrate = pd.Series(flowrate)*1.667e-5

    idx = af.get_index([temp,pres,flowrate])
    
    D = af.particle_diffusivity(diam,temp,pres)

    rmuu = D.values*tubelength*(1./flowrate.values.reshape(-1,1))
    
    penetration = np.nan*np.ones(rmuu.shape)

    condition1 = (rmuu<0.009)
    condition2 = (rmuu>=0.009)

    penetration[condition1] = 1.-5.5*rmuu[condition1]**(2./3.)+3.77*rmuu[condition1]
    penetration[condition2] = 0.819*np.exp(-11.5*rmuu[condition2])+0.0975*np.exp(-70.1*rmuu[condition2])
    
    if float_input:
        return penetration[0][0]
    else:
        return pd.DataFrame(index=idx,columns=diam.values,data=penetration)

def flow_velocity_in_pipe(tube_diam,flowrate):
    """
    Calculate fluid speed from the flow rate in circular tube
 
    Parameters
    ----------

    tube_diam : float or series of lenght m
        Diameter of circular tube (m)
    flowrate : float or series of lenght n
        Volumetric flow rate (lpm)

    Returns
    -------

    float or dataframe of shape (n,m)
        Speed of fluid (m/s) 

    """

    float_input = af.is_input_float([tube_diam,flowrate])

    tube_diam = pd.Series(tube_diam)
    flowrate = pd.Series(flowrate)
 
    tube_diam = tube_diam.values
    flowrate = flowrate.values.reshape(-1,1)
    
    volu_flow = flowrate/60000.
    cross_area = np.pi*(tube_diam/2.)**2
    
    vel = volu_flow/cross_area

    if float_input:
        return vel[0][0] 
    else:
        return pd.DataFrame(index = flowrate.flatten(), columns = tube_diam, data = vel)

def pipe_reynolds(
    tube_diam,
    flowrate,
    temp=293.15,
    pres=101325.0):
    """
    Calculate Reynolds number in a tube

    Parameters
    ----------

    tube_diam : float or series of length m
        Inner diameter of the tube (m)
    flowrate : float or series of lenght n
        Volumetric flow rate (lpm)
    temp : float or series of length n
        Temperature in K
    pres : float or series of length n
        Pressure in Pa

    Returns
    -------

    float or dataframe of shape (n,m)
        Reynolds number

    """

    float_input = af.is_input_float([tube_diam,flowrate,temp,pres])

    tube_diam = pd.Series(tube_diam)
    flowrate = pd.Series(flowrate)
    temp = pd.Series(temp)
    pres = pd.Series(pres)

    idx = af.get_index([flowrate,temp,pres]) 

    tube_diam = tube_diam.values
    flowrate = flowrate.values.reshape(-1,1)
         
    volu_flow = flowrate/60000.
    visc = af.gas_viscosity(temp)
    dens = af.air_density(temp,pres)

    visc = visc.values.reshape(-1,1)
    dens = dens.values.reshape(-1,1)

    Re = (dens*volu_flow*tube_diam)/(visc*np.pi*(tube_diam/2.0)**2)

    if float_input:
        return Re[0][0]
    else:
        return pd.DataFrame(index = idx, columns=tube_diam, data=Re)


def refmob_dp2volts(ref_voltage,dp,gas="air",ref_mobility = Z_THA):
    """
    Convert particle diameters to DMA voltages

    Parameters
    ----------

    ref_voltage : float
        Voltage at the reference mobility peak (V)
    dp : float or series
        Particle diameters (nm)
    gas : str
        air (default) or nitrogen
    ref_mobility : float
        Reference mobility (cm2 V-1 s-1)

        Default is the THA+ monomer as STP (0.97 cm2 V-1 s-1)

    Returns
    -------

    float or series:
        DMA voltage (V) corresponding to dp

    """
        
    Zp = diam2mob(dp, 293.15, 101325.0, 1, gas=gas)
   
    return (ref_voltage * ref_mobility)/Zp


def refmob_volts2dp(ref_voltage,dma_voltage,gas="air",ref_mobility=Z_THA):
    """
    Convert DMA voltages to particle diameters

    Parameters
    ----------

    ref_voltage : float
        Voltage at the reference mobility peak (V)
    dma_voltage : float
        DMA voltage (V)
    gas : str
        air (default) or nitrogen
    ref_mobility : float
        Reference mobility (cm2 V-1 s-1)

        Default is the THA+ monomer at STP (0.97 cm2 V-1 s-1)

    Returns
    -------

    float:
        particle diameter corresponding to DMA voltage (nm)

    """
        
    Zp = (ref_voltage*ref_mobility)/dma_voltage

    dp = mob2diam(Zp,293.15,101325.0,1,gas=gas)
    
    return dp

def calc_peg_mob(m_peg):
    """
    Calculate mobility for PEG molecule of a given mass
    
    Parameters
    ----------

    m_peg : float
        PEG mass in g/mol

    Returns
    -------

    float
        PEG mobility in cm2 V-1 s-1

    References
    ----------

    See https://pubs.acs.org/doi/10.1021/ac034138m
    """

    A=0.355
    B=0.0936
    return 1./np.sqrt(A + B*m**(1./3.))

def calc_peg_dp(m_peg):
    """
    Calculate mobility diameter for PEG molecule of a given mass
    
    Parameters
    ----------

    m_peg : float
        PEG mass in g/mol

    Returns
    -------

    float
        PEG mobility diameter in nm

    References
    ----------

    See https://pubs.acs.org/doi/10.1021/ac034138m
    """
    
    return mob2diam(calc_peg_mob(m_peg))

def eq_charge_frac(dp,N):
    """
    Calculate equilibrium charge fraction using Wiedensohler (1988) approximation

    Parameters
    ----------

    dp : float
        Particle diameter (m)
    N : int
        Amount of elementary charge in range [-2,2]

    Returns
    -------

    float
        Fraction of particles of diameter dp having N 
        elementary charges 

    """

    a = {-2:np.array([-26.3328,35.9044,-21.4608,7.0867,-1.3088,0.1051]),
        -1:np.array([-2.3197,0.6175,0.6201,-0.1105,-0.1260,0.0297]),
        0:np.array([-0.0003,-0.1014,0.3073,-0.3372,0.1023,-0.0105]),
        1:np.array([-2.3484,0.6044,0.4800,0.0013,-0.1544,0.0320]),
        2:np.array([-44.4756,79.3772,-62.8900,26.4492,-5.7480,0.5059])}

    if (np.abs(N)>2):
        raise Exception("Number of elementary charges must be 2 or less")
    elif ((dp<20e-9) & (np.abs(N)==2)):
        return 0
    else:
        return 10**np.sum(a[N]*(np.log10(dp*1e9)**np.arange(6)))

def calc_tube_residence_time(tube_diam,tube_length,flowrate):
    """
    Calculate residence time in a circular tube

    Parameters
    ----------

    tube_diam : float or series of length m
        Inner diameter of the tube (m)
    tube_length : float or series of length m
        Length of the tube (m)
    flowrate : float or series of length n
        Volumetric flow rate (lpm)

    Returns
    -------

    float or dataframe of shape (n,m)
        Average residence time in seconds

    """

    float_input = af.is_input_float([tube_diam,tube_length,flowrate])

    tube_diam = pd.Series(tube_diam)
    tube_length = pd.Series(tube_length)
    flowrate = pd.Series(flowrate)

    tube_diam = tube_diam.values
    tube_length = tube_length.values
    flowrate = flowrate.values.reshape(-1,1)
         
    volu_flow = flowrate/60000.
    tube_volume = np.pi*tube_diam**2*(1/4.)*tube_length

    rt = tube_volume/volu_flow

    if float_input:
        return rt[0][0]
    else:
        return pd.DataFrame(index = flowrate.flatten(), columns=tube_volume, data=rt)

def dma_volts2mob(Q,R1,R2,L,V):
    """
    Theoretical selected mobility from cylindrical DMA

    Parameters
    ----------

    Q : float
        sheath flow rate, unit lpm

    R1 : float
        inner electrode radius, unit m

    R2 : float
        outer electrode radius, unit m

    L : float
        effective electrode length, unit m

    V : float or series
        applied voltage, unit V

    Returns
    -------

    float or series
        selected mobility, unit cm2 s-1 V-1

    """

    return ((Q*1.667e-5)*np.log(R2/R1))/(2.*np.pi*L*V)*1e4

def dma_mob2volts(Q,R1,R2,L,Z):
    """
    Cylindrical DMA voltage corresponding to mobility

    Parameters
    ----------

    Q : float
        sheath flow rate, unit lpm

    R1 : float
        inner electrode radius, unit m

    R2 : float
        outer electrode radius, unit m

    L : float
        effective electrode length, unit m

    Z : float
        mobility, unit cm2 s-1 V-1

    Returns
    -------

    float
        DMA voltage, unit V

    """

    return ((Q*1.667e-5)*np.log(R2/R1))/(2.*np.pi*L*Z*1e-4)

def conical_dma_mob2volts(Q, R1_max, R2, L, alpha, Z):
    """
    Conical DMA voltage corresponding to mobility

    Parameters
    ----------

    Q : float
        sheath flow rate, unit lpm

    R1_max : float
        inner electrode radius at outlet at distance L, unit m

    R2 : float
        outer electrode radius, unit m

    L : float
        effective electrode length, unit m
        
    alpha : float
        tapering angle, unit degrees

    Z : float
        mobility, unit cm2 s-1 V-1

    Returns
    -------

    float
        DMA voltage, unit V

    """

    # Convert to radians
    alpha = alpha*(np.pi/180.)
    
    # Calculate the geometric factor K_T
    x = np.linspace(0,L,100)
    Rx = R1_max - (L-x) * np.tan(alpha)
    K_T = np.log(R2/R1_max)/L * np.trapezoid(1./np.log(R2/Rx), x)
    
    # Calculate the voltage
    V = ((Q*1.667e-5)*np.log(R2/R1_max))/(2*np.pi*L*(Z*1e-4)*K_T)
    
    return V


def tubeloss_turbulent(diam, flowrate, tube_length, tube_diam, temp=293.15, pres=101325.):
    """
    Calculate particle losses to walls of a straight cylindrical 
    tube assuming a turbulent flow regime and air as the carrier gas.

    Parameters
    ----------
    
    diam : float or series of length m
        Particle diameters for which to calculate the
        losses, unit: m
    flowrate : float or series of length n
        unit: L/min
    tube_length : float
        Length of the cylindrical tube
        unit: m
    tube_diam : float
        Diameter of the cylindrical tube
        unit: m
    temp : float or series of length n
        temperature
        unit: K
    pres : float or series of lenght n
        air pressure
        unit: Pa

    Returns
    -------

    float or dataframe of shape (n,m)
        Fraction of particles passing through.
        Each column represents diameter and each
        each row represents different temperature
        pressure and flowrate value

    """
    
    float_input=af.is_input_float([diam,flowrate,temp,pres])

    temp=pd.Series(temp)
    pres=pd.Series(pres)
    diam=pd.Series(diam)
    flowrate = pd.Series(flowrate)*1.667e-5

    idx = af.get_index([temp,pres,flowrate])
    
    # Average flow velocity
    flow_velo = flow_velocity_in_pipe(tube_diam, flowrate) # shape: (len(flowrate),len(tube_diam)) or float
    flow_velo = flow_velo.values.flatten()

    # Reynolds number
    Re = pipe_reynolds(tube_diam, flowrate, temp, pres) # shape: (maxlen(flowrate,temp,pres),len(tube_diam)) or float
    Re = Re.values.flatten()

    # Particle diffusivity
    D = af.particle_diffusivity(diam,temp,pres) # shape: (maxlen(temp,pres),len(diam)) or float
    D = D.values

    # Air density
    air_dens = af.air_density(temp,pres) # shape: maxlen(temp,pres) or float
    air_dens = air_dens.values.flatten()

    # Air viscosity
    air_visc = af.gas_viscosity(temp) # shape: len(temp) or float
    air_visc = air_visc.values.flatten()

    # Diffusive deposition velocity
    #V_d = ((0.04*flow_velo)/(Re**(1/4.)))*((air_dens*D)/(air_visc))**(2/3.)

    delta = (28.5*tube_diam*D**(1/4.))/(Re**(7/8.)*(air_visc/air_dens)**1/4.) 
    V_d = D/delta

    penetration = np.exp(-(4*V_d*tube_length)/(tube_diam * flow_velo))

    if float_input:
        return penetration[0][0]
    else:
        return pd.DataFrame(index=idx,columns=diam.values,data=penetration)

def activation_curve(dp, E_infty, dp50, dp0):
    return E_infty * (1.0 - np.exp(-(dp-dp0)/(dp50-dp0)*np.log(2)))

def det_eff_fit(diams, det_effs):
    r"""
    Fit activation curve to detection efficiency data

    The below function is used in the fit:

    .. math::
        
        \eta (d_p) = \eta_{\infty}\times \left( 1 - \exp \left( -\frac{d_p - d_{p50}}{d_{p50} - d_{p0}} \right) \times \log 2  \right) 
    

    Parameters
    ----------

    diams : 1-d array
        Selected diameters in nm

    det_effs : 1-d array
        Measured detection efficiencies

    Returns
    -------

    list
        Fitted parameters

        0. Plateau detection efficiency
        1. dp50
        2. dp0

    list
        parameter standard deviations

    """
        
    det_effs_max = np.max(det_effs)
    
    det_effs_norm = det_effs / det_effs_max

    D50 = np.interp(0.50, det_effs_norm, diams)

    D10 = np.interp(0.10, det_effs_norm, diams)

    initial_guess = [det_effs_max, D50, D10]

    params, cov = curve_fit(
        activation_curve, 
        diams, 
        det_effs, 
        p0=initial_guess,
    )

    return params, cov


def calc_equiv_loss_tube_length(q,l_ref,q_ref):
    """
    Calculate tube length to match losses for different flow rates

    Parameters
    ----------

    q : float
        Flow rate in the tube
    l_ref : float
        Length of the reference tube
    q_ref :  float
        Flow rate in the reference tube

    Returns 
    -------

    float
        Length of the tube

    """
    return l_ref/q_ref * q
