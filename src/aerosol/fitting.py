from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import aerosol.functions as af
from scipy.optimize import curve_fit

def to_meters(x):
    return (10**x)*1e-9

def gaussian(x, amplitude, mean, sigma):
    return amplitude * 1.0/(sigma * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ((x - mean) / sigma) ** 2.0)

def fit_gmm(samples,n_components,coef):

    gmm = GaussianMixture(
        n_components=n_components,
        n_init=20,
        init_params="kmeans",
        reg_covar=0.1**2,
        )
    
    gmm.fit(samples.reshape(-1,1))

    weights = gmm.weights_
    means = gmm.means_[:,0]
    stddevs = np.array([np.sqrt(c[0,0]) for c in gmm.covariances_])
    
    gaussians = []
    for i in range(n_components):
        
        gaussian = {
            "mean":means[i],
            "sigma":stddevs[i],
            "amplitude":weights[i]*coef,
        }
        gaussians.append(gaussian)

    return gaussians

def multimodal_gaussian(x, *params):
    n_components = len(params) // 3
    y = np.zeros_like(x)
    for i in range(n_components):
        amplitude = params[i * 3]
        mean = params[i * 3 + 1]
        sigma = params[i * 3 + 2]
        y += gaussian(x, amplitude, mean, sigma)
    return y

def calc_pred(x,gaussians):
    pred = np.zeros(len(x))
    for g in gaussians:
        pred = pred + gaussian(x, g["amplitude"], g["mean"], g["sigma"])

    return list(pred)

def fit_multimodal_gaussian(data_x, data_y, gaussians):
    initial_guesses = []
    lower_bounds = []
    upper_bounds = []

    for g in gaussians:

        initial_guesses.append(g["amplitude"])
        initial_guesses.append(g["mean"]) 
        initial_guesses.append(g["sigma"])

        lower_bounds.append(1)
        lower_bounds.append(-np.inf)
        lower_bounds.append(-np.inf)        
        upper_bounds.append(np.inf)
        upper_bounds.append(np.inf)
        upper_bounds.append(np.inf)

    n_components = len(initial_guesses) // 3

    try:
        params, _ = curve_fit(
            multimodal_gaussian,
            data_x,
            data_y,
            p0=initial_guesses,
            bounds=(lower_bounds,upper_bounds),
        )
    except:
        return None

    gaussians = []
    for i in range(n_components):
        amplitude = params[i * 3]
        mean = params[i * 3 + 1]
        sigma = params[i * 3 + 2]
        gaussian = {
            "mean":mean,
            "sigma":sigma,
            "amplitude":amplitude
        }
        gaussians.append(gaussian)

    return gaussians

def calc_pred_gaussians(x,gaussians):
    pred_gaussians = []
    for g in gaussians:
        pred_gaussian = list(gaussian(x,g["amplitude"],g["mean"],g["sigma"]))
        pred_gaussians.append(pred_gaussian)
    return pred_gaussians
    
def get_peak_positions(gaussians):
    dp = []
    for g in gaussians:
        dp.append(to_meters(g["mean"]))
    return dp

def calc_conc_ndist(x,ndist):
    conc = af.calc_conc(
            pd.Series(index = to_meters(x),data=ndist).to_frame().transpose(),
            to_meters(x.min()),
            to_meters(x.max())).iloc[0,0]
    return conc

def calc_conc_gaus(x,gaussians):
    mode_concs = []
    for g in gaussians:
        pred_gaussian = list(gaussian(x,g["amplitude"],g["mean"],g["sigma"]))
        pred_gaussian = pd.Series(index=to_meters(x), data = pred_gaussian)
        xmin = g["mean"] - 5 * g["sigma"]
        xmax = g["mean"] + 5 * g["sigma"]
        mode_conc = af.calc_conc(
                pd.Series(index = to_meters(x),data=pred_gaussian).to_frame().transpose(),
                to_meters(xmin),
                to_meters(xmax)).iloc[0,0]
        mode_concs.append(mode_conc)
    return mode_concs

def fit_multimode(
    x, 
    y, 
    timestamp = None, 
    n_modes = 1, 
    n_samples = 10000):
    """
    Fit multimodal Gaussian to aerosol number-size distribution

    Parameters
    ----------

    x : 1d numpy array
        log10 of bin diameters in nm.
    y : 1d numpy array
        Number size distribution
    timestamp : pandas Timestamp or None
        timestamp associated with the number size distributions
    n_modes : int
        number of modes to fit
    n_samples : int
        Number of samples to draw from the distribution
        during the fitting process.
    
    Returns
    -------

    dictionary:
        Fit results

    """
    all_ok = True

    # Convert to pandas Series
    ds = pd.Series(index = x, data = y)

    # Interpolate away the NaN values but do not extrapolate, remove any NaN tails
    s = ds.interpolate(limit_area="inside").dropna()
    
    # Set negative values to zero
    s[s<0]=0
    
    # Recover x and y for fitting
    x_interp = s.index.values
    y_interp = s.values
    
    
    if np.sum(y_interp)==0:
        print("sum was zero")
        all_ok = False

    if len(x_interp)<5:
        print("too few points")
        all_ok = False
    else:
        coef = np.trapezoid(y_interp,x_interp)
        samples = af.sample_from_dist(x_interp,y_interp,n_samples)

        #print(coef)
        #print(samples)

        # Initial guess from clustering
        gaussians_gmm = fit_gmm(samples, n_modes, coef)
        
        #print(gaussians_gmm)

        # The actual fit
        gaussians_lsq = fit_multimodal_gaussian(x_interp, y_interp, gaussians_gmm)
        
        #print(gaussians_lsq)
        
        if gaussians_lsq is None:
            print("Least-squares fit failed")
            all_ok = False
        else:
            pass

    if all_ok:

        try:
            print("all was ok")
            # Make sure all the data is json compatible
            x_points_log = np.linspace(x.min(),x.max(),1000)
            dp = get_peak_positions(gaussians_lsq)
            predicted_ndist = calc_pred(x_points_log,gaussians_lsq)
            predicted_gaussians = calc_pred_gaussians(x_points_log,gaussians_lsq)
            total_conc = calc_conc_ndist(x_points_log,predicted_ndist)
            mode_concs = calc_conc_gaus(x_points_log,gaussians_lsq)
            diams = list(10**x_points_log)
            if timestamp is None:
                time = None
            else:    
                time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        except:
            print("fit failed")
            dp = []
            gaussians = []
            gaussians_gmm = []
            gaussians_lsq = []
            diams = []
            predicted_ndist = [] 
            predicted_gaussians = []
            total_conc = np.nan
            mode_concs = []
            if timestamp is None:
                time = None
            else:    
                time = timestamp.strftime("%Y-%m-%d %H:%M:%S")        
    else:
        print("fit failed")
        dp = []
        gaussians = []
        gaussians_gmm = []
        gaussians_lsq = []
        diams = []
        predicted_ndist = [] 
        predicted_gaussians = []
        total_conc = np.nan
        mode_concs = []
        if timestamp is None:
            time = None
        else:    
            time = timestamp.strftime("%Y-%m-%d %H:%M:%S")    
            
    # Construct the result dictionary
    result = {
        "time": time,
        "gaussians": gaussians_lsq,
        "number_of_gaussians": len(gaussians_lsq),
        "peak_diams": dp,
        "predicted_ndist": predicted_ndist,
        "diams": diams,
        "predicted_gauss": predicted_gaussians,
        "total_conc": total_conc,
        "mode_concs": mode_concs,
    }

    return result

def fit_multimodes(df, n_modes = 1, n_samples = 10000):
    """
    Fit multimodal Gaussian to a aerosol number size distribution (dataframe)

    Parameters
    ----------

    df : pandas DataFrame
        Aerosol number size distribution
    n_modes : int
        Number of modes to fit
    n_samples : int
        Number of samples to draw from the distribution
        during the fitting process.
    
    Returns
    -------

    list:
        List of fit results

    """

    df = df.dropna(how="all",axis=0)

    x = np.log10(df.columns.values.astype(float)*1e9)
    
    fit_results = []
    
    for j in range(df.shape[0]):
        y = df.iloc[j,:].values.flatten()
        fit_result = fit_multimode(x, y, df.index[j], n_modes = n_modes, n_samples = n_samples)

        fit_results.append(fit_result)
        
        if (fit_result["number_of_gaussians"]>0):
            print(f'{df.index[j]}: fitting successfull!')
        else:
            print(f'{df.index[j]}: fitting failed!')

    return fit_results
