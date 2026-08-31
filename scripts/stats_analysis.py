"""
Provide functions for statistical analysis.
"""
import numpy as np
from scipy import stats
import statsmodels.api as sm
from scipy.optimize import minimize

################## Summary statistics ######################
def stats_summaries(A):
    """
    Compute summary statistics of the connection-weight distribution.
    args:
        A: Adjacency matrix of the network.
    returns:
        std: Standard deviation of positive connection weights.
        s: Skewness of positive connection weights.
        k: Kurtosis of positive connection weights.
    """
    weights = A[A>0]
    std = np.std(weights)
    s = stats.skew(weights)
    k = stats.kurtosis(weights)
    return std, s, k

################# Model comparison ##########################
def ks_stats(A):
    """
    CCompute KS statistics between weights of the simulated network and
    seven candidate distributions.
    Candidate distributions: exponential, inverse Gaussian, normal, gamma, inverse gamma, lognormal and Weibull.
    args:
        A: Adjacency matrix of the network.
    returns:
        ks_dist: KS statistics of 7 candidate distributions.
    """
    weights = A[A>0]
    ks_dist = np.zeros(7)

    loc, scale = stats.expon.fit(weights)
    res = stats.kstest(weights, stats.expon.cdf, args = (loc, scale))
    ks_dist[0] = res.statistic
 
    mu, loc, scale = stats.invgauss.fit(weights, floc = 0)
    res = stats.kstest(weights, stats.invgauss.cdf, args = (mu, loc, scale))
    ks_dist[1] = res.statistic
    
    mu, sig = stats.norm.fit(weights) # MLE
    res = stats.kstest(weights, stats.norm.cdf, args = (mu, sig))
    ks_dist[2] = res.statistic
    
    a, loc, scale = stats.gamma.fit(weights, floc = 0)
    res = stats.kstest(weights, stats.gamma.cdf, args = (a, loc, scale))
    ks_dist[3] = res.statistic

    a, loc, scale = stats.invgamma.fit(weights, floc = 0)
    res = stats.kstest(weights, stats.invgamma.cdf, args = (a, loc, scale))
    ks_dist[4] = res.statistic
    
    s, loc, scale = stats.lognorm.fit(weights, floc=0)
    res = stats.kstest(weights, stats.lognorm.cdf, args = (s, loc, scale))
    ks_dist[5] = res.statistic
    
    c, loc, scale = stats.weibull_min.fit(weights, floc = 0)
    res = stats.kstest(weights, stats.weibull_min.cdf, args = (c, loc, scale))
    ks_dist[6] = res.statistic
    
    return ks_dist

def best_ks(ks_stats):
    """
    Compare the best-fitting distribution with the other candidate distributions using Wilcoxon signed-rank tests.
    The best-fitting distribution is defined as the candidate with the smallest
    mean KS statistic across simulation instances.
    args:
        ks_stats: KS statistics of 7 candidate distributions across simulation instances.
    returns:
        wilcoxon_p: Dictionary containing p-values from the pairwise Wilcoxon
            signed-rank tests.
    """
    wilcoxon_p = {}
    dist_ls = ['Expon', 'InvNormal', 'Normal', 'Gamma', 'InvGamma', 'Lognorm', 'Weibull'] # candidate distributions

    x = np.mean(ks_stats, 0) # mean of KS statistics of instances for each candidate distributions
    best_dist_ind = np.where(x==np.min(x))[0] # the indices of the candidate distributions with the smallest mean KS statistics
    for L in best_dist_ind:
        wilcoxon_p[dist_ls[L]] = {}
        X = ks_stats[:, L] # KS statistics of all instances of the candidate distribution with the smallest mean
        for l, dist in enumerate(dist_ls):
            if l != L: # compare with other candidate distributions
                res = stats.wilcoxon(x = X, y = ks_stats[:, l], alternative='less') 
                wilcoxon_p[dist_ls[L]][dist] = res.pvalue

    return wilcoxon_p

################### Power-law fit #######################
def neg_log_likelihood(beta, X, y):
    """
    Compute the negative log-likelihood for a linear regression model.
    args:
        beta: Regression coefficients, with the intercept as the first element
            and the slope as the second element.
        X: Design matrix of independent variables.
        y: Observed values of the dependent variable.
    returns:
        Negative log-likelihood of the fitted model.
    """
    n = len(y)
    residuals = y - X @ beta
    sigma2 = np.sum(residuals**2) / n
    log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma2) - np.sum(residuals**2) / (2 * sigma2)
    return -log_likelihood  # minimize negative log-likelihood
    
def R2(y, ypred):
    """
    Compute the coefficient of determination (R^2) of a linear fit.
    args:
        y: Observed values of the dependent variable.
        ypred: Predicted values of the dependent variable.
    returns:
        r_squared: Coefficient of determination of the fit.
    """
    ss_tot = np.sum(np.power(y - np.mean(y), 2))
    if ss_tot == 0:
        return np.nan
    ss_res = np.sum(np.power(y - ypred, 2))
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared    

def PL_MLE(A, num_bins):
    """
    Fit the right tail of the connection-weight distribution with a linear
    model in log-log space.
    args:
        A: Adjacency matrix of the network.
        num_bins: Number of logarithmic bins used for the right tail of the
            weight distribution.
    returns:
        pl_fit: List containing the fitted intercept, slope, and R².
    """
    weights = A[A>0]
    wtrunc = weights[weights>=np.quantile(weights, 0.5)]

    # Get the values of the histogram
    lb = np.min(np.log10(wtrunc)); ub = np.max(np.log10(wtrunc))
    bins = np.logspace(lb, ub, num_bins*2 + 1)
    v, _ = np.histogram(weights, bins[::2], density = True)
    x = bins[1::2]
    x = x[v>0]; v = v[v>0]

    # independet and dependent variables of the linear fit
    X = np.log10(x)
    X = sm.add_constant(X) # Add intercept term
    Y = np.log10(v)
    # Initial guess
    beta_init = np.zeros(X.shape[1])
    # Perform optimization
    result = minimize(neg_log_likelihood, beta_init, args=(X, Y))
    # Estimated parameters
    beta_mle = result.x
    # Predict using the estimated coefficients
    Y_pred = X @ beta_mle
    # Get r2 of the fit
    r2 = R2(Y, Y_pred)
    
    pl_fit = [beta_mle[0], beta_mle[1], r2]
    return pl_fit

##################### Goodness of fit ################################
def zero_centered(A):
    """
    Center the log-transformed connection-weight distribution at zero.
    args:
        A: Adjacency matrix of the network.
    returns:
        log_w: Zero-centered log-transformed connection weights.
    """
    w = A[A>0]
    log_w = np.log(w)
    mu, sig = stats.norm.fit(log_w)
    log_w = log_w - mu
    return log_w
    
def Waller_gof(A_real, A_sim):
    """
    Evaluate goodness-of-fit between empirical and simulated weight distributions.
    The empirical distribution is compared with the pooled simulated distributions using the two-sample KS statistic, 
    and the resulting statistic is evaluated relative to KS statistics obtained between individual simulated networks 
    and the remaining simulated networks.
    args:
        A_real: Adjacency matrix of the empirical network.
        A_sim: Dictionary of adjacency matrices from simulated networks.
    returns:
        pvalue: Monte Carlo p-value for the goodness-of-fit test.
    """
    log_wreal = zero_centered(A_real)
    
    log_wsim_ls = []
    for k in A_sim.keys():
        log_wsim = zero_centered(A_sim[k])
        log_wsim_ls.append(list(log_wsim))
    log_wsim_pooled = [x for sublist in log_wsim_ls for x in sublist]

    res = stats.ks_2samp(log_wreal, log_wsim_pooled)
    real_vs_sim = res.statistic
    
    sim_vs_sim = []
    for k, log_wsim_k in enumerate(log_wsim_ls):
        log_wsim_nk = log_wsim_ls[:k] + log_wsim_ls[k+1:]
        log_wsim_nk_pooled = [x for sublist in log_wsim_nk for x in sublist]
        res = stats.ks_2samp(log_wsim_k, log_wsim_nk_pooled)
        sim_vs_sim.append(res.statistic)
    
    pvalue = np.sum(np.array(sim_vs_sim)>=real_vs_sim)/(len(A_sim)+1)
    return pvalue