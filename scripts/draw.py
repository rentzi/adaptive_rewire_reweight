"""
Provide functions for plotting and visualizing simulation results.
"""
import numpy as np
from scipy import stats
import networkx as nx

def draw_network_state(ax, G, nodes, pos, states, vmin = 0, vmax = 1, cmap = "Reds"):
    """
    Plot a network with node colors representing node states.
    args:
        ax: Matplotlib Axes objecton which the network is plotted.
        G: Network to be plotted.
        nodes: List of node labels.
        pos: Node positions.
        states: Node states used to determine node colors.
        vmin: Minimum value of the color scale.
        vmax: Maximum value of the color scale.
        cmap: Colormap used for node colors.
    """
    node_values = [states[n] for n in nodes]

    nx.draw_networkx_edges(
        G, pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=16,
        edge_color="black",
        width=1.2
    )

    nx.draw_networkx_nodes(
        G, pos,
        ax=ax,
        nodelist=nodes,
        node_color=node_values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        node_size=350,
        edgecolors="black",
        linewidths=1.2
    )

    nx.draw_networkx_labels(
        G, pos,
        ax=ax,
        labels={n: str(n) for n in nodes},
        font_size=10,
        font_color="black"
    )

    ax.set_aspect("equal")
    ax.axis("off")

def hist_log_weight(ax, weights, num_bins, color):
    """
    Plot the histogram of log-transformed connection weights.
    args:
        ax: Matplotlib Axes object.
        weights: Connection weights.
        num_bins: Number of points used to define the histogram bins.
        color: Color of the histogram.
    returns:
        x: Bin edges used for the histogram.
        mu: Location parameter of the fitted normal distribution.
        sig: Scale parameter of the fitted normal distribution.
    """
    log_w = np.log(weights)
    mu, sig = stats.norm.fit(log_w)
    lb = np.min([mu-3*sig, np.min(log_w)])
    ub = np.max([mu+3*sig, np.max(log_w)])
    x = np.linspace(lb, ub, num_bins)
    ax.hist(log_w, bins=x, density=True, color = color, alpha=0.7)
    ax.set_xlabel('ln(weights)',fontsize=8)
    ax.tick_params(axis = 'both', labelsize=8)
    return x, mu, sig

def draw_power_law(ax, weights, pl_fit, tau, num_bins, color, draw_fit):
    """
    Plot the right tail of the weight distribution and its power-law fit
    on logarithmic axes.
    args:
        ax: Matplotlib Axes object.
        weights: Connection weights.
        pl_fit: Parameters of the power-law fit.
        tau: Reweighting time constant associated with the weights.
        num_bins: Number of bins used to construct the logarithmic histogram.
        color: Color used for the empirical distribution.
        draw_fit: Whether to plot the fitted power-law relationship.
    """
    wtrunc = weights[weights>=np.quantile(weights, 0.5)]
    lb = np.min(np.log10(wtrunc)); ub = np.max(np.log10(wtrunc))
    bins = np.logspace(lb, ub, num_bins*2 + 1)
    v, _ = np.histogram(wtrunc, bins[::2], density = True)
    x = bins[1::2]
    x = x[v>0]; v = v[v>0]
    ax.plot(x, v*len(wtrunc)/len(weights), color = color,alpha=0.7, label = str(tau))
    if draw_fit == True:
        y = np.power(10, pl_fit['intercept'] + pl_fit['slope'] * np.log10(x))
        x = x[y>0]; y = y[y>0]
        ax.plot(x, y*len(wtrunc)/len(weights), color = 'k', linestyle='dashed',alpha=1)

def draw_mean_scatter(ax, X, Y, color, label = None, linestyle = '-', shift = 0.01, jitter = 0.003):
    """
    Plot mean values across parameter settings together with individual values.
    args:
        ax: Matplotlib Axes object.
        X: Positions of the parameter values on the x-axis.
        Y: Dictionary mapping each parameter setting to values obtained from
            individual networks.
        color: Color of the line and scatter points.
        label: Label of the mean-value line.
        linestyle: Line style of the mean-value line.
        shift: Horizontal shift applied to the scatter points.
        jitter: Standard deviation of the random horizontal jitter applied
            to the scatter points.
    """
    for i, key in enumerate(Y.keys()):
        K = len(Y[key])
        x_jittered = np.repeat(X[i], K) - shift + np.random.normal(0, jitter, size=K)
        ax.scatter(x_jittered, Y[key], s = 5, facecolors='none', edgecolor = color, alpha = 0.35, zorder=1)
    ax.plot(X, [np.nanmean(Y[key]) for key in Y.keys()], color = color, 
            label = label, alpha = 0.8, ls = linestyle, linewidth=1.8, zorder=3)

def forward(x):
    """
    Apply a square-root transformation to x-axis values.
    """
    return x**(1/2)

def inverse(x):
    """
    Apply the inverse square transformation to x-axis positions.
    """
    return x**2

def draw_KS_stats(ax, ks_stats, wcx_p, sig_thresh, xmin, xmax):
    """
    Plot violin plots of KS statistics of all candidate distributions.
    args:
        ax: Matplotlib Axes object.
        ks_stats: KS statistics for the candidate distributions.
        wcx_p: P-values from pairwise Wilcoxon signed-rank tests.
        sig_thresh: Significance threshold.
        xmin: Lower bound of the x-axis.
        xmax: Upper bound of the x-axis.
    """
    dist_ls = ['Expon', 'InvNormal', 'Normal', 'Gamma', 'InvGamma', 'Lognorm', 'Weibull'] # candidate distributions

    x = np.mean(ks_stats, 0)
    L = np.where(x==np.min(x))[0][0]
    X = ks_stats[:, L]
    
    all_data = [ks_stats[:,l] for l in range(len(dist_ls))]
    plots = ax.violinplot(all_data, showmeans=True, showmedians=False, orientation='horizontal')
    pc = plots['bodies'][L]
    
    pc.set_facecolor('r')
    pc.set_alpha(0.7)
    for l, dist in enumerate(dist_ls):
        if l != L:
            Y = ks_stats[:, l]
            res_p = wcx_p[dist_ls[L]][dist]
            if res_p > sig_thresh:
                pc = plots['bodies'][l]
                pc.set_color('m')
                pc.set_alpha(0.5)
    
    ax.axvline(x = x[L], color = 'k', ls = '--', lw = 0.5)
    ax.set_xlim([xmin, xmax])
    ax.set_xscale('function', functions=(forward, inverse))
    ax.set_yticks([l + 1 for l, dist in enumerate(dist_ls)], labels = [])
