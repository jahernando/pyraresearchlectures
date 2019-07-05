import matplotlib
import matplotlib.pyplot as plt
plt.style.context('seaborn-colorblind')

import numpy          as np
import scipy.stats    as stats
import scipy.optimize as optimize

import htcore    as htcore
import htplot    as htplot

h0color, h1color, datacolor = 'orange', 'green', 'black'
#
# def plt_hypotest(xs, h0s, h1s, x0 = None):
#     plt.plot(xs, h0s, label = '$H_0$', color = h0color, alpha = 0.5);
#     plt.plot(xs, h1s, label = '$H_1$', color = h1color, alpha = 0.5);
#     if (x0 is not None):
#         sel = xs <= x0
#         plt.plot(x0, 0., marker='*', markersize = 12, color = 'black', label = r'$x_0$')
#         plt.fill_between(xs[sel] , 0., h1s[sel] , color = h1color, alpha = 0.5)
#         plt.fill_between(xs[~sel], 0., h0s[~sel], color = h0color, alpha = 0.5)
#     plt.xlabel('x'); plt.legend(loc = 1)
#     return
#
#
# def plt_hypotest_bar(xs, h0s, h1s, x0):
#     plt.bar(xs, h0s, label = '$H_0$', color = 'None',
#             edgecolor = h0color, hatch='\\', alpha = 0.5);
#     plt.bar(xs, h1s, label = '$H_1$', color = 'None',
#             edgecolor = h1color, hatch='//', alpha = 0.5);
#     if (x0 is not None):
#         x0 = int(x0)
#         sel = xs == x0; sel0 = xs <= x0; sel1 = xs >= x0
#         plt.plot(x0, 0.5*(h0s[sel] + h1s[sel]), marker='*', markersize = 12,
#                  color = 'black', label = r'$x_0$')
#         plt.bar(xs[sel0], h1s[sel0], color = h1color, alpha = 0.2)
#         plt.bar(xs[sel1], h0s[sel1], color = h0color, alpha = 0.2)
#     plt.xlabel('x'); plt.legend(loc = 1)
#     return
#
# def prt_hypotest(xx, h0pval, h1pval, cls = None):
#     print('observed data :', float(xx))
#     print('H0 p-value    :', float(h0pval));
#     print('H1 p-value    :', float(h1pval));
#     if (cls is not None):
#         print('CLs           :', float(cls))
#     return

def gaussian(mu0, mu1, x0 = '', mutrue = False, nmus = 200, sigma0 = 1., sigma1 = 1.):
    h0, h1 = stats.norm(mu0, sigma0), stats.norm(mu1, sigma1)
    xlow, xupp = mu0-5*sigma0, mu1+5*sigma1
    xs = np.linspace(xlow, xupp, nmus)
    if (x0 == ''):
        x0 = h1.rvs(1) if mutrue else h0.rvs(1)
    h0s, h1s = h0.pdf(xs), h1.pdf(xs)
    htplot.plt_hypotest(xs, h0s, h1s, x0 = x0)
    if (x0 is None): return
    h0pval = 1-h0.cdf(x0)
    h1pval = h1.cdf(x0)
    cls    = h1pval/(h0.cdf(x0))
    htplot.prt_hypotest(x0, h0pval, h1pval, cls);
    return

def poisson(mu0, mu1, x0 = '', mutrue = False, nmus = 200):
    h0, h1 = stats.poisson(mu0), stats.poisson(mu1)
    nlow, nupp = max(0, int(mu0-5*np.sqrt(mu0))), int(mu1+5*np.sqrt(mu1))
    xs = np.array(range(nlow, nupp+1))
    h0s, h1s = h0.pmf(xs), h1.pmf(xs)
    if (x0 == ''):
        x0 = h1.rvs(1) if mutrue else h0.rvs(1)
    htplot.plt_hypotest_bar(xs, h0s, h1s, x0 = x0)
    if (x0 is None): return
    x0 = int(x0)
    h0pval = 1-h0.cdf(x0) + h0.pmf(x0)
    h1pval = h1.cdf(x0)
    cls    = h1pval/(h0.cdf(x0))
    htplot.prt_hypotest(x0, h0pval, h1pval, cls);
    return


def poisson_control_rvs(mu, nu, tau, nbins = None):
    nbins = tau + 1 if nbins is None else nbins
    ni = stats.poisson((tau+1) * nu).rvs()
    mi = stats.poisson(mu).rvs()
    print('bkg events :', ni, 'signal events :', mi)
    xns = (tau+1.)*(stats.uniform().rvs(ni)-0.5)
    xms =           stats.uniform().rvs(mi)-0.5
    data = list(xms)+list(xns)

    zrange = (-0.5*(tau+1), 0.5*(tau+1))
    zbins = np.linspace(-0.5*(tau+1.), 0.5*(tau+1), nbins+1)
    ycounts, xbins, _ = plt.hist(data, zbins, histtype='step',
                                 color = 'black', alpha = 0.5);
    xcenters = 0.5*(xbins[1:] + xbins[:-1])
    plt.plot(xcenters, ycounts, 'o', color = 'black', label = 'data');

    # plot pdfs
    mbins = 1000
    zbins = np.linspace(-0.5*(tau+1.), 0.5*(tau+1), mbins)
    h0s  = nu*(tau+1)/(1.*nbins) * np.ones(mbins)
    h1s  = mu*(tau+1)/(1.*nbins) * np.ones(mbins)
    zsel = ((zbins >= -0.5) & (zbins < 0.5))
    h1s[~zsel] = 0.
    hts = h0s + h1s
    plt.plot(zbins, h0s, color = h0color, ls='--', label = 'bkg')
    plt.plot(zbins, h1s, color = h1color, ls='-.', label = 'signal')
    plt.plot(zbins, hts, color = h1color, ls='--', label = 'total');
    plt.xlabel('x'); plt.legend();
    return


def npoisson_rvs(bs, ss, mu = 1.):
    bb, ss = np.array(bs), np.array(ss)
    nbins = len(bs)
    ms = [stats.poisson(bi + mu*si).rvs() for bi, si in zip(bs, ss)]
    zs = np.arange(nbins)
    plt.bar (zs, bs, color = h0color, hatch='\\',
             alpha = 0.4, label=r'b')
    plt.bar (zs, bb + mu*ss, color =  h1color, hatch='//',
             alpha = 0.4, label=r'b+$\mu$s')
    plt.plot(zs, ms, marker = 'o', ls='none', color = datacolor, label = 'data',
             markersize = 12);
    plt.xlabel('bins'); plt.grid(); plt.legend()
    return
