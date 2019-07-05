import matplotlib
import matplotlib.pyplot as plt
plt.style.context('seaborn-colorblind')

import numpy          as np
import scipy.stats    as stats
import scipy.optimize as optimize

import htcore    as htcore
import htplot    as htplot

h0color, h1color, datacolor = 'orange', 'green', 'black'
marker = 'o'

def normal_likelihood(xs, ylog = False):

    def ll(x, xs):
        dx = (x-xs)
        return np.sum(dx*dx)

    nsize = len(xs)
    mus = np.linspace(-3., 3., 100)
    lls = np.array([ll(mu, xs) for mu in mus])

    print('mu mean :', np.mean(xs), ', mu std :', 1./np.sqrt(nsize))

    plt.hist(xs, bins=50, range=(-3., 3.),
            color = datacolor, histtype = 'step', label = 'data');
    plt.ylabel('events'); plt.xlabel('$x$')
    ax = plt.gca(); axb = ax.twinx()
    axb.plot(mus, lls, alpha=0.5, color='red', lw=2,
             label = r'$-2 \, \log \mathcal{L}(x | \mu)$')
    deltaL = np.min(lls) +1
    axb.plot(mus, deltaL*np.ones(len(mus)), ls='--',
             label = r'$\Delta \mathcal{L}(\mu) = 1$');
    axb.set_ylabel('$-2 \, \log \mathcal{L}(x | \mu)$')
    if (ylog): axb.set_yscale('log');
    plt.legend(loc = 1);

    return


def normal_posterior(xs):

    def posterior(x, xs):
        """ the posterior of a gaussian likelihood with known sigma and mu mean,
        and flat prior on mu
        is a gaussian with mean the average of the n-measurements and sigma=  sigma/sqrt(n)
        """
        mu = np.mean(xs)
        msig = np.sqrt(1./(1.*len(xs)))
        return stats.norm.pdf(x, mu, msig)

    mus = np.linspace(-3., 3., 100)
    dx  = mus[1]-mus[0]
    pos = np.array([posterior(mu, xs) for mu in mus])

    plt.hist(xs, bins=50, range=(-3., 3.), color= datacolor, histtype='step',
            label='data')
    print('posterior integral {:5.3f}'.format(np.sum(pos)*dx))
    ax = plt.gca(); axb = ax.twinx();
    axb.plot(mus, pos, color='red', lw=2, label="$\pi'(\mu)$");
    ax.set_xlabel('$\mu$'); axb.set_ylabel(r"$\pi'(\mu)$");
    plt.legend()

    return

def dice_posterior(xi, hpriors, faces = [4, 6, 12, 24]):

    def pxh(x, xface):
        if (x <= xface): return 1./(1.*xface)
        return 0.

    pxhs = np.array([pxh(xi, faces[i])*hpriors[i] for i in range(4)])
    px = np.sum(pxhs)

    hpos = pxhs/px
    hpos = hpos/np.sum(hpos)
    return hpos


def binomial_poisson(N, p):

    N, p = 100, 0.08
    ns = np.arange(20)
    plt.bar(ns, stats.binom.pmf(ns, N, p), color='blue', alpha=0.5, label='binomial')
    plt.bar(ns, stats.poisson.pmf(ns, N*p), color='red', alpha=0.5, label='poisson')
    plt.xlabel('$x$')
    plt.set_ylabel('$f(x$)', fontsize=16)
    plt.legend(fontsize=14)
    return
