import matplotlib
import matplotlib.pyplot as plt
plt.style.context('seaborn-colorblind')

import numpy          as np
import scipy.stats    as stats
import scipy.optimize as optimize

import htcore     as htcore
import htcatalog  as htcata
import htplot     as htplot


def poisson_control_muhat_hist(mu, nu, tau, size):

    par, mask, masknu  = (mu, nu, tau), (True, True, False), (False, True, False)

    # generate an observation and obtain the best-estimate of mu, nu and nu given mu
    rv      = htcata.poisson_control_signal
    xs      = rv.rvs(*par, size = size)

    ht       = htcore.htcomposite(rv, par, mask = mask, masknu = masknu)
    parbests = [ht.parbest(x0) for x0 in xs]
    mubests, nubests = [par[0] for par in parbests], [par[1] for par in parbests]

    plt.hist(mubests, 80, (0, 2*mu), histtype='step', density = True, label = r'$\mu$');
    plt.hist(nubests, 80, (0, 2*mu), histtype='step', density = True, label = r'$\nu$');
    plt.legend(loc = 1);
    return

def poisson_control_ci(mu, nu, tau):
    par, mask, masknu  = (mu, nu, tau), (True, True, False), (False, True, False)

    # generate an observation and obtain the best-estimate of mu, nu and nu given mu
    rv      = htcata.poisson_control_signal
    x0      = rv.rvs(*par, size =1)[0]
    ht      = htcore.htcomposite(rv, par, mask = mask, masknu = masknu)
    parbest = ht.parbest(x0)
    mubest, nubest = parbest[0], parbest[1]
    nus     = np.linspace(max(1, nubest - nu), nubest + nu, 40)
    mus     = np.linspace(max(1, mubest - mu), mubest + mu, 40)
    parmubests = [ht.parmubest(x0, mu = mui) for mui in mus]
    nusbests = [parmui[1] for parmui in parmubests]
    chi2    = stats.chi2(2)
    def pnus(mui, nui):
        pari = np.array(par)
        pari[0], pari[1] = mui, nui
        tmui = htcore.tmu(x0, ht.llike, parmu = pari, parbest = parbest);
        pmu  = 1.- chi2.cdf(tmui)
        return pmu
    xpnus = [[pnus(mui, nui) for mui in mus] for nui in nus]

    print('data     :', x0)
    print('best par :', parbest[:2])
    plt.plot(parbest[0], parbest[1], marker='*', markersize = 12, color = 'black', label = r'$(\hat{\mu}, \hat{\nu})$')
    plt.contour(mus, nus, xpnus, levels = 10, cmap = 'jet', alpha = 0.8,);
    plt.plot(mus, nusbests, color = 'black', label = r'$\hat{\nu}(\mu)$');
    plt.grid();
    plt.xlabel(r'$\mu$'); plt.ylabel(r'$\nu$'); plt.legend();
    plt.colorbar(label = 'p-value');

    ci_central = ht.tmu_cint(x0, beta = 0.9, parbest = parbest);
    print('central CI :', ci_central);
    return
