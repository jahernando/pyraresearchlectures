

import matplotlib
import matplotlib.pyplot as plt
plt.style.context('seaborn-colorblind')

import numpy          as np
import scipy.stats    as stats
import scipy.optimize as optimize

import htcore    as htcore
import htplot    as htplot
import htcint    as htcint



def plt_tmus(x0, rv, mu, mu0, size):

    ht   = htcore.htcomposite(rv, mu)
    tmu0 = ht.tmu(x0, mu, mu0)
    xs   = rv.rvs(mu, size = size)
    tmus = [ht.tmu(xi, mu, mu0) for xi in xs]
    plt.hist(tmus, 100, alpha = 0.5, label = r'$g(t_\mu$ | \mu)', density = True);
    plt.plot((tmu0, tmu0), (0., 1.), color = 'green', label = r'$t_\mu(x_0)$');
    plt.xlabel(r'$t_\mu$'); plt.legend(); plt.grid();
    print('p-value t0 :', np.sum(tmus >= tmu0)/(1.*size))
    xl, xu = np.min(xs[tmus <= tmu0]), np.max(xs[tmus <= tmu0])
    print('(xl, xu) with tmu <= tmu0 : (', xl, ', ', xu, ')')
    return


def bayes_poisson_ci(b, x0, nmax = 10, nbins = 100):
    ns = np.linspace(0., nmax, nbins)
    dn = ns[1]-ns[0]
    ys = np.array([stats.gamma.pdf(b + ni, x0 +1) for ni in ns])
    norma = np.sum(ys*dn); ys = ys/norma
    print('integral :', np.sum(ys)*dn)
    plt.plot(ns, ys, color = 'black');
    betas  = [0.68, 0.90, 0.95]
    colors = ['red', 'green', 'yellow']
    for i, beta in enumerate(betas):
        nl, nu = htcint.ciarray.ci(ns, ys, beta)
        print('CI ',str(int(100.*beta)), '% CL : (', nl, ', ', nu, ')' )
        sel = (ns >= nl) & (ns <= nu)
        plt.fill_between(ns[sel], ys[sel], color = colors[i],
                         alpha = 0.5, label = str(int(100*beta))+'% CL');
    plt.legend(); plt.xlabel(r'$s$')
    plt.grid()
    return


def bayes_poisson_upper(b, x0, nmax = 10, nbins = 100):
    ns = np.linspace(0., nmax, nbins)
    dn = ns[1]-ns[0]
    ys = np.array([stats.gamma.pdf(b + ni, x0 +1) for ni in ns])
    norma = np.sum(ys*dn); ys = ys/norma
    print('integral :', np.sum(ys)*dn)
    plt.plot(ns, ys, color = 'black');
    betas  = [0.68, 0.90, 0.95]
    colors = ['red', 'green', 'yellow']
    for i, beta in enumerate(betas):
        nu = htcint.ciarray.upper(ns, ys, 1.- beta)
        print('Upper limit ',str(int(100.*beta)), '% CL :', nu )
        sel = (ns <= nu)
        plt.fill_between(ns[sel], ys[sel], color = colors[i],
                         alpha = 0.5, label = str(int(100*beta))+'% CL');
    plt.legend(); plt.xlabel(r'$s$')
    plt.grid()
    return
