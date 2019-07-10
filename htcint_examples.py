

import matplotlib
import matplotlib.pyplot as plt
plt.style.context('seaborn-colorblind')

import numpy          as np
import scipy.stats    as stats
import scipy.optimize as optimize

import htcore    as htcore
import htplot    as htplot



def plt_tmus(x0, rv, mu, mu0, size):

    ht   = htcore.htcomposite(rv, mu)
    tmu0 = ht.tmu(x0, mu, mu0)
    xs   = rv.rvs(mu, size = size)
    tmus = [ht.tmu(xi, mu, mu0) for xi in xs]
    plt.hist(tmus, 100, alpha = 0.5, label = r'$t_\mu$', density = True);
    plt.plot((tmu0, tmu0), (0., 1.), color = 'green', label = r'$t_\mu(x_0)$');
    plt.xlabel(r'$t_\mu$'); plt.legend(); plt.grid();
    print('p-value t0 :', np.sum(tmus >= tmu0)/(1.*size))
    xl, xu = np.min(xs[tmus <= tmu0]), np.max(xs[tmus <= tmu0])
    print('(xl, xu) with tmu <= tmu0 : (', xl, ', ', xu, ')')
    return
