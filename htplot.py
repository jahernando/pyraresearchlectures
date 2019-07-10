import matplotlib
import matplotlib.pyplot as plt
plt.style.context('seaborn-colorblind')

import numpy          as np
import scipy.stats    as stats
import scipy.optimize as optimize

xsize, ysize = 6 , 4.5
h0color, h1color, datacolor = 'orange', 'green', 'black'
marker = 'o'


def plt_htsimple(hs, x0 = None, nbins = 100):
    q0s, q1s = hs.q0s, hs.q1s
    qrange = hs.qrange()
    c, _ , _ = plt.hist(q0s, nbins, density = True, range = qrange, color = h0color,
             alpha = 0.8, label = r'$q(x|H_0)$', histtype = 'step');
    plt.hist(q1s, nbins, density = True, range = qrange, color = h1color ,
             alpha = 0.8, label = r'$q(x|H_1)$', histtype = 'step');
    if (x0 is not None):
        q0 = hs.q(x0)
        print('q0 ', q0)
        plt.plot((q0, q0), (0., np.max(c)), ls = '-', color = 'black');
        alpha, beta, CLs = hs.p0value(q0), hs.p1value(q0), hs.cls(q0)
        prt_hypotest(x0, alpha, beta, CLs)
    plt.xlabel('$q(x)$'); plt.ylabel('$g(q|x)$')
    plt.legend();
    return


def hist(vars, nbins, label=''):
    plt.hist(vars, 100, histtype = 'step', density = 'True');
    plt.xlabel(label);
    print(label,': ', np.mean(vars),'+-', np.std(vars));
    return


def chi2_test(chis, ndf = 1, nbins = 100, label = '', factor = 1.):
    plt.figure(figsize = (xsize, 2*ysize))
    plt.subplot(2, 1, 1)
    _, xbins, _ = plt.hist(chis, nbins, histtype = 'step', density = True);
    xbins = 0.5*(xbins[1:] + xbins[:-1])
    chi2 = stats.chi2(ndf)
    plt.plot(xbins, factor * chi2.pdf(xbins), color = 'black', label = r'$\chi^2$');
    plt.yscale('log'); plt.xlabel(label); plt.legend();
    print('chi2 ', factor * np.mean(chis)/(1.*ndf));
    plt.subplot(2, 1, 2)
    ps  = np.array([chi2.cdf(ichi) for ichi in chis])
    plt.hist(ps, nbins, histtype='step', density = True);
    plt.xlabel(r'$p$-value');
    print('p-value mean :', np.mean(ps),'; std * sqrt(12.) :', np.std(ps)*np.sqrt(12.))
    #plt.tight_layout();
    return


def prt_wilks_pvalues(beta, ci_central = None, ci_upper = None, p0 = None):
    if (ci_upper is not None):
        print('upper limit :', ci_upper  , 'at ', str(int(100*beta)), '% CL');
    if (ci_central is not None):
        print('central CI  :', ci_central, 'at ', str(int(100*beta)), '% CL');
    if (p0 is not None):
        print('p0 value    :', p0)
    return


def plt_wilks_pvalues(beta, mus, ptmus = None, pqmus = None):
    mu0, mu1 = min(mus), max(mus)
    if (ptmus is not None):
        plt.plot(mus, ptmus, label = r'$p$-value $t_\mu$');
    if (pqmus is not None):
        plt.plot(mus, pqmus, label = r'$p$-value $q_\mu$');
    plt.plot((mu0, mu1), (1-beta, 1-beta), color = 'black', ls='-.',
             label = str(int(100*beta))+'% CL');
    plt.xlabel(r'$\mu$'); plt.ylabel(r'$p$-value'); plt.legend(loc = 1); plt.grid();


def plt_hypotest(xs, h0s, h1s, x0 = None):
    plt.plot(xs, h0s, label = '$H_0$', color = h0color, alpha = 0.5);
    plt.plot(xs, h1s, label = '$H_1$', color = h1color, alpha = 0.5);
    if (x0 is not None):
        sel = xs <= x0
        plt.plot(x0, 0., marker='*', markersize = 12, color = 'black', label = r'$x_0$')
        plt.fill_between(xs[sel] , 0., h1s[sel] , color = h1color, alpha = 0.5)
        plt.fill_between(xs[~sel], 0., h0s[~sel], color = h0color, alpha = 0.5)
    plt.xlabel('x'); plt.legend(loc = 1)
    return


def plt_hypotest_bar(xs, h0s, h1s, x0):
    plt.bar(xs, h0s, label = '$H_0$', color = 'None',
            edgecolor = h0color, hatch='\\', alpha = 0.5);
    plt.bar(xs, h1s, label = '$H_1$', color = 'None',
            edgecolor = h1color, hatch='//', alpha = 0.5);
    if (x0 is not None):
        x0 = int(x0)
        sel = xs == x0; sel0 = xs <= x0; sel1 = xs >= x0
        plt.plot(x0, 0.5*(h0s[sel] + h1s[sel]), marker='*', markersize = 12,
                 color = 'black', label = r'$x_0$')
        plt.bar(xs[sel0], h1s[sel0], color = h1color, alpha = 0.2)
        plt.bar(xs[sel1], h0s[sel1], color = h0color, alpha = 0.2)
    plt.xlabel('x'); plt.legend(loc = 1)
    return

def prt_hypotest(xx, h0pval, h1pval, cls = None):
    print('observed data :', str(xx))
    print('H0 p-value    :', float(h0pval));
    print('H1 p-value    :', float(h1pval));
    if (cls is not None):
        print('CLs           :', float(cls))
    return
