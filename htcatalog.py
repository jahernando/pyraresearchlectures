import numpy          as np
import scipy.stats    as stats
import scipy.optimize as optimize

class poisson_control_signal:


    def rvs(mu = None, nu = None, tau = None, size = 1):
        #mu, nu, tau = self._par(mu, nu, tau)
        ns = [float(stats.poisson.rvs(tau * nu)) for i in range(size)]
        ms = [float(stats.poisson.rvs(nu  + mu)) for i in range(size)]
        vals = list(zip(ns, ms))
        #print('rvs', vals)
        return vals

    def logpdf(x, mu = None, nu = None, tau = None):
        #mu, nu, tau = self._par(mu, nu, tau)
        ns, ms = x[0], x[1]
        pns = stats.poisson.logpmf(ns, tau * nu)
        pms = stats.poisson.logpmf(ms, nu  + mu)
        _ll = np.sum(pns) + np.sum(pms)
        #print('llike ', _ll)
        return _ll

def _vals(xs, ns):
    vals = [[xi[j] for xi in xs] for j in range(ns)]
    return vals

def _xs(vals, ns, size):
    xs = [[vals[i][j] for i in range(ns)] for j in range(size)]
    return xs


class poisson_ncounter:

    def __init__(self, ss, bs = None, mu = 1.):
        nbins = len(ss)
        bs = bs if bs is not None else np.zeros(nbins)
        self.par     = np.array([mu,])
        self.nbins   = len(ss)
        self.ss      = np.array(ss)
        self.bs      = np.array(bs)

    def _par(self, mu):
        mu = np.array(mu) if mu is not None else self.par
        #print('_mu', mu)
        return mu

    def rvs(self, mu = None, size = 1):
        mu = self._par(mu)
        vals = [stats.poisson.rvs(bi + mu * si, size = size)
                for bi, si in zip(self.bs, self.ss)]
        ns = _xs(vals, self.nbins, size)
        #print('rvs ', ns)
        return ns

    def logpdf(self, x, mu = None):
        mu = self._par(mu)
        pms = [stats.poisson.logpmf(xi, bi + mu * si)
             for bi, si, xi in zip(self.bs, self.ss, x)]
        ll = np.sum(np.array(pms))
        #print('llike ', ll)
        return ll


class extended_norm_uniform:


    def rvs(nug, nue, tau, xmu, sig, size = 1):
        #print(nug, nue, tau, xmu, sig, size)
        #nug, nue, tau, xmu, sig = self._par(nug, nue, tau, xmu, sig)
        def _rvs():
            ne  = stats.poisson.rvs(nue, size = 1)
            ng  = stats.poisson.rvs(nug, size = 1)
            xts = stats.uniform.rvs(0. , tau, size = int(ne))
            xgs = stats.norm   .rvs(xmu, sig, size = int(ng))
            return np.array(list(xts) + list(xgs))
        vals = [_rvs() for i in range(size)]
        #print('rvs ', vals)
        return vals

    def logpdf(x, nug, nue, tau, xmu, sig):
        #nug, nue, tau, xmu, sig = self._par(nug, nue, tau, xmu, sig)
        fe = 1.*nue/float(nue + nug)
        fg = 1. - fe
        #print('fe, fg ', fe, fg)
        if ((nue < 0) or (nug < 0)): return 1e-320
        def _px(xi):
            val =  fe    * stats.uniform.pdf(xi, 0., tau) + (1-fe) * stats.norm  .pdf(xi, xmu, sig)
            return val
        lpx = np.sum(np.log(_px(x)))
        nn = 1.*len(x)
        lpn = stats.poisson.logpmf(nn, 1.*nue + 1.*nug)
        #print('llike ', lpx, lpn)
        return lpx + lpn


class extended_norm_expon:

    def rvs(nug, nue, tau, xmu, sig, size = 1):
        #nug, nue, tau, xmu, sig = self._par(nug, nue, tau, xmu, sig)
        def _rvs():
            ne  = stats.poisson.rvs(nue, size = 1)
            ng  = stats.poisson.rvs(nug, size = 1)
            xts = stats.expon  .rvs(tau, size = int(ne))
            xgs = stats.norm   .rvs(xmu, sig, size = int(ng))
            return np.array(list(xts) + list(xgs))
        vals = [_rvs() for i in range(size)]
        #print('rvs ', vals)
        return vals

    def logpdf(x, nug = None, nue = None, tau = None,
               xmu = None, sig = None):
        #nug, nue, tau, xmu, sig = self._par(nug, nue, tau, xmu, sig)
        fe = 1.*nue/float(nue + nug)
        fg = 1. - fe
        #print('fe, fg ', fe, fg)
        if ((nue < 0) or (nug < 0)): return 1e-320
        def _px(xi):
            val =  fe    * stats.expon.pdf(xi, tau) + (1-fe) * stats.norm  .pdf(xi, xmu, sig)
            return val
        lpx = np.sum(np.log(_px(x)))
        nn = 1.*len(x)
        lpn = stats.poisson.logpmf(nn, 1.*nue + 1.*nug)
        #print('llike ', lpx, lpn)
        return lpx + lpn


#
# class ExtExp2Gaus(HypoTestComp):
#
#     def __init__(self, nue, nug, nug2, tau, mug, sigma, mug2, sigma2):
#         self.parameter = np.array([nug, nue, nug2])
#         self.gen_nue   = stats.poisson(nue)
#         self.gen_exp   = stats.expon(tau)
#         self.gen_nug   = stats.poisson(nug)
#         self.gen_gau   = stats.norm(mug, sigma)
#         self.gen_nug2  = stats.poisson(nug2)
#         self.gen_gau2  = stats.norm(mug2, sigma2)
#         self.tau       = tau
#         self.mug       = mug
#         self.sigma     = sigma
#         self.mug2      = mug
#         self.sigma2    = sigma
#
#     def rvs(self, msize = 1):
#         def _rvs():
#             ne  = self.gen_nue .rvs()
#             ng  = self.gen_nug .rvs()
#             ng2 = self.gen_nug2.rvs()
#             xts = self.gen_exp.rvs(int(ne))
#             xgs = self.gen_gau.rvs(int(ng))
#             xg2s = self.gen_gau2.rvs(int(ng2))
#             return np.array(list(xts) + list(xgs) + list(xg2s))
#         return [_rvs() for i in range(msize)]
#
#     def ll(self, x, par = None):
#         ng, ne, ng2 = par[0], par[1], par[2]
#         fe  = (1.* ne)/float(ne + ng + ng2)
#         fg  = (1 * ng)/float(ne + ng + ng2)
#         fg2 = 1. - fe - fg
#         if ((ne < 0) or (ng <0)): return 1e-320
#         def _px(xi):
#             return fe * self.gen_exp.pdf(xi) + fg * self.gen_gau.pdf(xi) + fg2 * self.gen_gau2.pdf(xi)
#         lpx = float(np.sum([np.log(_px(xi)) for xi in x]))
#         nn = len(x)
#         lpn = float(stats.poisson(ne + ng + ng2).logpmf(nn))
#         return lpx + lpn
#
#     def par0(sefl, x):
#         return self.parameter
#
#
# class ExtUniGaus(HypoTestComp):
#
#     def __init__(self, nue, nug, tau, mug, sigma):
#         self.parameter = np.array([nug, nue])
#         self.gen_nue   = stats.poisson(nue)
#         self.gen_uni   = stats.uniform(0, tau)
#         self.gen_nug   = stats.poisson(nug)
#         self.gen_gau   = stats.norm(mug, sigma)
#         self.tau       = tau
#         self.mug       = mug
#         self.fe        = 1.*nue/(1.*(nue + nug))
#         self.fg        = 1.*nug/(1.*(nue + nug))
#         self.sigma     = sigma
#
#     def rvs(self, msize = 1):
#         def _rvs():
#             ne  = self.gen_nue.rvs()
#             ng  = self.gen_nug.rvs()
#             xts = self.gen_uni.rvs(int(ne))
#             xgs = self.gen_gau.rvs(int(ng))
#             return np.array(list(xts) + list(xgs))
#         return [_rvs() for i in range(msize)]
#
#     def ll(self, x, par = None):
#         par = self.parameter if par is None else par
#         ng, ne = float(par[0]), float(par[1])
#         fe = ne/(ne + ng)
#         fg = 1. - fe
#         #if ((ne < 0) or (ng < 0)): return np.log(1e-320)
#         def _px(xi):
#             return fe * self.gen_uni.pdf(xi) + fg * self.gen_gau.pdf(xi)
#
#         lpx = float(np.sum([np.log(_px(xi)) for xi in x]))
#         nn = len(x)
#         lpn = float(stats.poisson(ne + ng).logpmf(nn))
#         return lpx + lpn
#
#     def par0(sefl, x):
#         return self.parameter
#
#
# class ExtUni2Gaus(HypoTestComp):
#
#     def __init__(self, nue, nug, nug2, tau, mug, sigma, mug2, sigma2):
#         self.parameter = np.array([nug, nue, nug2])
#         self.gen_nue   = stats.poisson(nue)
#         self.gen_uni   = stats.uniform(0, tau)
#         self.gen_nug   = stats.poisson(nug)
#         self.gen_gau   = stats.norm(mug, sigma)
#         self.gen_nug2  = stats.poisson(nug2)
#         self.gen_gau2  = stats.norm(mug2, sigma2)
#         self.tau       = tau
#
#     def rvs(self, msize = 1):
#         def _rvs():
#             ne  = self.gen_nue.rvs()
#             xts = self.gen_uni.rvs(int(ne))
#             ng  = self.gen_nug.rvs()
#             xgs = self.gen_gau.rvs(int(ng))
#             ng2  = self.gen_nug2.rvs()
#             xg2s = self.gen_gau2.rvs(int(ng2))
#             return np.array(list(xts) + list(xgs) + list(xg2s))
#         return [_rvs() for i in range(msize)]
#
#     def ll(self, x, par = None):
#         par = self.parameter if par is None else par
#         ng, ne, ng2 = float(par[0]), float(par[1]), float(par[2])
#         fe  = ne/(ne + ng + ng2)
#         fg  = ng/(ne + ng + ng2)
#         fg2 = ng2/(ne + ng + ng2)
#         def _px(xi):
#             return fe * self.gen_uni.pdf(xi) + fg * self.gen_gau.pdf(xi) + fg2 * self.gen_gau2.pdf(xi)
#
#         #lpx = float(np.sum([np.log(_px(xi)) for xi in x]))
#         lpx = np.sum(np.log(_px(x)))
#         nn = len(x)
#         lpn = float(stats.poisson(ne + ng + ng2).logpmf(nn))
#         return lpx + lpn
#
#     def par0(sefl, x):
#         return self.parameter
