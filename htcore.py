import numpy          as np
import scipy.stats    as stats
import scipy.optimize as optimize


def _llike(rv):
    meths = dir(rv)
    if 'logpdf' in meths: rv.llike = rv.logpdf
    if 'logpmf' in meths: rv.llike = rv.logpmf
    if 'llike' not in dir(rv): print('not llike method')
    return rv.llike

def _masks(size, mask = None, masknu = None):
    mask   = np.array(mask)   if mask   is not None else np.ones (size, dtype = bool)
    masknu = np.array(masknu) if masknu is not None else np.zeros(size, dtype = bool)
    maskmu = np.logical_and(mask, ~masknu)
    #print('_masks', mask, maskmu, masknu)
    return mask, maskmu, masknu

def _par(par, mask):
    ps = par[mask] if par.size > 1 else par
    # print('_par ', ps)
    return np.array(ps)

def _setpar(par, par0, mask):
    if (par.size == 1): return np.array([float(par0),])
    p = np.array(par)
    p[mask] = par0
    # print('_setpar ', p)
    return p

def mle(x, llike, par, mask = None):
    #print('mle par, size', par, par.size, mask)
    mask = mask if mask is not None else np.ones(par.size, dtype = bool)
    ps = _par(par, mask)
    def _mll(ps):
        ms = _setpar(par, ps, mask)
        #print('xs, ms ', x, *ms)
        return np.sum(-2. * llike(x, *ms))
    result = optimize.minimize(_mll, ps, method='Nelder-Mead')
    if (not result.success): print('mle: warning')
    #print('mle ', result)
    return result.x

def parbest(x, llike, par, mask = None):
    muhat  = mle(x, llike, par, mask = mask)
    pbest  = _setpar(par, muhat, mask)
    #print('parbest ', pbest);
    return pbest


def qtest(x, llike0, llike1, par0 = None, par1 = None):
    ll0s = llike0(x) if par0 is None else llike0(x, *par0)
    ll1s = llike1(x) if par1 is None else llike1(x, *par1)
    val = 2*(ll1s - ll0s)
    #print('qtest', val)
    return val


def tmu(x, llike, parmu, parbest):
    lbest = np.sum(llike(x, *parbest))
    lmu   = np.sum(llike(x, *parmu))
    tm    = 2*(lbest - lmu)
    #print('tmu : lbest ', lbest, ', lmu ', lmu)
    #print('tmu :', tm)
    return tm


def tmu_pvalue(tmu):
    z0 = np.sqrt(tmu)
    p0 = 2.*(1. - stats.norm(0., 1.).cdf(z0))
    return p0

def qmu_pvalue(qmu):
    z0 = np.sqrt(qmu)
    p0 = 1. - stats.norm(0., 1.).cdf(z0)
    return p0

def q0_pvalue(q0):
    return qmu_pvalue(q0)


class htsimple:


    def __init__(self, rv0, rv1, size):
        self.rv0  = rv0
        self.rv1  = rv1
        self.size = int(size)
        self.llike0 = _llike(rv0)
        self.llike1 = _llike(rv1)
        x0s = self.rv0.rvs(size = size)
        x1s = self.rv1.rvs(size = size)
        self.x0s = x0s
        self.x1s = x1s
        self.q0s = [self.q(xi) for xi in x0s]
        self.q1s = [self.q(xi) for xi in x1s]
        #print(x0s)
        #print(self.q0s)
        return


    def q(self, x):
        return qtest(x, self.llike0, self.llike1)


    def qrange(self):
        return (np.min(self.q0s), np.max(self.q1s))


    def p0value(self, q):
        nsel = np.sum(self.q0s >= q)
        return 1.*nsel/(1.*len(self.q0s))


    def p1value(self, q):
        nsel = np.sum(self.q1s <= q)
        return 1.*nsel/(1.*len(self.q1s))


    def cls(self, q):
        beta0 = 1.*np.sum(self.q0s <= q)/(1.*len(self.q0s))
        beta1 = 1.*np.sum(self.q1s <= q)/(1.*len(self.q1s))
        return beta1/beta0


class htcomposite:


    def __init__(self, rv, par, mask = None, masknu = None):
        self.rvs    = rv.rvs
        self.llike  = _llike(rv)
        self.par    = np.array(par)
        if (self.par.size == 1): self.par = np.array([par,])
        mask, maskmu, masknu = _masks(self.par.size, mask, masknu)
        self.maskmu   = maskmu
        self.masknu   = masknu
        self.mask     = mask

    def _has_nus(self):
        return (np.sum(self.masknu) > 0)


    def mubest(self, x, mu0 = None, par0 = None):
        bp  = self.parbest(x, mu0, par0)
        res = _par(bp, self.maskmu)
        #print('mubest ', res)
        return res


    def parbest(self, x, mu0 = None, par = None):
        par = self.par if par is None else par
        muhat  = mle(x, self.llike, par, mask = self.mask)
        pbest  = _setpar(par, muhat, self.mask)
        if (mu0 is not None):
            if (_par(pbest, self.maskmu) < mu0):
                pbest = self.parmubest(x, mu0, par)
        #print('parbest :', pbest)
        return pbest


    def parmubest(self, x, mu, par = None):
        par = self.par if par is None else par
        par = _setpar(par, mu, self.maskmu)
        pbest = par
        if (self._has_nus()):
            nuhat = mle(x, self.llike, par, self.masknu)
            pbest = _setpar(par, nuhat, self.masknu)
        #print('parmubest :', pbest)
        return pbest


    def tmu(self, x, mu, mu0 = None, par = None, parbest = None, parmubest = None):
        if (parbest is None):
            parbest   = self.parbest  (x, mu0, par)
        if (parmubest is None):
            parmubest = self.parmubest(x, mu, par)
        res = tmu(x, self.llike, parmubest, parbest)
        #print('tmu :', res)
        return res


    def tmu_rvs(self, mu = None, par = None, size = 1000):
        mu      = mu  if mu  is not None else _par(self.par, self.maskmu)
        parmu   = par if par is not None else self.par
        parmu = _setpar(parmu, mu, self.maskmu)
        xs   = [self.rvs(*parmu)[0] for i in range(size)]
        tmus = np.array([self.tmu(xi, mu) for xi in xs])
        return tmus


    def tmu_pvalue_rvs(self, tmu, tmus = None, mu = None, par = None, size = 1000):
        tmus = self.tmus_rvs(x, mu, par, size) if tmus is None else tmus
        #tmu0 = self.tmu(x, mu)
        pmu = (1.*np.sum(tmus >= tmu))/(1.*size)
        #print('tmu pval rvvs: ', pmu0, tmu0, np.mean(tmus))
        return pmu


    def qmu(self, x, mu, par = None, parbest = None, parmubest = None):
        if (parbest is None):
            parbest   = self.parbest  (x, par = par)
        res = 0.
        if (_par(parbest, self.maskmu) < mu):
            res = self.tmu(x, mu, par = par, parbest = parbest, parmubest = parmubest)
        #print('qmu :', res)
        return res


    def q0(self, x, mu0, par = None, parbest = None, parmubest = None):
        if (parbest is None):
            parbest   = self.parbest  (x, par = par)
        res = 0.
        if (_par(parbest, self.maskmu) > mu0):
            res =  self.tmu(x, mu0, par = par, parbest = parbest, parmubest = parmubest)
        #print('q0 :', res)
        return res


    def tmu_cint(self, x, par = None, parbest = None, beta = 0.68):
        par = par if par is not None else self.par
        parbest = self.parbest(x, par = par) if parbest is None else parbest
        mubest  = _par(parbest, self.maskmu)
        mu0l, mu0u = mubest - 0.1*abs(mubest), mubest + 0.1*abs(mubest)
        xl = self._root(x, mu0l, parbest, self.tmu, tmu_pvalue, beta)
        xu = self._root(x, mu0u, parbest, self.tmu, tmu_pvalue, beta)
        res = np.array((float(xl), float(xu)))
        #print('tmu_cint: ', res)
        return res


    def qmu_ulim(self, x, par = None, parbest = None, beta = 0.9):
        par     = par if par is not None else self.par
        parbest = self.parbest(x, par = par) if parbest is None else parbest
        mubest  = _par(parbest, self.maskmu)
        mu0     = mubest + 0.1*abs(mubest)
        ulim    = self._root(x, mu0, parbest, self.qmu, qmu_pvalue, beta)
        #print('qmu_ulim: ', ulim)
        return ulim

    def _root(self, x, mu0, par, tvar, pvar, beta):
        ms = _setpar(par, mu0, self.maskmu)
        ps = _par(ms, self.maskmu)
        def _root(ps):
            ms = _setpar(par, ps, self.maskmu)
            tm = tvar(x, ps)
            pm = pvar(tm) -1. + beta
            return pm
        res = optimize.root(_root, ps)
        if (not res.success): print('_root: warning!')
        #print('_root: ', res)
        return res.x
