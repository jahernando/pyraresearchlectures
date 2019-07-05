# functions to operate with FC

import numpy as np
import scipy.stats as stats


type = 'classical'

def rmu(mu, b, beta, full_output = False, type = type):
    """ return the mu-region for classical belt for poisson case
    inputs:
        mu: the signal strength
        b: the background level
        beta: the confidence level of the interval [0., 1.]
        ndim: maximun value of n explored (default ndim)
        type: 'classical', 'fc'
        full_output: add full output
    return:
        result: tuple (int, int) with the Rmu segment
        if (full_output) is True:
        result:
            tuple (int, int) with the Rmu segment
            tuple [(tmu, px, x)] orderted by tmu with the px and x-value
    """
    if (type == 'fc'):
        return _rmu_fc(mu, b, beta, full_output)
    return _rmu_classical(mu, b, beta, full_output)


def ci_belt(mus, b, beta, type = type):
    """ creates the FC belt for a given beta for poisson case
    inputs:
        mus: array with the list os mus
        b: background level
        beta: CL level [0., 1]
        ndim: maximum number of observables (default ndim)
    returns:
        tuple with three list: mus, x0s (lower value of the intervals),
        x1s (upper values of the intervals)
    """
    vals = [rmu(imu, b, beta, type = type) for imu in mus]
    def xci(x):
        sel = [(ixs[0] <= x) & (x <= ixs[1]) for ixs in vals]
        if (sum(sel) <= 1): return None
        _ci = min(mus[sel]), max(mus[sel])
        return _ci
    x0s = np.array([ival[0] for ival in vals])
    x1s = np.array([ival[1] for ival in vals])
    return x0s, x1s, xci

def ci(mus, b, beta, type = type):
    xs = np.array([rmu(imu, b, beta, type = type) for imu in mus])
    def xci(x):
        sel = [(ixs[0] <= x) & (x <= ixs[1]) for ixs in xs]
        if (sum(sel) <= 1): return None
        _ci = min(mus[sel]), max(mus[sel])
        return _ci
    return xci

#---- private


def _rmu_classical(mu, b, beta, full_output = False):

    ndim = int(10*(b + mu))
    alpha = (1-beta)/2.
    ns  = np.array(range(ndim))
    cps = np.array([stats.poisson.cdf(ni, b + mu) for ni in ns])
    ils = [i for i in ns if cps[i] >= alpha]
    ius = [i for i in ns if cps[i] >= 1- alpha]
    il, iu = min(ils), min(ius)

    result = (il, iu)
    if (full_output):
        ps = [stats.poisson.pmf(ni, b + mu) for ni in ns]
        result = result, list(zip(cps, ps, ns))
    return result


def _rmu_fc(mu, b, beta, full_output = False):

    ndim = int(10*(b + mu))

    # range of n-s to study
    ns = np.arange(ndim)
    # probability of n
    ps = np.array([stats.poisson.pmf(ni, b+mu) for ni in ns])
    # best mu estimate for each n
    nhats = np.array([max(0, ni-b) for ni in ns])
    # probability for each n with best estimate of mu
    phs = np.array([stats.poisson.pmf(ni, b+nihat) for ni,nihat in zip(ns, nhats)])
    # the likelihood ratio
    ts = ps/phs

    # compact the values
    zs = list(zip(ts, ps, ns))
    zs.sort()
    zs.reverse()
    ts = np.array([zi[0] for zi in zs])
    ps = np.array([zi[1] for zi in zs])
    ns = np.array([zi[-1] for zi in zs])
    cps = np.array([sum(ps[:i+1]) for i in range(ndim)])
    sel = (cps < beta)
    xbeta = min(cps[~sel])
    sel = (cps <= xbeta)
    iu, il = min(ns[sel]), max(ns[sel])

    result = (iu, il)
    if (full_output):
        result = result, list(zip(ts, cps, ns))
    return result

    # compute the FC interval
    #j, p = 0, 0.
    #while (p < beta):
    #        j = min(j+1, ndim)
    #    jps = np.array([zi[1] for zi in zs[: j]])
    #    p = np.sum(jps)
    #    # print j, p
    #ids = [zi[2] for zi in zs[0: j+1]]
    #i0, i1 =  min(ids), max(ids)

    result = (i0, i1)
    if (full_output):
        result = (result, zs)
    return result
