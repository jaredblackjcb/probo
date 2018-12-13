import abc
import enum
import numpy as np
from scipy.stats import binom
from scipy.stats import norm
from scipy.stats.mstats import gmean


class PricingEngine(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def calculate(self):
        """A method to implement a pricing model.

           The pricing method may be either an analytic model (i.e.
           Black-Scholes), a PDF solver such as the finite difference method,
           or a Monte Carlo pricing algorithm.
        """
        pass


class BinomialPricingEngine(PricingEngine):
    def __init__(self, steps, pricer):
        self.__steps = steps
        self.__pricer = pricer

    @property
    def steps(self):
        return self.__steps

    @steps.setter
    def steps(self, new_steps):
        self.__steps = new_steps

    def calculate(self, option, data):
        return self.__pricer(self, option, data)


def EuropeanBinomialPricer(pricing_engine, option, data):
    expiry = option.expiry
    strike = option.strike
    (spot, rate, volatility, dividend) = data.get_data()
    steps = pricing_engine.steps
    nodes = steps + 1
    dt = expiry / steps
    u = np.exp((rate * dt) + volatility * np.sqrt(dt))
    d = np.exp((rate * dt) - volatility * np.sqrt(dt))
    pu = (np.exp(rate * dt) - d) / (u - d)
    pd = 1 - pu
    disc = np.exp(-rate * expiry)
    spotT = 0.0
    payoffT = 0.0

    for i in range(nodes):
        spotT = spot * (u ** (steps - i)) * (d ** (i))
        payoffT += option.payoff(spotT) * binom.pmf(steps - i, steps, pu)
    price = disc * payoffT

    return price


def AmericanBinomialPricer(pricingengine, option, data):
    expiry = option.expiry
    strike = option.strike
    (spot, rate, volatility, dividend) = data.get_data()
    steps = pricingengine.steps
    nodes = steps + 1
    dt = expiry / steps
    u = np.exp((rate * dt) + volatility * np.sqrt(dt))
    d = np.exp((rate * dt) - volatility * np.sqrt(dt))
    pu = (np.exp(rate * dt) - d) / (u - d)
    pd = 1 - pu
    disc = np.exp(-rate * dt)
    dpu = disc * pu
    dpd = disc * pd

    Ct = np.zeros(nodes)
    St = np.zeros(nodes)

    for i in range(nodes):
        St[i] = spot * (u ** (steps - i)) * (d ** i)
        Ct[i] = option.payoff(St[i])

    for i in range((steps - 1), -1, -1):
        for j in range(i + 1):
            Ct[j] = dpu * Ct[j] + dpd * Ct[j + 1]
            St[j] = St[j] / u
            Ct[j] = np.maximum(Ct[j], option.payoff(St[j]))

    return Ct[0]


class MonteCarloEngine(PricingEngine):
    def __init__(self, replications, time_steps, pricer):
        self.__replications = replications
        self.__time_steps = time_steps
        self.__pricer = pricer

    @property
    def replications(self):
        return self.__replications

    @replications.setter
    def replications(self, new_replications):
        self.__replications = new_replications

    @property
    def time_steps(self):
        return self.__time_steps

    @time_steps.setter
    def time_steps(self, new_time_steps):
        self.__time_steps = new_time_steps

    def calculate(self, option, data):
        return self.__pricer(self, option, data)


def BlackScholesDelta(spot, t, strike, expiry, volatility, rate, dividend):
    tau = expiry - t
    d1 = (np.log(spot / strike) + (rate - dividend + 0.5 * volatility * volatility) * tau) / (volatility * np.sqrt(tau))
    delta = np.exp(-dividend * tau) * norm.cdf(d1)
    return delta


def NaiveMonteCarloPricer(engine, option, data):
    expiry = option.expiry
    strike = option.strike
    (spot, rate, vol, div) = data.get_data()
    replications = engine.replications
    dt = expiry / engine.time_steps
    disc = np.exp(-rate * dt)

    z = np.random.normal(size=replications)
    spotT = spot * np.exp((rate - div - 0.5 * vol * vol) * dt + vol * np.sqrt(dt) * z)
    payoffT = option.payoff(spotT)

    prc = payoffT.mean() * disc

    return prc


def AntitheticMonteCarloPricer(engine, option, data):
    expiry = option.expiry
    strike = option.strike
    (spot, rate, vol, div) = data.get_data()
    replications = engine.replications
    dt = expiry / engine.time_steps
    disc = np.exp(-(rate - div) * dt)

    z1 = np.random.normal(size=replications)
    z2 = -z1
    z = np.concatenate((z1, z2))
    spotT = spot * np.exp((rate - div) * dt + vol * np.sqrt(dt) * z)
    payoffT = option.payoff(spotT)

    prc = payoffT.mean() * disc

    return prc


def ControlVariatePricer(engine, option, data):
    expiry = option.expiry
    strike = option.strike
    (spot, rate, volatility, dividend) = data.get_data()
    dt = expiry / engine.time_steps
    nudt = (rate - dividend - 0.5 * volatility * volatility) * dt
    sigsdt = volatility * np.sqrt(dt)
    erddt = np.exp((rate - dividend) * dt)
    beta = -1.0
    cash_flow_t = np.zeros((engine.replications,))
    price = 0.0

    for j in range(engine.replications):
        spot_t = spot
        convar = 0.0
        z = np.random.normal(size=int(engine.time_steps))

        for i in range(int(engine.time_steps)):
            t = i * dt
            delta = BlackScholesDelta(spot, t, strike, expiry, volatility, rate, dividend)
            spot_tn = spot_t * np.exp(nudt + sigsdt * z[i])
            convar = convar + delta * (spot_tn - spot_t * erddt)
            spot_t = spot_tn

        cash_flow_t[j] = option.payoff(spot_t) + beta * convar

    price = np.exp(-rate * expiry) * cash_flow_t.mean()
    # stderr = cash_flow_t.std() / np.sqrt(engine.replications)
    return price


# class BlackScholesPayoffType(enum.Enum):
#    call = 1
#    put = 2

class BlackScholesPricingEngine(PricingEngine):
    def __init__(self, payoff_type, pricer):
        self.__payoff_type = payoff_type
        self.__pricer = pricer

    @property
    def payoff_type(self):
        return self.__payoff_type

    def calculate(self, option, data):
        return self.__pricer(self, option, data)


def BlackScholesPricer(pricing_engine, option, data):
    strike = option.strike
    expiry = option.expiry
    (spot, rate, volatility, dividend) = data.get_data()
    d1 = (np.log(spot / strike) + (rate - dividend + 0.5 * volatility * volatility) * expiry) / (
                volatility * np.sqrt(expiry))
    d2 = d1 - volatility * np.sqrt(expiry)

    if pricing_engine.payoff_type == "call":
        price = (spot * np.exp(-dividend * expiry) * norm.cdf(d1)) - (strike * np.exp(-rate * expiry) * norm.cdf(d2))
    elif pricing_engine.payoff_type == "put":
        price = (strike * np.exp(-rate * expiry) * norm.cdf(-d2)) - (spot * np.exp(-dividend * expiry) * norm.cdf(-d1))
    else:
        raise ValueError("You must pass either a call or a put option.")

    # try:
    #    #if pricing_engine.payoff_type == BlackScholesPayoffType.call:
    #    if pricing_engine.payoff_type == "call":
    #        price = (spot * np.exp(-dividend * expiry) * norm.cdf(d1)) - (strike * np.exp(-rate * expiry) * norm.cdf(d2))
    #    #else pricing_engine.payoff_type == BlackScholesPayoffType.put:
    #    else pricing_engine.payoff_type == "put":
    #        price = (strike * np.exp(-rate * expiry) * norm.cdf(-d2)) - (spot * np.exp(-dividend * expiry) * norm.cdf(-d1))
    # except ValueError:
    #    print("You must supply either a call or a put option to the BlackScholes pricing engine!")

    return price


def AssetPaths(spot, mu, sigma, expiry, div, nreps, nsteps):
    paths = np.empty((nreps, nsteps + 1))
    h = expiry / nsteps
    paths[:, 0] = spot
    mudt = (mu - div - 0.5 * sigma * sigma) * h
    sigmadt = sigma * np.sqrt(h)

    for t in range(1, nsteps + 1):
        z = np.random.normal(size=nreps)
        paths[:, t] = paths[:, t - 1] * np.exp(mudt + sigmadt * z)

    return paths

def blackScholesCall(spot, strike, rate, vol, div, expiry):
    d1 = (np.log(spot / strike) + (rate - div + 0.5 * vol * vol) * expiry) / (vol * np.sqrt(expiry))
    d2 = d1 - vol * np.sqrt(expiry)
    callPrice = (spot * np.exp(-div * expiry) * norm.cdf(d1)) - (strike * np.exp(-rate * expiry)  * norm.cdf(d2))
    return callPrice

def geometricAsianCall(spot, strike, rate, vol, div, expiry, N):
    dt = expiry / N
    nu = rate - div - 0.5 * vol * vol
    a = N * (N+1) * (2.0 * N + 1.0) / 6.0
    V = np.exp(-rate * expiry) * spot * np.exp(((N + 1.0) * nu / 2.0 + vol * vol * a / (2.0 * N * N)) * dt)
    vavg = vol * np.sqrt(a) / pow(N, 1.5)
    callPrice = blackScholesCall(V, strike, rate, vavg, div, expiry)
    return callPrice

def AsianControlVariatePricer(engine, option, data):
    nreps = engine.replications
    nsteps = engine.time_steps
    K = option.strike
    (S, r, sig, div) = data.get_data()
    expiry = option.expiry
    dt = expiry / nsteps
    nudt = (r - div - 0.5 * sig ** 2) * dt
    sigsdt = sig * np.sqrt(dt)
    t = np.zeros(nsteps)

    sum_CT = 0
    sum_CT2 = 0
    #global paths
    paths = AssetPaths(S, r, sig, expiry, div, nreps, nsteps)

    # repeat nreps times
    for j in range(nreps):
        A = np.mean(paths[j])  # calculate means for each path
        G = gmean(paths[j])
        # get payoff
        CT = option.payoff(A) - option.payoff(G)
        sum_CT += CT
        sum_CT2 += CT * CT

    portfolio_value = sum_CT / nreps * np.exp(-r * expiry)
    SD = np.sqrt((sum_CT2 - sum_CT * sum_CT / nreps) * np.exp(-2 * r * expiry) / (nreps - 1))
    SE = SD / np.sqrt(nreps)
    callPrice = portfolio_value + geometricAsianCall(S, K, r, sig, div, expiry, nsteps)
    return (callPrice, SE)  # return paths





# def AsianControlVariatePricer(engine, option, data):
# #     # paste paths = AssetPaths(spot, mu, sigma, expiry, div, nreps, nsteps):
# #     (spot, rate, vol, div) = data.get_data()
# #     (portfolio_value, SE, paths) = ControlVariate(engine, option, data)
# #     expiry = option.expiry
# #     nreps = engine.replications
# #     nsteps = engine.time_steps
# #     call_t = 0.0
# #
# #     for i in range(nreps):
# #         call_t += option.payoff(paths[i])
# #
# #     call_t /= nreps
# #     call_t *= np.exp(-rate * expiry)
# #     # add a line that adds standard error
# #     return (call_t, SE)


# go to McDonald ch.19 ex.19.2 to check and see if calculations are pretty close to the same value
# the controlvariate pricer is the only pricer we need to turn in
# the pathwisemontecarlopricer is an intermediate step and we can add


def AsianPricer(engine, option, data):
    expiry = option.expiry
    strike = option.strike
    (spot, rate, vol, div) = data.get_data()
    replications = engine.replications
    dt = expiry / engine.time_steps
    disc = np.exp(-rate * dt)

    z = np.random.normal(size=replications)

    for i in range(replications):
        pass
        for j in range(engine.time_steps):
            pass





















