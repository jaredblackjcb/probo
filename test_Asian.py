from probo.marketdata import MarketData
from probo.payoff import VanillaPayoff, call_payoff, put_payoff
from probo.engine import MonteCarloEngine,  AsianControlVariatePricer
from probo.facade import OptionFacade

## Set up the market data
spot = 100
rate = 0.06
volatility = 0.2
dividend = 0.03
thedata = MarketData(rate, spot, volatility, dividend)

## Set up the option.
expiry = 1.0
strike = 100
thecall = VanillaPayoff(expiry, strike, call_payoff)
theput = VanillaPayoff(expiry, strike, put_payoff)

## Set up Naive Monte Carlo
nreps = 10000
steps = 10
pricer = AsianControlVariatePricer
mcengine = MonteCarloEngine(nreps, steps, pricer)

## Calculate the price
option1 = OptionFacade(thecall, mcengine, thedata)
price1, se1 = option1.price()
print("The call price via Naive Monte Carlo is: {0:.3f}".format(price1))
print("The standard error of the option price is: {0:.6f}".format(se1))


#option2 = OptionFacade(theput, mcengine, thedata)
#price2 = option2.price()
#print("The put price via Naive Monte Carlo is: {0:.3f}".format(price2))



