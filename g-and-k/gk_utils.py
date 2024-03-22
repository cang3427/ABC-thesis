import numpy as np
from math import pi, exp, inf
from scipy.optimize import fsolve, minimize

def gk_sample(params, numSamples, c = 0.8):
    normalSamples = np.random.normal(size = numSamples).reshape((numSamples, 1))
    return gk_quantile(normalSamples, params, c).reshape((numSamples, 1))

def gk_quantile(normalSample, params, c = 0.8):    
    (a, b, g, k) = params
    val = a + b * (1 + c * (1 - np.exp(-g * normalSample)) / (1 + np.exp(-g * normalSample))) * (1 + normalSample**2)**k * normalSample
    return val

def gk_is_proper(params, c = 0.8):
    (a, b, g, k) = params
    if b <= 0 or k < -0.5:
        return False
    
    g = abs(g)
    gkFunc = lambda z: -(1 - exp(-g * z)) / (1 + exp(-g * z)) - \
                        2*g * z * exp(-g * z) / ((1 + exp(-g * z))**2 * (1 + (2*k + 1) * z**2) / (1 + z**2))
                        
    minVal = min([gkFunc(minimize(gkFunc, [20 / (2*g)], bounds = [(0, 20 / g)]).x[0]), -1])
    
    return minVal >= -1 / c

def gk_log_likelihood(gkSample, params, c = 0.8):
    if not gk_is_proper(params, c):
        print("NOT PROPER")
        return -inf
    
    sample = gkSample.flatten()
    gkFunc = lambda stdNormValues: sample - gk_quantile(stdNormValues, params, c)
    stdNormValues = fsolve(gkFunc, sample - params[0])
        
    return -np.sum(gk_quantile_log_derivative(stdNormValues, params, c))  
    
def gk_quantile_log_derivative(stdNormValue, params, c = 0.8):
    (a, b, g, k) = params
    zu = stdNormValue
    zu2 = stdNormValue**2
    exp_gzu = np.exp(-g * zu)
    return np.log((b * c * (2*pi)**0.5 * np.exp(zu2 / 2) * (1 + zu2)**k) * \
                  ((1 / c + (1 - exp_gzu) / (1 + exp_gzu)) * \
                   (1 + (2*k + 1) * zu2) / (1 + zu2) + \
                   (2*g * zu * exp_gzu) / (1 + exp_gzu)**2))
