import numpy as np
from scipy.optimize import fsolve, minimize
from typing import Tuple, Union

def gk_sample(params: Tuple[float, float, float, float], size: int, c: float = 0.8) -> np.ndarray:
    normal_samples = np.random.normal(size=size)
    return gk_quantile(normal_samples, *params, c)

def gk_quantile(normal_quantiles: np.ndarray, a: float, b: float, g: float, k: float, c: float = 0.8) -> np.ndarray:
    vals = a + b * (1 + c * (1 - np.exp(-g * normal_quantiles)) / (1 + np.exp(-g * normal_quantiles))) * (1 + normal_quantiles**2)**k * normal_quantiles
    return vals

def gk_is_proper(a: float, b: float, g: float, k: float, c: float = 0.8) -> bool:
    if b <= 0 or k < -0.5:
        return False
    
    g = abs(g) + 1e-7
    gk_func = lambda z: (-(1 - np.exp(-g * z)) / (1 + np.exp(-g * z)) -
                        2 * g * z * np.exp(-g * z) / ((1 + np.exp(-g * z))**2 * (1 + (2 * k + 1) * z**2) / (1 + z**2)))
                        
    min_val = min([gk_func(minimize(gk_func, [20 / (2 * g)], bounds=[(0, 20 / g)]).x[0]), -1])
    
    return min_val >= -1 / c

def gk_log_likelihood(gk_sample: Union[float, np.ndarray], params: Tuple[float, float, float, float], c: float = 0.8) -> float:
    if not gk_is_proper(*params, c):
        return -np.inf
    
    sample = gk_sample.flatten()
    gk_func = lambda std_norm_values: sample - gk_quantile(std_norm_values, *params, c)
    std_norm_values = fsolve(gk_func, sample - params[0])
        
    return -np.sum(gk_quantile_log_derivative(std_norm_values, *params, c))  
    
def gk_quantile_log_derivative(std_norm_value: Union[float, np.ndarray], a: float, b: float, g: float, k: float, c: float = 0.8) -> Union[float, np.ndarray]:
    zu = std_norm_value
    zu2 = std_norm_value**2
    exp_gzu = np.exp(-g * zu)
    return np.log((b * c * (2 * np.pi)**0.5 * np.exp(zu2 / 2) * (1 + zu2)**k) *
                  ((1 / c + (1 - exp_gzu) / (1 + exp_gzu)) *
                   (1 + (2 * k + 1) * zu2) / (1 + zu2) +
                   (2 * g * zu * exp_gzu) / (1 + exp_gzu)**2))
