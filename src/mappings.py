
import math
import numpy as np

def adv(epsilon: float) -> float:
    """Adversarial advantage bound used in Gaze3P:
       (e^eps - 1) / (e^eps + 1) = tanh(eps/2)
    """
    epsilon = float(epsilon)
    return math.tanh(epsilon / 2.0)

def inv_adv(g: float) -> float:
    """Inverse mapping epsilon = log((1+g)/(1-g))."""
    g = float(g)
    if not (-1.0 < g < 1.0):
        raise ValueError("g must be strictly between -1 and 1.")
    return math.log((1.0 + g) / (1.0 - g))

def _validate_level(level: int, L: int):
    if int(level) != level or not (1 <= level <= L):
        raise ValueError(f"privacy level must be an integer in [1, {L}]")

def linear_epsilon(level: int, eps_min: float = 0.1, eps_max: float = 5.0, L: int = 7) -> float:
    _validate_level(level, L)
    a_min, a_max = adv(eps_min), adv(eps_max)
    g = a_min + ((level - 1) / (L - 1)) * (a_max - a_min)
    return inv_adv(g)

def exponential_epsilon(level: int, eps_min: float = 0.1, eps_max: float = 5.0, L: int = 7) -> float:
    _validate_level(level, L)
    a_min, a_max = adv(eps_min), adv(eps_max)
    g = (a_min ** ((L - level) / (L - 1))) * (a_max ** ((level - 1) / (L - 1)))
    return inv_adv(g)

def sequential_epsilon(level: int, eps_min: float = 0.1, eps_max: float = 5.0, L: int = 7) -> float:
    _validate_level(level, L)
    a_min, a_max = adv(eps_min), adv(eps_max)
    if level in (1, 2):
        g = a_min
    elif level in (L - 1, L):
        g = a_max
    else:
        g = 0.5 * (a_min + a_max)
    return inv_adv(g)

def sigmoid_epsilon(level: int, eps_min: float = 0.1, eps_max: float = 5.0, L: int = 7, k: float = 1.5) -> float:
    """Implements the sigmoid-based mapping shown in Gaze3P Sec. 6.1.4.
       Endpoints are handled explicitly to avoid division by zero.
    """
    _validate_level(level, L)
    a_min, a_max = adv(eps_min), adv(eps_max)

    if level == 1:
        g = a_min
    elif level == L:
        g = a_max
    else:
        ratio = ((L - 1) / (level - 1)) - 1.0
        denom = 1.0 + (ratio ** (-k))
        g = a_max + (a_min - a_max) / denom
    return inv_adv(g)

def all_mappings(level: int, eps_min: float = 0.1, eps_max: float = 5.0, L: int = 7, k: float = 1.5):
    return {
        "linear": linear_epsilon(level, eps_min, eps_max, L),
        "exponential": exponential_epsilon(level, eps_min, eps_max, L),
        "sequential": sequential_epsilon(level, eps_min, eps_max, L),
        "sigmoid": sigmoid_epsilon(level, eps_min, eps_max, L, k),
    }
