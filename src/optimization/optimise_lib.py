import numpy as np
from jax import grad, jit, jacfwd, jacrev
import jax.numpy as jnp
from scipy.optimize import line_search as _line_search
from scipy.linalg import orth

from src.utils.utils import no_op, log
from src.information import mutual


norm = lambda x: x / jnp.linalg.norm(x)


def jacobian(fun):
    return grad(fun)


def hessian(fun):
    return jit(jacfwd(jacrev(fun)))


def hvp(fun, x, v):
    return jit(grad(lambda x: jnp.vdot(grad(fun)(x), v)))(x)


def line_search(obj, x, d, log=no_op):
    res = _line_search(obj, jacobian(obj), x, d, maxiter=1000)

    log(f"result of line search = {res}")

    a, _, _, f, _, _ = res
    return a or 0.01, f or obj(x + d)


def cdm_alpha(fun, x, d):
    num = jnp.dot(d, jacobian(fun)(x))
    denom = jnp.dot(d, hvp(fun, x, d))
    return -(num / denom)


def cdm_beta(fun, x, d, j):
    Hd = hvp(fun, x, d)
    num = jnp.dot(j, Hd)
    denom = jnp.dot(d, Hd)
    return num / denom


def conjugate_directions_method_update(x, fun, d, j, log=log):
    a = cdm_alpha(fun, x, d)
    new_x = x + (a * d)

    j = jacobian(fun)(new_x)
    b = cdm_beta(fun, new_x, d, j)
    new_d = -j + (b * d)

    return norm(new_x), new_d, fun(new_x)


def gradient_method_update(x, fun, d, j, log=log):
    d = -d
    alpha, f = line_search(lambda w: -fun(w), x, d, log=log)
    new_x = norm(x + (alpha * d))
    new_d = jacobian(fun)(new_x)

    return new_x, new_d, -f


def initialise_rfv(stimulus, F, H):
    s0 = jnp.inf

    for _ in range(10):
        rfv = 2 * (jnp.array(np.random.random((F * H, 1))) - 0.5)
        rfv = jnp.array(orth(rfv)).reshape(F * H)

        y = mutual.project_stimulus(stimulus, rfv)
        Y = mutual.precompute_difference_matrix(y)
        s = jnp.max(Y) / 2

        if s < s0:
            s0 = s
            initial_rfv = rfv

    return initial_rfv, s0
