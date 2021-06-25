import jax.numpy as jnp


def gaussian_kernel(Y, sigma, d):
    """
    Computes the elementwise d-dimensional gaussian pdf of Y
    """
    prefactor = 1 / ((2 * jnp.pi) ** (d / 2))
    prefactor /= (2 * sigma ** 2) ** 0.5
    return prefactor * jnp.exp(-(Y ** 2) / (2 * (2 * sigma ** 2)))
