import jax.numpy as jnp
from jax import ops as jops
from jax.scipy.signal import convolve2d as conv2d

import src.probability.gaussian as prob
from src.utils import dims, make_2d


def project_stimulus(stimulus, rfv):
    """
    function returns projection of stimuli sets onto passed receptive
    field vectors - faster and less memory used than simple matrix multiplication.

    Input:
        stimulus: set of stimuli
        rfv: 1 x (f_size * frame_history) [numpy array]
        frame_history: frame history elements [scalar]

    Output:
        y_hat: projected stimuli

    """
    if dims(rfv == 1):
        rfv = make_2d(rfv)

    return jnp.transpose(jnp.matmul(stimulus, jnp.transpose(rfv)))


def precompute_weights(r):
    """
    Precompute the components of the QMI expression that will remain constant across the entire optimizatoin
    Note: the quantities returned are not technically the information potentials Vin, Vall and Vbtw - just
    the matrix that is elementwise multiplied with the guassian matrix in each potential
    Input:
        r: 1 x N numpy array of un-normalised binned spike counts
    Output:
        v_in: N x N numpy array
        v_all: N x N numpy array
        v_btw: N x N numpy array
    """
    if dims(r) == 1:
        r = make_2d(r)

    # obtain vectors representing prior probabilities for the 'spike' and 'no spike' classes
    p_spike = r / jnp.max(r)
    p_nonspike = 1 - p_spike
    n = r.shape[1]

    weight_matrix = (
        jnp.matmul(jnp.transpose(p_spike), p_spike)
        + jnp.matmul(jnp.transpose(p_nonspike), p_nonspike)
    ) / (n ** 2)
    weights_column_sums = jnp.sum(weight_matrix, axis=0)
    weights_overall_sum = jnp.sum(weights_column_sums)

    v_in = weight_matrix
    v_all = weights_overall_sum * jnp.ones((n, n)) / (n ** 2)

    v_btw = jnp.zeros((n, n))
    for i in range(n):
        v_btw = jops.index_update(v_btw, jops.index[i], weights_column_sums)
    v_btw /= n

    return v_in + v_all - 2 * v_btw


def precompute_difference_matrix(y):
    """
    Precompute the difference matrix that we pass as argument to the gaussian function
    Input:
        y: 1 x N numpy array representing stimulus projected onto single RFV
    Output:
        Y: N x N numpy array
    """
    return jnp.abs(jnp.transpose(y) - y)


def quadratic(x, y, sigma, weights=None, difference_matrix=None):
    """
    Calculate quadratic mutual information of two 2 low dimensional inputs
    with parzen kernel density estimation
    Dtype: jnp.float32
    Input:
        x: N x M x ... x D_x, D_x is the dimension of variable of x
        y: N x M x ... x D_y, D_y is the dimension of variable of x
        sigma: current parzen window width
        weights (optional): tuple containing the three weight matrices for the information potentials
        difference_matrix (optional): N x N numpy array whose (i,j)th element = y[i] - y[j]
    Output:
        mutual_info: scalar
    """
    ## make sure you use lamba or partial function to cache this function,
    ## for example the V_ij V_i V_j in the matlab code
    ## as it will be run many times during optimization

    if weights is None:
        weights = precompute_weights(x)

    if difference_matrix is None:
        difference_matrix = precompute_difference_matrix(y)

    return jnp.sum(
        jnp.multiply(weights, prob.gaussian_kernel(difference_matrix, sigma, 1))
    )

