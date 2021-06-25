from jax import grad
import numpy as np
import jax.numpy as jnp

from src.information import mutual
from src.optimization.optimise_lib import (
    jacobian,
    initialise_rfv,
    gradient_method_update,
    conjugate_directions_method_update,
    norm,
)
from src.utils import log, smooth


def maximise_qmi(stimulus, response, F, options):
    log_wrapper = lambda itr, message: log(f"iteration {itr}: {message}")

    if options["algorithm"] == "gradient method":
        update_func = gradient_method_update
    elif options["algorithm"] == "conjugate directions method":
        update_func = conjugate_directions_method_update
    else:
        raise Exception(
            "algorithm must be either gradient method or conjugate directions method"
        )

    # helper functions for converting between 1D RFV and 2D RF
    def unroll(x):
        return x.reshape((options["frame_height"], options["frame_width"]))

    def roll(x):
        return x.reshape((options["frame_height"] * options["frame_width"]))

    # initialise random rfv
    rfv, sigma = initialise_rfv(stimulus, F, options["frame_history"])

    # precompute weights matrix from response
    weights = mutual.precompute_weights(response)

    # wrapper func computes qmi as an argument only of rfv
    def qmi_wrapper(rfv):
        rfv = norm(rfv)
        y = mutual.project_stimulus(stimulus, rfv)  # shape (1, N)
        Y = mutual.precompute_difference_matrix(y)  # shape (N, N)
        return mutual.quadratic(
            response, y, sigma, weights=weights, difference_matrix=Y
        )

    # compute sigma from rfv
    def compute_sigma_bar(rfv):
        y = mutual.project_stimulus(stimulus, rfv)
        Y = mutual.precompute_difference_matrix(y)
        return jnp.max(Y) / 2

    # get RF estimate by maximising QMI
    maximise = (
        maximise_qmi_with_bandwidth_search
        if options["include_bandwidth_search"]
        else maximise_qmi_without_bandwidth_search
    )
    rfv, qmi_vals, sigma = maximise(
        qmi_wrapper,
        compute_sigma_bar,
        rfv,
        sigma,
        update_func,
        options,
        log=log_wrapper,
    )

    # reverse if necessary
    y = mutual.project_stimulus(stimulus, rfv)
    binary_label = (response > 0).astype(float)
    mu_spike = jnp.sum(y * binary_label) / jnp.sum(binary_label)
    mu_nonspike = jnp.sum(y * (1 - binary_label)) / jnp.sum(1 - binary_label)
    rfv *= 1 if mu_spike > mu_nonspike else -1

    # unroll RFV into 2D RF
    rfv = unroll(norm(rfv))

    # if we don't want smoothing, return
    if not options["smooth"]:
        return rfv, qmi_vals

    # if we do want smoothing, smooth
    np.savetxt("unsmoothed.txt", rfv)
    smoothed = smooth(np.array(rfv), 19)
    best_qmi = qmi_wrapper(roll(rfv))

    smooth_itr = 0
    while True:
        s = smooth(np.array(smoothed), 19)
        q = qmi_wrapper(roll(s))

        log(f"smoothing iteration {smooth_itr}: qmi = {q}, best_qmi = {best_qmi}")

        if q <= best_qmi:
            break

        best_qmi = q
        smoothed = s

    return norm(smoothed), qmi_vals


def maximise_qmi_without_bandwidth_search(
    qmi_wrapper, sigma_func, rfv, sigma, update_func, options, log=log
):
    itr = 0
    delta = jnp.inf

    rfv = norm(rfv)

    def stop(itr, delta):
        v = options["stopping_condition_variable"]
        t = options["stopping_condition_threshold"]
        if v == "delta":
            return delta <= t
        elif v == "num_itr":
            return itr >= t

    # compute initial objective
    obj = qmi_wrapper(rfv)
    obj_vals = [obj]

    # optimise
    j = grad(qmi_wrapper)(rfv)
    d = -j
    while not stop(itr, delta):
        rfv, d, f = update_func(rfv, qmi_wrapper, d, j)

        delta = jnp.abs((f - obj) / f)
        obj = f
        obj_vals.append(obj)

        itr += 1

    return rfv, obj_vals, sigma


def maximise_qmi_with_bandwidth_search(
    qmi_wrapper, sigma_func, rfv, initial_sigma, update_func, options, log=log
):
    threshold = (
        options["stopping_condition_threshold"]
        if options["stopping_condition_variable"] == "delta"
        else 1e-2
    )

    rfv = norm(rfv)

    # compute initial qmi
    qmi = qmi_wrapper(rfv)
    qmi_vals = [qmi]
    sigma = initial_sigma

    # ------------- EXPANSION --------------
    grad = jacobian(qmi_wrapper)(rfv)
    d = -grad
    itr = 0
    delta = jnp.inf
    while True:
        rfv, d, f = update_func(rfv, qmi_wrapper, d, grad)

        # compute sigma_bar and delta
        sigma_bar = sigma_func(rfv)
        delta = jnp.abs((f - qmi) / f)
        log(itr, f"delta = {delta}")

        if sigma_bar > sigma:
            sigma = sigma_bar
            qmi = qmi_wrapper(rfv)
        else:
            qmi = -f
            if delta < threshold:
                break

        qmi_vals.append(qmi)
        grad = jacobian(qmi_wrapper)(rfv)
        itr += 1

    # ------------- CONTRACTION --------------
    sigma /= 2
    qmi = qmi_wrapper(rfv)
    grad = jacobian(qmi_wrapper)(rfv)
    d = -grad
    while True:
        rfv, d, f = update_func(rfv, qmi_wrapper, d, grad)

        # compute delta
        delta = jnp.abs((f - qmi) / f)
        log(itr, f"delta = {delta}")

        if delta < threshold:
            if sigma == initial_sigma:
                break
            elif sigma < initial_sigma:
                s = initial_sigma
            else:
                s = sigma / 2
            sigma = s
            qmi = qmi_wrapper(rfv)
        else:
            qmi = -f

        qmi_vals.append(qmi)
        grad = jacobian(qmi_wrapper)(rfv)
        itr += 1

    return rfv, qmi_vals, sigma
