import jax.numpy as jnp

from src.utils import (
    get_stimulus_and_responses,
    pad_stimulus,
    log,
    normalise,
    flatten_stimulus,
)
from src.optimization.optimise import maximise_qmi


default_options = {
    "frame_height": 304,
    "frame_width": 608,
    "frame_history": 1,
    "max_stimulus_repeats": 5,
    "account_for_pupil_position": False,
    "algorithm": "gradient method",
    "stopping_condition_variable": "delta",
    "stopping_condition_threshold": 1e-2,
    "include_bandwidth_search": False,
    "smooth": False
}


def parse_options(options={}):
    default = default_options
    for k in options.keys():
        if k in default.keys():
            default[k] = options[k]
    return default


def map_receptive_fields_allen(
    cache, uids, stimulus_name, options={},
):
    """
    Estimate the receptive fields of a given subset of units from the Allen dataset
    """
    # parse input options
    options = parse_options(options)

    # get stimulus and response data for all specified units
    stimulus, responses = get_stimulus_and_responses(
        cache,
        uids,
        stimulus_name,
        max_repeats=options["max_stimulus_repeats"],
        do_pupil=options["account_for_pupil_position"],
    )

    # obtain dict mapping uid to RF estimate
    rfv_dict = {}
    for i in range(len(uids)):
        log(f"\n\n\nestimating rfv for unit {uids[i]}")
        rfv, _, = map_receptive_field(stimulus, responses[i], options)
        rfv_dict[uids[i]] = rfv

    return rfv_dict


def map_receptive_field(stimulus, response, options={}):
    """
    Estimate the receptive field of a single neuron by QMI optimisation

    Args:
        stimulus (jax.numpy array): stimulus data of shape (num_frames, height, width)
        response (jax.numpy array): framewise spike counts recorded for the neuron in response to the stimulus
        options (dict): dict of options 

    Returns:
        rfv (jax.numpy array): estimated receptive field vector
        qmi_history (list): history of QMI values for each iteration of the optimisation
        rfv_history (jax.numpy array): history of rfv values for each iteration of the optimisation
    """
    # parse input options
    options = parse_options(options)

    # normalise stimulus and response
    stimulus = normalise(stimulus)
    response = normalise(response)

    # flatten stimulus
    _, height, width = stimulus.shape
    F = height * width
    flattened_stimulus = pad_stimulus(
        flatten_stimulus(stimulus), options["frame_history"]
    )

    # obtain qmi-maximising rfv estimate
    return maximise_qmi(flattened_stimulus, response, F, options)


def map_receptive_field_sta(stimulus, response):
    """
    Estimate the receptive field of a single neuron by STA

    Args:
        stimulus (jax.numpy array): (un-normalised) stimulus data of shape (num_frames, height, width)
        response (jax.numpy array): (un-normalised) framewise spike counts recorded for the neuron in response to the stimulus
    
    Returns:
        rf (jax.numpy array): estimated receptive field
    """
    N, H, W = stimulus.shape

    # need stimulus to have zero mean
    stimulus = (2 * normalise(flatten_stimulus(stimulus))) - 1

    # then we can compute STA as
    sta = jnp.matmul(jnp.transpose(stimulus), response.reshape((N, 1)))
    return sta.reshape((H, W))
