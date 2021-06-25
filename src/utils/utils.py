import time
from datetime import datetime

import numpy as np
import jax.numpy as jnp
import xarray as xr
from tqdm import tqdm
from .pupil import gaze_screen, movie_eyeSaccade

########## MISCELLANEOUS GENERAL UTILITY FUNCTIONS ##########
def no_op(x):
    return x


def log(message):
    with open("./log.txt", "a+") as f:
        f.write(f"{datetime.now()}: {message}\n")


def dims(x):
    """
    Get the dimensionality of a jax / numpy array 
    """
    return len(x.shape)


def make_2d(x):
    """
    Convert 1D jax / numpy array to 2D
    """
    return x.reshape((1, len(x)))


########## UTILITY FUNCTIONS RELATED TO ALLEN SDK / DATA ##########
def get_frame_rate_for_stimulus(stimulus_name):
    if stimulus_name in ["natural_movie_one", "natural_movie_three"]:
        return 30
    raise Exception("Support pending for non-natural movie stimuli")


def get_session_id_for_unit(cache, uid):
    """
    Returns the id for the session a given unit was recorded in
    """
    units = cache.get_units()
    return units[units.index == uid]["ecephys_session_id"].values[0]


def get_stimulus_and_responses(
    cache, uids, stimulus_name, max_repeats=20, do_pupil=False
):
    """
    Gets stimulus and response data for a given set of Allen units and stimulus name
    """
    # get the raw stimulus data
    if stimulus_name == "natural_movie_one":
        stimulus = cache.get_natural_movie_template(1)
    elif stimulus_name == "natural_movie_three":
        stimulus = cache.get_natural_movie_template(3)
    else:
        raise Exception("Support pending for non-natural movie stimuli")

    sids = np.zeros((len(uids)))
    responses = []

    # get the response data for each neuron
    for i in range(len(uids)):
        uid = uids[i]
        sids[i] = get_session_id_for_unit(cache, uid)

        log(f"getting stimulus and response for unit {uid} from session {sids[i]}")

        # get start and stop times for the desired stimulus from the given session
        # don't need to redo this if unit is from same session as the previous one
        if i == 0 or sids[i] != sids[i - 1]:
            session = cache.get_session_data(sids[i])

            stimulus_table = session.get_stimulus_table()
            stimulus_table = stimulus_table.loc[
                stimulus_table["stimulus_name"] == stimulus_name
            ]

            start_times = stimulus_table.loc[:, ["start_time"]].values
            stop_times = stimulus_table.loc[:, ["stop_time"]].values

            start_times = start_times[: max_repeats * 900]
            stop = stop_times[(max_repeats * 900) - 1]

        # extend stimulus to account for repeats
        if i == 0:
            stimulus = repeat_stimulus(
                stimulus, int(len(start_times) / stimulus.shape[0])
            )

        if do_pupil:
            gaze = gaze_screen(session, jnp.min(start_times), len(start_times))
            stimulus = movie_eyeSaccade(stimulus, gaze[0], gaze[1])

        # get the corresponding response
        spikes_all = session.spike_times
        spikes = jnp.array([spikes_all[uid] for uid in uids])
        responses.append(count_spikes_absolute(spikes, start_times, stop))

    return stimulus, responses


def count_spikes_absolute(spikes, start, stop):
    bin_edges = np.append(start, stop)
    data = np.zeros((spikes.shape[0], len(start)))

    for i in range(spikes.shape[0]):
        sfu = spikes[i, :]
        sfu = sfu[sfu >= start[0]]
        sfu = sfu[sfu <= stop]

        hist, _ = np.histogram(sfu, bins=bin_edges)
        data[i] = hist

    return jnp.array(data)


########## UTILITY FUNCTIONS RELATED TO DATA PRE-/POST-PROCESSING IN GENERAL (NOT ALLEN-SPECIFIC) ##########
def smooth(x, w):
    height, width = x.shape
    smoothed = np.zeros((height, width))

    pad = int((w - 1) / 2)

    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            patch = x[i - pad : i + pad, j - pad : j + pad]
            s = np.sum(patch)
            smoothed[i : i + pad, j : j + pad] = s

    m = np.mean(smoothed)
    smoothed[:, :pad] = m
    smoothed[:pad, :] = m

    smoothed = smoothed / np.linalg.norm(smoothed)
    smoothed = smoothed ** 1.2

    return smoothed


def normalise(x):
    """
    Normalises array / matrix such that all values lie between 0 and 1
    """
    return (x - jnp.min(x)) / (jnp.max(x) - jnp.min(x))


def flatten_stimulus(stimulus):
    """
    Takes a 3D stimulus array, where each index along the first dimension is a matrix
    representing one frame, and returns a flattened 2D array where each element is a 
    vector representing one frame
    """
    n, h, w = stimulus.shape
    return stimulus.reshape((n, h * w))


def repeat_stimulus(stimulus, n_repeats):
    repeated = stimulus
    for _ in range(n_repeats - 1):
        repeated = jnp.concatenate((repeated, stimulus))
    return repeated


def pad_stimulus(stimulus, H):
    N = stimulus.shape[0]
    padded = stimulus
    for i in range(1, H):
        s = jnp.pad(stimulus[: N - i], ((i, 0), (0, 0)))
        padded = jnp.concatenate((s, padded), axis=1)
    return padded


# replace NaN elements in a DataArray with 0
def remove_nans(da):
    (num_rows, num_cols) = da.values.shape
    for ridx in range(num_rows):
        for cidx in range(num_cols):
            if jnp.isnan(da.values[ridx][cidx]):
                da.values[ridx][cidx] = 0
    return da


########## UTILITY FUNCTIONS RELATED TO PLOTTING / VISUALISATION
def save_svg(fig, pathname, metadata_dict):
    fig.savefig(pathname)
    time.sleep(1)
    add_description_to_svg(
        pathname, ",".join([f"{k}:{v}" for k, v in metadata_dict.items()])
    )


def add_description_to_svg(pathname, description):
    with open(pathname) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.strip() == "</metadata>":
            break
    lines.insert(i + 1, f" <desc>{description}</desc>")

    with open(pathname, "w") as f:
        f.writelines(lines)


def get_coarse_region(acronym):
    if acronym.startswith("VIS"):
        return "cortex"
    if acronym.startswith("CA") or acronym in ["DG", "SUB", "ProS"]:
        return "hippocampus"
    if acronym in ["LGd", "LGv", "LP"]:
        return "thalamus"
    if acronym == "APN":
        return "midbrain"
    return "other"


def get_region_colours():
    return {
        "cortex": jnp.array([27, 158, 119]) / 255,
        "hippocampus": jnp.array([217, 95, 2]) / 255,
        "thalamus": jnp.array([117, 112, 179]) / 255,
        "midbrain": jnp.array([231, 41, 138]) / 255,
        "other": jnp.array([166, 118, 29]) / 255,
    }


def get_unit_colours(unit, vals, max_val, baseline):
    colours = get_region_colours()
    region = get_coarse_region(unit["ecephys_structure_acronym"])

    x = baseline
    y = 1 - x

    vals = jnp.multiply(
        vals, x / vals + y / max_val
    )  # gives 0 if val is 0, else 0.5 + 0.5(val/max_val)
    colours = jnp.array([jnp.append(colours[region], v) for v in vals])

    return colours


# takes in a DataArray of spike count histogram values (dimensions units * bins)
# and a dataframe of units (e.g. from cache.get_units())
# and returns a new DataArray of dimensions units * bins * 4, where da[unit][bin] is an rgba colour value
# encoding both the brain region that unit belongs to, as well as the number of spikes from that unit in that bin.
def colour_counts_data(data, units, baseline=0.5):
    print("colouring spike count data...")

    # find the max spike count
    max_count = jnp.max(data.values)

    (nr, nc) = data.values.shape
    image = jnp.zeros((nr, nc, 4))

    unit_ids = data.coords["unit_id"].values
    bin_nos = data.coords["bin_no"].values

    for ridx in tqdm(range(nr)):
        unit = units.loc[unit_ids[ridx]]
        image[ridx] = get_unit_colours(unit, data.values[ridx], max_count, baseline)

    return xr.DataArray(
        image, [unit_ids, bin_nos, jnp.zeros(4)], ["unit_id", "bin_no", "colour"]
    )

