import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xarray as xr

from src.utils import (
    get_cache,
    get_spikes,
    count_spikes_absolute,
    count_spikes_relative,
    get_region_colours,
    colour_counts_data,
    save_svg,
    log,
    remove_nans,
)

# histogram produces a histogram of unitwise spike counts for a given stimulus type and list of session ids
# spikes can be binned according to either
#    a) relative time since stimulus onset (in which case we average over all presentations of the stimulus in a given session)
#    b) absolute spike timing (in which case no averaging is performed)
#
# Running this function over multiple sessions can take a while. Fortunately, processed spike counts are saved to disk after each session
# - if execution is interrupted and you want to return, just run the function with the original set of session ids and 'resume' set to True
# - the code will then read the data previously saved to disk and pick up from there.
#
# will produce an .eps image in the specified output directory
def histogram(
    cache,
    sids,
    stimulus_name,
    output_dir,
    relative=True,  # controls whether to plot times relative to stimulus onset
    bin_size=1,
    resume=False,
    snr_cutoff=1.5,
    colour=True,  # controls whether to colour units according to brain region
):
    logpath = os.path.join(output_dir, "tmp", "log")

    sids_done = []
    data = None

    if resume:
        # if resuming, get list of previously processed sessions from disk
        with open(os.path.join(output_dir, "tmp", "sessions")) as f:
            sids_done = [int(line) for line in f]

        # and read spike count data from disk
        data = xr.open_dataarray(os.path.join(output_dir, "tmp", "spike_counts.nc"))

    for sid in sids:
        session = cache.get_session_data(sid)

        # skip this session if a) we've already done it, or b) it doesn't contain the stimulus we want
        if (
            sid in sids_done
            or stimulus_name
            not in session.stimulus_presentations["stimulus_name"].values
        ):
            log(logpath, f"skipping {sid}")
            continue

        log(logpath, f"processing {sid}")

        spikes = get_spikes(session, stimulus_name, snr_cutoff)
        counts = (
            count_spikes_relative(spikes, bin_size)
            if relative
            else count_spikes_absolute(spikes, bin_size)
        )

        # save progress
        if data is None:
            data = counts
        else:
            data = xr.concat([data, counts], dim="unit_id")
        data.to_netcdf(os.path.join(output_dir, "tmp", "spike_counts.nc"))
        with open(os.path.join(output_dir, "tmp", "sessions"), "a+") as f:
            f.write(f"{sid}\n")

        sids_done.append(sid)

    data = remove_nans(data)

    if colour:
        data = colour_counts_data(data, cache.get_units(), baseline=0.25)

    fig = plot_spike_counts(
        data,
        f"Unitwise spike counts for {stimulus_name} ({len(sids)} sessions)",
        bin_size,
        colour=colour,
    )
    save_svg(
        fig,
        os.path.join(output_dir, f"{stimulus_name}_spike_histogram.svg"),
        {"sessions": ",".join([str(sid) for sid in sids_done])},
    )
    plt.show()


def plot_spike_counts(data, title, bin_size, colour=True):
    # data = remove_nans(data)

    fig, ax = plt.subplots(figsize=(24, 12))
    aspect = data.values.shape[1] / data.values.shape[0]
    img = ax.imshow(data, interpolation="none", aspect=aspect)

    if not colour:
        cbar = plt.colorbar(img, pad=0.01)
        cbar.set_label("spike count", fontsize=12)

    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_ylabel("unit", fontsize=12)

    # ax.set_xticks(np.arange(0, len(data["bin_no"]), 1))
    # ax.set_xticklabels(
    #     [f"{(i*bin_size):1.2f}" for i in data["bin_no"].values], rotation=45
    # )
    ax.set_xlabel("time (s)", fontsize=12)
    ax.set_title(title, fontsize=14)

    if colour:
        patches = [
            mpatches.Patch(color=v, label=k) for (k, v) in get_region_colours().items()
        ]
        plt.legend(
            handles=patches,
            bbox_to_anchor=(1.01, 1),
            loc=2,
            borderaxespad=0.0,
            prop={"size": 14},
        )

    return fig


def visualise_rfv(rfv, stimulus, projected_stimulus, spike_rate, frame_history):
    binary_label = (spike_rate > 0).astype(float)
    mu_spike = jnp.sum(projected_stimulus * binary_label) / jnp.sum(binary_label)
    mu_nonspike = jnp.sum(projected_stimulus * (1 - binary_label)) / jnp.sum(
        1 - binary_label
    )

    rfv = rfv.reshape((stimulus.shape[1], stimulus.shape[2], frame_history)) * (
        1 if mu_spike > mu_nonspike else -1
    )

    rfv = (rfv - jnp.mean(rfv)) / jnp.std(rfv)
    rfv = rfv / (jnp.max(rfv) - jnp.min(rfv))

    fig, ax = plt.subplots()
    ax.imshow(rfv)
    plt.show()
