import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats import kde


# List of epoch blocks corresponding to the chosen stimulus
def get_stimulusBlocks(session, stimulus_name):
    scenes = session.get_stimulus_table([stimulus_name])
    return np.unique(scenes["stimulus_block"].to_numpy())


# Starting and ending time points for a given block
def get_blockTimes(session, block_n):
    data = session.get_stimulus_table()
    block = data.loc[data["stimulus_block"] == block_n]
    start = min(block["start_time"])
    dt = block["stop_time"].to_numpy()
    return [dt, start, np.max(dt)]


def completeGaze_screen(session, time_start, time_end):
    data = session.get_screen_gaze_data()
    start, end = closest_timestep(data.index.to_numpy(), time_start, time_end)

    x_col = "raw_screen_coordinates_x_cm"
    y_col = "raw_screen_coordinates_y_cm"
    x_pos = data[x_col][start:end]
    y_pos = data[y_col][start:end]

    return [x_pos, y_pos]


def binGaze_precise(session, dt, start):
    data = session.get_screen_gaze_data()
    data = data[data.index.to_numpy() >= start]

    x_col = "raw_screen_coordinates_x_cm"
    y_col = "raw_screen_coordinates_y_cm"

    j = 1
    i = 0
    i_temp = 0
    x = np.array([0.0])
    y = np.array([0.0])

    while j < len(dt):
        if data.index[i] < dt[j]:
            # index = data.index[i]
            x_temp = data[x_col].iloc[i]
            y_temp = data[y_col].iloc[i]
            if np.isnan(x_temp) or np.isnan(y_temp):
                i_temp += 1
            else:
                x[j - 1] += x_temp
                y[j - 1] += y_temp
            i += 1
        else:
            x[j - 1] += x[j - 1] / (i - i_temp)
            y[j - 1] += y[j - 1] / (i - i_temp)
            x = np.append(x, 0)
            y = np.append(y, 0)
            j += 1
            i_temp = i - 1

    return [x, y]


def binGaze(session, dt):
    data = session.get_screen_gaze_data()
    data = data[data.index.to_numpy() >= np.min(dt)]
    data = data[data.index.to_numpy() <= np.max(dt)]

    x_col = "raw_screen_coordinates_x_cm"
    y_col = "raw_screen_coordinates_y_cm"
    time_steps = data[x_col].index.to_numpy()
    x_col = data[x_col].to_numpy()
    y_col = data[y_col].to_numpy()

    x_col = np.nan_to_num(x_col, nan=np.nanmean(x_col))
    y_col = np.nan_to_num(y_col, nan=np.nanmean(y_col))

    x, _ = np.histogram(time_steps, dt, weights=x_col)
    y, _ = np.histogram(time_steps, dt, weights=y_col)

    return [x, y]


def gaze_screen(session, start, length):

    data = session.get_screen_gaze_data()
    data = data[data.index.to_numpy() >= start]
    data = data.iloc[:length]

    x_col = "raw_screen_coordinates_x_cm"
    y_col = "raw_screen_coordinates_y_cm"
    x_col = data[x_col].to_numpy()
    y_col = data[y_col].to_numpy()

    x_col = jnp.nan_to_num(x_col, nan=jnp.nanmean(x_col))
    y_col = jnp.nan_to_num(y_col, nan=jnp.nanmean(y_col))

    return [x_col, y_col]


def delta_screen_degree(session, time_start, time_end):
    data = session.get_screen_gaze_data()
    start, end = closest_timestep(data.index.to_numpy(), time_start, time_end)

    x_col = "raw_screen_coordinates_spherical_x_deg"
    y_col = "raw_screen_coordinates_spherical_y_deg"
    x_min_deg = min(data[x_col][start:end])
    x_max_deg = max(data[x_col][start:end])
    y_min_deg = min(data[y_col][start:end])
    y_max_deg = max(data[y_col][start:end])

    return [x_max_deg - x_min_deg, y_max_deg - y_min_deg]


def closest_timestep(array, start, end):
    start = np.argmin(abs(array - start))
    end = np.argmin(abs(array - end))
    return [array[start], array[end]]


def get_block_completeScreenGaze(session, block_n):
    _, start, stop = get_blockTimes(session, block_n)
    return completeGaze_screen(session, start, stop)


def get_block_screenGaze_precise(session, block_n):
    dt, start, _ = get_blockTimes(session, block_n)
    return binGaze_precise(session, dt, start)


def get_block_screenGaze(session, block_n):
    dt, start, _ = get_blockTimes(session, block_n)
    return gaze_screen(session, start, len(dt))


def get_stimulus_screenGaze(session, stimulus_name, which_block):
    block_list = get_stimulusBlocks(session, stimulus_name)
    dt, start, _ = get_blockTimes(session, block_list[which_block])
    return gaze_screen(session, start, len(dt))


def plot_gaze(gaze_x, gaze_y):
    plt.plot(gaze_x, gaze_y)
    plt.show()


def plot_gazeDensity(gaze_x, gaze_y, nbins=300):
    k = kde.gaussian_kde([gaze_x, gaze_y])
    xi, yi = np.mgrid[
        gaze_x.min() : gaze_x.max() : nbins * 1j,
        gaze_y.min() : gaze_y.max() : nbins * 1j,
    ]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading="auto")
    plt.show()


def movie_eyeSaccade(movie, x, y):
    assert movie.shape[0] == len(
        x
    ), "The number of frames and of gaze recordings should be the same"

    monitor_width = 51.8
    pixelcm = monitor_width / movie.shape[2]

    for i in range(movie.shape[0]):
        x_shift = int(round(x[i] / pixelcm))
        y_shift = int(round(y[i] / pixelcm))

        if x_shift > 0 and y_shift > 0:
            frame = movie[i, y_shift:, x_shift:]
            frame = jnp.hstack((frame, jnp.zeros((frame.shape[0], x_shift))))
            frame = jnp.vstack((frame, jnp.zeros((y_shift, frame.shape[1]))))

        elif x_shift > 0 and y_shift < 0:
            frame = movie[i, :y_shift, x_shift:]
            frame = jnp.hstack((frame, jnp.zeros((frame.shape[0], x_shift))))
            frame = jnp.vstack((jnp.zeros((-y_shift, frame.shape[1])), frame))

        if x_shift < 0 and y_shift > 0:
            frame = movie[i, y_shift:, :x_shift]
            frame = jnp.hstack((jnp.zeros((frame.shape[0], -x_shift)), frame))
            frame = jnp.vstack((frame, jnp.zeros((y_shift, frame.shape[1]))))

        elif x_shift < 0 and y_shift < 0:
            frame = movie[i, :y_shift, :x_shift]
            frame = jnp.hstack((jnp.zeros((frame.shape[0], -x_shift)), frame))
            frame = jnp.vstack((jnp.zeros((-y_shift, frame.shape[1])), frame))

        movie[i, :, :] = frame

    return movie
