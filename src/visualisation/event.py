import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

class Event():

    def get_specific_units(self, spike_times, ids):
        return np.array([spike_times[k] for k in ids])

    def get_units_list(self, spike_times, upper_limit, lower_limit=0):
        unit_counter = -1
        ids = []
        for unit_id in spike_times:
            unit_counter += 1
            if lower_limit<=unit_counter<upper_limit:
                ids.append(unit_id)
        return np.array([spike_times[k] for k in ids])

    def raster_plot(self, spike_times, fig_size=(90.0,500.0), offset=1, color='k', session_id=0, show_plot=1, save_dir=None):
        if type(spike_times) is dict:
            spike_times = np.array([spike_times[k] for k in spike_times])
        print(f'times: {spike_times.shape}')
        
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(1,1,1)

        lineoffsets = np.array([i+1 for i in range(spike_times.shape[0])])
        lineoffsets *= offset
        linelength = (lineoffsets[1] - lineoffsets[0])*np.ones(lineoffsets.shape)

        ax.eventplot(spike_times, 'horizontal', colors='k', lineoffsets=lineoffsets, linelengths=linelength)
        plt.title(f'Raster plot for all units of session {session_id}')
        plt.xlabel('time (s)')
        plt.ylabel('unit')
        plt.tick_params(axis='y',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False) # labels along the bottom edge are off

        if show_plot:
            plt.show()
        if save_dir not None:
            plt.savefig(save_dir)