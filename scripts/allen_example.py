import os
from pathlib import Path

import numpy as np

from src.utils import get_cache, log
from src.mapping import map_receptive_fields_allen

# change this if you're pietro
data_dir = "/rds/general/user/mt4217/home/fyp/data"
results_dir = "/rds/general/user/mt4217/home/fyp/results/RFs"

# initialise AllenSDK cache
cache = get_cache(os.path.join(data_dir, "ecephys_cache_dir/manifest.json"))

# get all units from first session
sid = 715093703
session = cache.get_session_data(sid)
units = session.units

# get dLGN units
dlgn_units = units[units["ecephys_structure_acronym"] == "LGd"]
dlgn_uids = dlgn_units.sort_values("snr", ascending=False).index.values

# function to estimate and save RFs for first 10 dLGN units with given options
def get_and_save_RFs(options, filename):
    # get RFVs
    rfv_dict = map_receptive_fields_allen(
        cache, dlgn_uids[:10], "natural_movie_one", options=options
    )

    for uid, rfv in rfv_dict.items():
        save_dir = os.path.join(results_dir, "dLGN", str(uid))
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        fp = os.path.join(save_dir, filename)
        np.savetxt(fp, rfv)


experiment_options = [
    {"frame_history": 1, "account_for_pupil_position": False},
    {"frame_history": 1, "account_for_pupil_position": True},
    {"frame_history": 2, "account_for_pupil_position": False},
    {"frame_history": 3, "account_for_pupil_position": False},
]

experiment_filenames = [
    "FH_1_no_pupil.txt",
    "FH_1_with_pupil.txt",
    "FH_3_no_pupil.txt",
    "FH_3_with_pupil.txt",
]

for i in range(len(experiment_options)):
    get_and_save_RFs(experiment_options[i], experiment_filenames[i])
