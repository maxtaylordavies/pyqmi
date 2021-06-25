# pyQMI

## Getting started

Activate or create a new virtual environment using either [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) or [venv](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments). Then, from the repo root, run
`pip install -r requirements.txt && pip install -e .`

<br>

## Estimating RFs for units in the Allen dataset

Once you've created an AllenSDK cache following [their instructions](https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_data_access.html), you can obtain RF estimates for any set of units in the dataset using

```python
from src.utils import get_cache
from src.mapping import map_receptive_fields_allen

cache = get_cache("path_to_your_manifest.json")

uids = [...]

rf_dict = map_receptive_fields_allen(
    cache, uids, "natural_movie_one"
)
```

This returns a dictionary mapping unit ids to their estimated RFs. You can customise the estimation process by passing an `options` dict, e.g.

```python
rf_dict = map_receptive_fields_allen(
    cache, uids, "natural_movie_one", options={"frame_history": 3}
)
```

the possible fields accepted by this object are

```python
{
    "frame_history": int >= 1 (default 1),
    "max_stimulus_repeats": int >= 1 (default 1),
    "account_for_pupil_position": Boolean (default False),
    "algorithm": "gradient method" | "conjugate directions method" (default "gradient method"),
    "stopping_condition_variable": "num_itr" | "delta" (default delta),
    "stopping_condition_threshold": float (detult 1e-2),
    "include_bandwidth_search": Boolean (default True),
}
```

<br>

## Estimating RFs from non-Allen data

You can also generate RF estimates using arbitrary neural data from outside the Allen dataset. You will need a 3D `jax.numpy` array containing the stimulus data (with shape `num_frames, frame_height, frame_width`) and a 1D `jax.numpy` array containing the spike counts for each stimulus frame. You can then call

```python
rf_estimate, qmi_history = map_receptive_field(
    stimulus, response
)
```

<br>

## Running on HPC GPUs

Here's an example job script for running on the HPC. Currently, the code is unable to parallelise across multiple GPUs, so you should select only one.

```bash
#!/bin/bash
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000

cd /rds/general/user/mt4217/home/fyp/code/pyqmi/scripts

module load cuda/10.1
module load cudnn/7.0
module load anaconda3/personal
source activate fyp

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/apps/cuda/10.1
export TF_CPP_MIN_LOG_LEVEL=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

python model_neuron_example.py
```
