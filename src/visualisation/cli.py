import argparse

from src.utils import get_cache
from visualisation import histogram


def getBooleanAnswer(question):
    ans = ""
    while ans != "y" and ans != "n":
        ans = input(f"\n{question} y/n\n")
    return ans == "y"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_path",
        default="/Volumes/LaCie/Brain/Data/ecephys_cache_dir/manifest.json",
    )
    parser.add_argument(
        "--output_dir", default="/Volumes/LaCie/Brain/Figures",
    )
    args = parser.parse_args()

    stimulus_names = [
        "spontaneous",
        "gabors",
        "flashes",
        "drifting_gratings",
        "natural_movie_three",
        "natural_movie_one",
        "static_gratings",
        "natural_scenes",
    ]

    # welcome the user
    print("Welcome to the neuropixels data visualisation console v0.1")
    print("Initialising cache...\n")

    # create an EcephysProjectCache from our local manifest
    cache = get_cache(args.manifest_path)
    session_ids = cache.get_session_table().index.values

    # select stimulus type
    s = "Please choose a stimulus type to visualise data for. Available stimulus types are:\n"
    for sn in stimulus_names:
        s += f"{sn}\n"
    stimulus_name = input(f"{s}\n")

    # select sessions
    num_sessions = 0
    while num_sessions < 1 or num_sessions > 58:
        num_sessions = int(
            input(f"\nHow many sessions would you like to visualise? (1-58)\n")
        )
    sids = session_ids[:num_sessions]

    # select relative or absolute
    relative = getBooleanAnswer(
        "Would you like to use spike times relative to stimulus onset (as opposed to absolute times)?"
    )

    # colour?
    colour = getBooleanAnswer(
        "Would you like to colour the units according to their brain region?"
    )

    # resume?
    resume = getBooleanAnswer(
        "Are you resuming / continuing from a previous visualisation?"
    )

    print("\nThank you. Generating visualisation...")
    histogram(
        cache,
        sids,
        stimulus_name,
        args.output_dir,
        resume=resume,
        relative=relative,
        colour=colour,
    )


if __name__ == "__main__":
    main()
