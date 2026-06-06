import argparse
import os
import glob
import pickle
from tqdm import tqdm


def load_sequential_pickle_file(path):
    """
    Reads a file containing repeated pickle.dump(record, f) calls.

    Returns:
        records: list[dict]
    """
    records = []

    with open(path, "rb") as f:
        while True:
            try:
                record = pickle.load(f)
                records.append(record)

            except EOFError:
                break

            except Exception as e:
                print(
                    f"[WARN] Stopped reading {path}. "
                    f"The file may have a corrupted/incomplete tail. "
                    f"Loaded {len(records)} valid records from this file. "
                    f"Error: {repr(e)}"
                )
                break

    return records


def consolidate_goal_cache(
    cache_dir,
    output_path,
    pattern="goal_cache*.pkl",
    overwrite=False,
):
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}\n"
            f"Use --overwrite if you want to replace it."
        )

    data_paths = sorted(glob.glob(os.path.join(cache_dir, pattern)))

    # Ignore index files.
    data_paths = [
        p for p in data_paths
        if not p.endswith(".index.pkl")
    ]

    # Also ignore the consolidated output itself if it is inside the same folder.
    output_abs = os.path.abspath(output_path)
    data_paths = [
        p for p in data_paths
        if os.path.abspath(p) != output_abs
    ]

    if len(data_paths) == 0:
        raise FileNotFoundError(
            f"No cache files found in {cache_dir} using pattern: {pattern}"
        )

    print("=" * 80)
    print(f"Found {len(data_paths)} cache files.")
    print(f"Output will be saved to: {output_path}")
    print("=" * 80)

    goal_dict = {}

    total_records = 0
    duplicate_count = 0
    corrupted_tail_files = 0

    for path in tqdm(data_paths, desc="Reading cache files"):
        records = load_sequential_pickle_file(path)

        for record in records:
            key = record["key"]

            # Since you said duplicates do not exist, this is just a lightweight warning.
            # It does not run a separate duplicate-search pass.
            if key in goal_dict:
                duplicate_count += 1
                print(f"[WARN] Duplicate key found and overwritten: {key}")

            goal_dict[key] = record
            total_records += 1

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    tmp_output_path = output_path + ".tmp"

    print("=" * 80)
    print(f"Writing consolidated cache with {len(goal_dict)} records...")
    print("=" * 80)

    with open(tmp_output_path, "wb") as f:
        pickle.dump(goal_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.replace(tmp_output_path, output_path)

    print("=" * 80)
    print("Done.")
    print(f"Total records read:        {total_records}")
    print(f"Unique records written:    {len(goal_dict)}")
    print(f"Duplicate records found:   {duplicate_count}")
    print(f"Saved to:                  {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Directory containing goal_cache*.pkl files.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the consolidated dictionary pickle file.",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="goal_cache*.pkl",
        help="Pattern for input cache files.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )

    args = parser.parse_args()

    consolidate_goal_cache(
        cache_dir=args.cache_dir,
        output_path=args.output_path,
        pattern=args.pattern,
        overwrite=args.overwrite,
    )