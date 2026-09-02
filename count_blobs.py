import os

import pandas as pd
import xarray as xr


def count_blobs(base_dir):
    print(f"--- Results for {base_dir} ---")

    tracked_dir = os.path.join(base_dir, "tracked")
    if not os.path.exists(tracked_dir):
        print(f"Directory {tracked_dir} not found.")
        return

    day_dirs = sorted(
        [
            d
            for d in os.listdir(tracked_dir)
            if os.path.isdir(os.path.join(tracked_dir, d))
        ]
    )

    for day in day_dirs:
        traj_file = os.path.join(tracked_dir, day, "trajectories.nc")
        if not os.path.exists(traj_file):
            print(f"{day}: No trajectories.nc found.")
            continue

        try:
            # Try to load as xarray first (tobac/xarray format)
            ds = xr.open_dataset(traj_file)
            df = ds.to_dataframe().reset_index()
        except Exception:
            # Fallback for pandas/csv formats if any exist
            try:
                df = pd.read_csv(traj_file)
            except Exception as e:
                print(f"Failed to read {traj_file}: {e}")
                continue

        if "height" not in df.columns or "cell" not in df.columns:
            print(f"{day}: Missing required columns in trajectories.nc")
            continue

        counts = (
            df.dropna(subset=["cell"]).groupby("height")["cell"].nunique().to_dict()
        )

        print(f"Day: {day}")
        if not counts:
            print("  No blobs found.")
        else:
            for phenomenon_level, count in sorted(counts.items()):
                print(f"  {phenomenon_level}: {count} blobs")
    print()


if __name__ == "__main__":
    count_blobs("data")
