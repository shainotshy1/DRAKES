import argparse
import os
import pandas as pd

def merge_csvs(dir_name):
    fns = {}

    for fn in os.listdir(dir_name):
        file_path = os.path.join(dir_name, fn)
        name = fn.lower()
        if name.endswith(".csv"):
            info_lst = name.split('_')
            worker_info = info_lst[-1]
            group_name = name[:-4] # Remove .csv extension
            if worker_info[0] == 'w':
                group_name = name[:-(len(worker_info)+1)] # +1 to catch underscore

            if group_name not in fns:
                fns[group_name] = []

            fns[group_name].append((fn, file_path))

    for group_name in fns:
        dfs = []
        fn_group = fns[group_name]
        
        if len(fn_group) == 1:
            continue

        for fn, file_path in fn_group:
            if fn.lower()[:-4] == group_name:
                continue
            df = pd.read_csv(file_path)
            dfs.append(df)

        group_fn = dir_name + "/" + group_name + ".csv"
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(group_fn, index=False, mode='w')
        print(f"Saved group: {group_fn}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSV data merger")
    parser.add_argument("dir", help="Directory containing csv files")
    args = parser.parse_args()

    merge_csvs(args.dir)