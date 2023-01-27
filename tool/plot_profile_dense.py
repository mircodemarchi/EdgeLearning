import os
import argparse
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

DPI = 500

parser = argparse.ArgumentParser(description="Profile Tool", epilog="Mirco De Marchi")
parser.add_argument("--folder",
                    "-f",
                    dest="folder",
                    required=True,
                    help="Folder path in which there are the profiling values")
args = parser.parse_args()

output_path = os.path.join(".", "plots_profile_dense")
if not os.path.exists(output_path):
    os.makedirs(output_path)

def get_data(file_prefix, optimizations, data_shapes, files, folder):
    df = pd.DataFrame()
    for opt in optimizations:
        for s in data_shapes:
            fn = "{}_{}_{}.csv".format(file_prefix, opt, s)
            if fn not in files:
                continue
            f = os.path.join(folder, fn)
            profile_times = pd.read_csv(f).to_numpy().flatten() / 1e6
            data = {
                "ms": profile_times, 
                "optimization": np.repeat(opt, len(profile_times)),
                "shape": np.repeat(s, len(profile_times))
            }
            df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

    for s in df["shape"].unique():
        for opt in df["optimization"].unique():
            print(" - {:10} {:12}: {{ mean: {:10.4f}, median: {:10.4f}, std: {:10.4f} }}".format(
                opt, s,
                df.loc[(df["optimization"] == opt) & (df["shape"] == s)]["ms"].mean(),
                df.loc[(df["optimization"] == opt) & (df["shape"] == s)]["ms"].median(),
                df.loc[(df["optimization"] == opt) & (df["shape"] == s)]["ms"].std()
            ))
    return df

def plot(file_prefix, optimizations, data_shapes, files, folder, types=[]):
    if not types or len(types) == 0:
        return None
    types = [t for t in types if t in ["boxplot", "lineplot", "violinplot", "barplot"]]

    df = get_data(file_prefix, optimizations, data_shapes, files, folder)

    fig, axs = plt.subplots(nrows=len(types), figsize=(15,20))
    i = 0
    if "boxplot" in types: 
        sns.boxplot(data=df, x="shape", y="ms", hue="optimization", showfliers=False, linewidth=0.5, ax=axs[i] if len(types) > 1 else None)
        axs[i].set(yscale='log')
        i += 1
    if "lineplot" in types: 
        sns.lineplot(data=df, x="shape", y="ms", hue="optimization", ax=axs[i] if len(types) > 1 else None)
        axs[i].set(yscale='log')
        i += 1
    if "barplot" in types: 
        sns.barplot(data=df, x="shape", y="ms", hue="optimization", ax=axs[i] if len(types) > 1 else None, linewidth=0.2, errorbar="sd")
        axs[i].set(yscale='log')
        i += 1
    if "violinplot" in types:
        sns.violinplot(data=df, x="shape", y="ms", hue="optimization", ax=axs[i] if len(types) > 1 else None)
        axs[i].set(yscale='log')
        i += 1
    return fig



def plot_dense_on_optimizations(files, folder):
    print("-- plot_dense_on_optimizations --")
    FILE_PREFIX = "dense_on"
    OPTIMIZATIONS = ["sequential", "thread_opt", "simd_opt"]
    DATA_SHAPES = ["10x10", "10x100", "100x100", "100x1000", "1000x1000", "1000x10000", "10000x10000"]
    TYPES = ["boxplot", "barplot"]
    fig = plot(FILE_PREFIX, OPTIMIZATIONS, DATA_SHAPES, files, folder, TYPES)
    if fig:
        fig.savefig(os.path.join(output_path, "dense.jpg"), dpi=DPI)    
        plt.close(fig)

def plot_dense_1_on_optimizations(files, folder):
    print("-- plot_dense_1_on_optimizations --")
    FILE_PREFIX = "dense_1_on"
    OPTIMIZATIONS = ["sequential", "thread_opt"]
    DATA_SHAPES = ["10x10", "100x100", "1000x1000", "1000x10000", "10000x10000"]
    TYPES = ["boxplot", "barplot"]
    fig = plot(FILE_PREFIX, OPTIMIZATIONS, DATA_SHAPES, files, folder, TYPES)
    if fig:
        fig.savefig(os.path.join(output_path, "dense_1.jpg"), dpi=DPI)    
        plt.close(fig)


def main():
    profile_fp = args.folder

    if not os.path.isdir(profile_fp):
        raise Exception("Input arguments are not folder path")

    profile_files = [f for f in os.listdir(profile_fp) if f.endswith(".csv")]
    files = profile_files
    files.sort()

    functions = [
        plot_dense_on_optimizations,
        plot_dense_1_on_optimizations
    ]
    for fun in functions:
        fun(files, folder=profile_fp)



if __name__ == '__main__':
    main()