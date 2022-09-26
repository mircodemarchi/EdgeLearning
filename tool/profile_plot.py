import os
import argparse
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

DPI=500

parser = argparse.ArgumentParser(description="Profile Tool", epilog="Mirco De Marchi")
parser.add_argument("--mlpack-folder",
                    "-mf",
                    dest="mlpack_fp",
                    required=True,
                    help="Folder path in which there are MLPACK profiling values")
parser.add_argument("--edgelearning-folder",
                    "-ef",
                    dest="edgelearning_fp",
                    required=True,
                    help="Folder path in which there are EDGELEARNING profiling values")
args = parser.parse_args()

def get_data(file_prefix, range_list, files, framework_name, folder, x, techniques=[""]):
    df = pd.DataFrame()
    for technique in techniques:
        for i in range_list:
            fn = file_prefix + technique + (str(i) if i else "") + ".csv"
            if fn not in files:
                continue
            f = os.path.join(folder, fn)
            profile_times = pd.read_csv(f).to_numpy().flatten() / 1e3
            data = {
                "ms": profile_times, 
                "framework": np.repeat(framework_name + "_" + technique, len(profile_times)),
                x: np.repeat(i if i else 0, len(profile_times))
            }
            df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

    print("{:15}: {{ mean: {:10.4f}, median: {:10.4f}, std: {:10.4f} }}".format(
        framework_name,
        df["ms"].mean(),
        df["ms"].median(),
        df["ms"].std()))
    for i in df[x].unique():
        print(" - {:4}: {{ mean: {:10.4f}, median: {:10.4f}, std: {:10.4f} }}".format(
            i,
            df.loc[df[x] == i]["ms"].mean(),
            df.loc[df[x] == i]["ms"].median(),
            df.loc[df[x] == i]["ms"].std()
        ))
    return df

def plot(file_prefix, range_list, files, mlpack_folder, edgelearning_folder, x="", types=[]):
    if not types or len(types) == 0:
        return None
    types = [t for t in types if t in ["boxplot", "lineplot", "violinplot"]]

    mlpack_df = get_data(file_prefix, range_list, files, "mlpack", mlpack_folder, x)
    edgelearning_df = get_data(file_prefix, range_list, files, "edgelearning", edgelearning_folder, x)
    df = pd.concat([mlpack_df, edgelearning_df], ignore_index=True)

    fig, axs = plt.subplots(nrows=len(types), figsize=(15,20))
    i = 0
    if "boxplot" in types: 
        sns.boxplot(data=df, x=x, y="ms", hue="framework", showfliers=False, linewidth=0.5, ax=axs[i] if len(types) > 1 else None)
        i += 1
    if "lineplot" in types: 
        sns.lineplot(data=df, x=x, y="ms", hue="framework", ax=axs[i] if len(types) > 1 else None)
        axs[i].set_xticks(range_list)
        i += 1
    if "violinplot" in types:
        sns.violinplot(data=df, x=x, y="ms", hue="framework", ax=axs[i] if len(types) > 1 else None)
        i += 1
    return fig


def plot_techniques(file_prefix, range_list, files, folder, techniques, x="", types=[]):
    if not types or len(types) == 0:
        return None
    types = [t for t in types if t in ["boxplot", "lineplot", "violinplot"]]
    
    df = get_data(file_prefix, range_list, files, "edgelearning", folder, x, techniques)

    fig, axs = plt.subplots(nrows=len(types), figsize=(15,20))
    i = 0
    if "boxplot" in types: 
        sns.boxplot(data=df, x=x, y="ms", hue="framework", showfliers=False, linewidth=0.5, ax=axs[i] if len(types) > 1 else None)
        i += 1
    if "lineplot" in types: 
        sns.lineplot(data=df, x=x, y="ms", hue="framework", ax=axs[i] if len(types) > 1 else None)
        axs[i].set_xticks(range_list)
        i += 1
    if "violinplot" in types:
        sns.violinplot(data=df, x=x, y="ms", hue="framework", ax=axs[i] if len(types) > 1 else None)
        i += 1
    return fig


def plot_training_on_epochs_amount(files, mlpack_folder, edgelearning_folder):
    print("-- plot_training_on_epochs_amount --")
    FILE_PREFIX = "training_on_epochs_amount"
    EPOCHS_RANGE = np.arange(1,21)
    TYPES = ["boxplot", "lineplot"]
    fig = plot(FILE_PREFIX, EPOCHS_RANGE, files, mlpack_folder, edgelearning_folder, "epochs", TYPES)
    if fig:
        fig.savefig(FILE_PREFIX + ".jpg", dpi=DPI)    
        plt.close(fig)

def plot_prediction(files, mlpack_folder, edgelearning_folder):
    print("-- plot_prediction --")
    FILE_PREFIX = "prediction"
    RANGE = [None]
    TYPES = ["violinplot"]
    fig = plot(FILE_PREFIX, RANGE, files, mlpack_folder, edgelearning_folder, "", TYPES)
    if fig:
        fig.savefig(FILE_PREFIX + ".jpg", dpi=DPI)
        plt.close(fig)

def plot_training_on_dataset_size(files, mlpack_folder, edgelearning_folder):
    print("-- plot_training_on_dataset_size --")
    FILE_PREFIX = "training_on_dataset_size"
    RANGE = [10,50,100,200,300,400,600,800,1000,3200]
    TYPES = ["boxplot", "lineplot"]
    fig = plot(FILE_PREFIX, RANGE, files, mlpack_folder, edgelearning_folder, "dataset_size", TYPES)
    if fig:
        fig.savefig(FILE_PREFIX + ".jpg", dpi=DPI)
        plt.close(fig)

def plot_prediction_on_dataset_size(files, mlpack_folder, edgelearning_folder):
    print("-- plot_prediction_on_dataset_size --")
    FILE_PREFIX = "prediction_on_dataset_size"
    RANGE = [10,50,100,200,300,400,600,800,1000,3200]
    TYPES = ["boxplot", "lineplot"]
    fig = plot(FILE_PREFIX, RANGE, files, mlpack_folder, edgelearning_folder, "dataset_size", TYPES)
    if fig:
        fig.savefig(FILE_PREFIX + ".jpg", dpi=DPI)
        plt.close(fig)

def plot_training_on_hidden_layers_amount(files, mlpack_folder, edgelearning_folder):
    print("-- plot_training_on_hidden_layers_amount --")
    FILE_PREFIX = "training_on_hidden_layers_amount"
    RANGE = list(range(1,11))
    TYPES = ["boxplot", "lineplot"]
    fig = plot(FILE_PREFIX, RANGE, files, mlpack_folder, edgelearning_folder, "hidden_layers_amount", TYPES)
    if fig:
        fig.savefig(FILE_PREFIX + ".jpg", dpi=DPI)
        plt.close(fig)

def plot_prediction_on_hidden_layers_amount(files, mlpack_folder, edgelearning_folder):
    print("-- plot_prediction_on_hidden_layers_amount --")
    FILE_PREFIX = "prediction_on_hidden_layers_amount"
    RANGE = list(range(1,11))
    TYPES = ["boxplot", "lineplot"]
    fig = plot(FILE_PREFIX, RANGE, files, mlpack_folder, edgelearning_folder, "hidden_layers_amount", TYPES)
    if fig:
        fig.savefig(FILE_PREFIX + ".jpg", dpi=DPI)
        plt.close(fig)

def plot_training_on_hidden_layers_shape(files, mlpack_folder, edgelearning_folder):
    print("-- plot_training_on_hidden_layers_shape --")
    FILE_PREFIX = "training_on_hidden_layers_shape"
    RANGE = list(range(10,21))
    TYPES = ["boxplot", "lineplot"]
    fig = plot(FILE_PREFIX, RANGE, files, mlpack_folder, edgelearning_folder, "hidden_layers_shape", TYPES)
    if fig:
        fig.savefig(FILE_PREFIX + ".jpg", dpi=DPI)
        plt.close(fig)

def plot_prediction_on_hidden_layers_shape(files, mlpack_folder, edgelearning_folder):
    print("-- plot_prediction_on_hidden_layers_shape --")
    FILE_PREFIX = "prediction_on_hidden_layers_shape"
    RANGE = list(range(10,21))
    TYPES = ["boxplot", "lineplot"]
    fig = plot(FILE_PREFIX, RANGE, files, mlpack_folder, edgelearning_folder, "hidden_layers_shape", TYPES)
    if fig:
        fig.savefig(FILE_PREFIX + ".jpg", dpi=DPI)
        plt.close(fig)


def plot_training_parallelism_techniques(files, folder):
    print("-- plot_training_parallelism_techniques --")
    FILE_PREFIX = "training_"
    TECHNIQUES = ["sequential_on_batch_size", "thread_parallelism_batch_on_batch_size", "thread_parallelism_entry_on_batch_size"]
    RANGE = [1,2,4,8,16,32,64,128]
    TYPES = ["boxplot", "lineplot"]
    fig = plot_techniques(FILE_PREFIX, RANGE, files, folder, TECHNIQUES, "batch_size", TYPES)
    if fig:
        fig.savefig(FILE_PREFIX + "technique_parallelism.jpg", dpi=DPI)
        plt.close(fig)


def main():
    mlpack_fp = args.mlpack_fp
    edgelearning_fp = args.edgelearning_fp

    if not os.path.isdir(mlpack_fp) or not os.path.isdir(edgelearning_fp):
        raise Exception("Input arguments are not folder path")

    mlpack_profile_files = [f for f in os.listdir(mlpack_fp) if f.endswith(".csv")]
    edgelearning_profile_files = [f for f in os.listdir(edgelearning_fp) if f.endswith(".csv")]
    files = list(set(mlpack_profile_files) & set(edgelearning_profile_files))
    files.sort()

    functions = [
        plot_training_on_epochs_amount,
        plot_prediction,
        plot_training_on_dataset_size,
        plot_prediction_on_dataset_size,
        plot_training_on_hidden_layers_amount,
        plot_prediction_on_hidden_layers_amount,
        plot_training_on_hidden_layers_shape,
        plot_prediction_on_hidden_layers_shape
    ]
    for fun in functions:
        fun(files, mlpack_folder=mlpack_fp, edgelearning_folder=edgelearning_fp)

    files = edgelearning_profile_files
    functions = [
        plot_training_parallelism_techniques
    ]
    for fun in functions:
        fun(files, folder=edgelearning_fp)


if __name__ == '__main__':
    main()