import warnings

import mpl_scatter_density
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sns.set_context("paper")


def compare_marginal_scores(data: pd.DataFrame):
    """Lineplots of marginal scores across features for each synth.

    One subplot for each epsilon level (extracted from synth name).
    Averages over repeats.
    """

    # Transpose and relable dataframe
    _data = data.T[1:].reset_index()
    col_names = ["synth"]
    col_names.extend(data["feature"])
    _data.columns = col_names

    # Extract repeat suffix from synth name
    _data["epsilon"] = _data["synth"].apply(lambda s: s.split("_")[-2])
    _data["epsilon"] = pd.to_numeric(_data["epsilon"])
    _data["synth"] = _data["synth"].apply(lambda s: "_".join(s.split("_")[:-2]))

    # Convert other columns to numeric
    _data[col_names[1:]] = _data[col_names[1:]].apply(pd.to_numeric)

    # Plot the results
    n_eps_vals = _data["epsilon"].nunique()
    fig, axes = plt.subplots(n_eps_vals, 1, figsize=(9, 9))
    for i, eps in enumerate(_data["epsilon"].unique()):

        _data[_data["epsilon"] == eps].drop("epsilon", axis=1).groupby(["synth"]).agg(
            "mean"
        ).T.plot(
            kind="line",
            ax=axes[i] if n_eps_vals > 1 else axes,
            rot=90,
            sharex=True,
            marker="x",
            title=f"Epsilon={eps}",
        )
        ax = axes[i] if n_eps_vals > 1 else axes
        ax.set_xticks(range(len(col_names) - 1))
        ax.set_xticklabels(col_names[1:])
        ax.set_ylabel("Marginal distance")
        ax.set_xlabel("Feature")

    return fig


def visualise_metrics(
    data: pd.DataFrame, use_log=False, benchmark_synth="CTGAN_Synthesiser", emph_synth=None
):
    """
    Plot final metrics across synths and datasets.
    """

    sns.set_context("talk")

    # Get names of metrics and datasets
    metrics = list(data.columns)
    for c in ["dataset", "epsilon", "synth"]:
        metrics.remove(c)
    datasets = data["dataset"].unique()

    # Only work with a copy of the data to prevent side-effects
    _data = data.copy()
    _data["epsilon"] = pd.to_numeric(_data["epsilon"])
    _data = _data.sort_values(by=["dataset", "synth", "epsilon"], ascending=True)

    # Create a set of subplots (M x D) for M metrics and D datasets
    fig, axes = plt.subplots(
        len(metrics), len(datasets), figsize=(8 * len(datasets), 4 * len(metrics))
    )

    # Define handles and labels list for the legend
    handles, labels = [], []

    # Set the color palette
    colors = sns.color_palette("husl", len(_data["synth"].unique()))

    for j, dataset in enumerate(datasets):
        for i, metric in enumerate(metrics):
            ax = axes[i, j] if len(metrics) > 1 else axes[j]
            for k, synth in enumerate(_data["synth"].unique()):
                _subset = _data[(_data["dataset"] == dataset) & (_data["synth"] == synth)]
                _subset = (
                    _subset.drop(["dataset", "synth"], axis=1)
                    .groupby("epsilon")
                    .agg(["mean", "std"])
                    .reset_index()
                )

                emphasise = emph_synth is not None and emph_synth in synth

                if synth == benchmark_synth:
                    line = ax.axhline(
                        _subset[(metric, "mean")].values.mean(),
                        label=f"{synth} (no DP)",
                        color="black",
                        linestyle="--",
                        linewidth=2.0,
                        alpha=0.4,
                    )
                    handles.append(line)
                    labels.append(f"{synth} (no DP)")
                    continue

                error_container, _, _ = ax.errorbar(
                    _subset["epsilon"],
                    _subset[(metric, "mean")],
                    yerr=_subset[(metric, "std")],
                    marker=".",
                    capsize=4,
                    label=synth + r" $\pm\sigma$",
                    barsabove=True,
                    # Make the lines thicker
                    linewidth=1.5 if emphasise else 1,
                    elinewidth=0.8,
                    color=colors[k],
                    alpha=1.0 if emphasise else 0.7,
                )

                handles.append(error_container)
                labels.append(synth + r" $\pm\sigma$")

            ax.set_ylabel(metric)
            ax.set_title(dataset)

            if use_log:
                ax.set_xscale("log")

        ax.set_xlabel("epsilon")

    # Convert handles and labels into dictionaries to remove duplicates, then back into lists
    unique = dict(zip(labels, handles))
    labels = list(unique.keys())
    handles = list(unique.values())

    # Add the legend to the figure
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(_data["synth"].unique()) // 2,
        bbox_to_anchor=(0.5, -0.40 + 0.1 * len(metrics)),
        fontsize=16,
    )

    return fig


def visualise_marginals(real_dataset: pd.DataFrame, fake_dataset: pd.DataFrame, data_types):
    """
    Plots histograms to compare marginal distributions of real vs fake datasets.
    """
    plt.cla()
    plt.clf()
    _, ax = plt.subplots(2, len(data_types), figsize=(32, 8))

    for i, (col, var_type) in enumerate(data_types):
        axes = ax[:, i]
        # if col == "capital-gain":
        #     import ipdb; ipdb.set_trace()
        if "int" in var_type or "float" in var_type:
            colour = "#1a63a5"

            # Calculate bin width using Freedman-Diaconis rule
            iqr = np.subtract(*np.percentile(real_dataset[col].dropna(), [75, 25]))
            bin_width = 2 * iqr * len(real_dataset[col].dropna()) ** (-1 / 3) if iqr != 0 else 1
            bins = min(int((real_dataset[col].max() - real_dataset[col].min()) / bin_width), 50)

            real_dataset[col].plot(
                kind="hist",
                ax=axes[0],
                color=colour,
                sharex=False,
                sharey=True,
                title=f"{col} real",
                bins=bins,
                density=True,
            )
            fake_dataset[col].plot(
                kind="hist",
                ax=axes[1],
                color=colour,
                sharex=False,
                sharey=True,
                title=f"{col} fake",
                bins=bins,
                density=True,
            )

            # Overlay KDE
            sns.kdeplot(real_dataset[col], ax=axes[0], color="black", warn_singular=False)
            sns.kdeplot(fake_dataset[col], ax=axes[1], color="black", warn_singular=False)

        elif "categorical" in var_type:
            colour = "#12bf3f"

            # Get categories and their frequencies
            real_counts = real_dataset[col].value_counts(normalize=True)
            fake_counts = fake_dataset[col].value_counts(normalize=True)

            # Ensure fake_counts includes all categories present in real_counts
            fake_counts = fake_counts.reindex(real_counts.index, fill_value=0)

            # Sort categories by their labels
            real_counts = real_counts.sort_index()
            fake_counts = fake_counts.sort_index()

            real_counts.plot(
                kind="bar",
                ax=axes[0],
                color=colour,
                sharex=True,
                sharey=True,
                title=f"{col} real",
            )
            fake_counts.plot(
                kind="bar",
                ax=axes[1],
                color=colour,
                sharex=True,
                sharey=True,
                title=f"{col} fake",
            )

        # Set X-axis limits to include all data, but have same range
        xmin = min(axes[0].get_xlim()[0], axes[1].get_xlim()[0])
        xmax = max(axes[0].get_xlim()[1], axes[1].get_xlim()[1])
        axes[0].set_xlim(xmin, xmax)
        axes[1].set_xlim(xmin, xmax)

        # Disable y-axis tick labels
        axes[0].axes.yaxis.set_ticklabels([])
        axes[1].axes.yaxis.set_ticklabels([])

    return ax


def visualise_violins(real_dataset: pd.DataFrame, fake_dataset: pd.DataFrame):
    """
    Visualisation code for split violin plots
    """

    # Set up scaler for features and fit on real data
    scaler = StandardScaler().fit(real_dataset.values)

    # Make a copy of fake and label with new column
    data = fake_dataset.copy()
    data["veracity"] = "Fake"

    # Add the real data and label as real
    data = pd.concat([data, real_dataset], axis=0)
    data["veracity"].fillna("Real", inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Scale all the data (real and fake in one dataframe)
    colnames = list(fake_dataset.columns)
    _ver = data["veracity"]
    data = pd.DataFrame(scaler.transform(data[colnames].values), columns=colnames)
    data["veracity"] = _ver
    data = data.melt(
        id_vars=["veracity"], value_vars=colnames, var_name="feature", value_name="feature_value"
    )

    # Plot violins across features, splitting real and fake
    plt.cla()
    plt.clf()
    return sns.violinplot(
        x="feature_value",
        y="feature",
        hue="veracity",
        data=data,
        split=True,
        figsize=(16, 9),
        sharex=False,
    )


def visualise_pca(real_dataset, fake_dataset=None, mult=1.2, legend=False):
    """
    Visualisation code for PCA
        Passing in only real dataset generates plot of real data.
        Passing in real and fake datasets generates plot of fake data.
    """
    pca = PCA(n_components=2)
    _real = pca.fit_transform(real_dataset)
    _xlim, _ylim = (
        (mult * _real[:, 0].min(), mult * _real[:, 0].max()),
        (mult * _real[:, 1].min(), mult * _real[:, 1].max()),
    )
    _data = _real
    if fake_dataset is not None:
        _data = pca.transform(fake_dataset)

    plt.cla()
    plt.clf()

    # Ignore warnings from scatter_density code
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        norm = ImageNormalize(vmin=0.0, vmax=500, stretch=LogStretch())
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
        density = ax.scatter_density(_data[:, 0], _data[:, 1], cmap=plt.cm.hot, norm=norm)

    if legend:
        fig.colorbar(density, label="Number of points per pixel")
    plt.xlim(_xlim)
    plt.ylim(_ylim)

    return ax
