import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.pylab as pylab
import scipy as sc

params = {
    "legend.fontsize": "large",
    "figure.figsize": (9, 6),
    "axes.labelsize": "large",
    "axes.titlesize": "large",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
    "axes.labelpad": 12.5,
}
pylab.rcParams.update(params)


def load_data(filename="HeartDisease.csv"):
    df = pd.read_csv(filename)
    return df


def data_to_numerical(df, skip_feature=None):
    df = df.drop(columns=skip_feature, axis=1)
    data = df.to_numpy(dtype=np.float32)

    return data


def standardize_data(df, skip_feature=None):
    data = data_to_numerical(df, skip_feature=skip_feature)
    n, m = np.shape(data)

    data_standardized = data - np.ones((n, 1)) * data.mean(axis=0)
    data_standardized /= np.std(data_standardized, axis=0, ddof=1)  # is ddof supposed to be 1??

    return data_standardized


def svd_transposed(data):
    u, s, v = sc.linalg.svd(data, full_matrices=False)
    return u, s, v.T


def variance_explained(data):
    _, s, _ = svd_transposed(data)

    rho = (s * s) / (s * s).sum()

    return rho


def plot_variance_explained(data, threshold=0.9, name_to_save="variance_explained.png"):
    rho = variance_explained(data)
    cum_rho = np.cumsum(rho)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(rho) + 1), rho, "x-", color="red")
    plt.plot(range(1, len(rho) + 1), cum_rho, "o-", color="blue")
    plt.plot([1, len(rho)], [threshold, threshold], "k--")

    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained")

    plt.legend(
        ["Individual", "Cumulative", "Threshold"],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )

    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/" + name_to_save)


def component_coefficients(v, idx):
    coeff = v[:, idx]

    return coeff


def plot_comp_coefficients(
    data, n_of_components=2, name_to_save="comp_coefficients.png", barwidth=0.2
):
    _, _, v = svd_transposed(data)
    _, m = data.shape
    pcs = [i for i in range(n_of_components)]
    legend_names = ["PC" + str(i + 1) for i in pcs]
    attribute_names = [
        "sbp",
        "tobacco",
        "ldl",
        "adiposity",
        "famhist",
        "typea",
        "obesity",
        "alcohol",
        "age",
    ]

    bar_positions = np.arange(1, m + 1)
    colors = ["blue", "red"]

    plt.figure(figsize=(8, 6))
    for i, pc in enumerate(pcs):
        plt.barh(
            bar_positions + pc * barwidth,
            component_coefficients(v, pc),
            height=barwidth,
            color=colors[i % len(colors)],
        )

    plt.yticks(
        bar_positions + barwidth * n_of_components / 2,
        attribute_names,
    )
    plt.ylabel("Attributes")
    plt.xlabel("Component Coefficients")
    plt.xlim(-0.6, 0.6)
    plt.legend(
        legend_names,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
    )
    plt.tight_layout()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/" + name_to_save)


def project_data(data):
    _, _, v = svd_transposed(data)
    data_projected = np.dot(data, v[:, :2])

    return data_projected


def plot_projected_data(data, chd_values, name_to_save="projected_data.png"):
    data_projected = project_data(data)
    colors = ["blue" if chd == 0 else "red" for chd in chd_values]

    plt.figure(figsize=(8, 6))
    plt.scatter(data_projected[:, 0], data_projected[:, 1], alpha=0.5, color=colors)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    blue_dot = mlines.Line2D(
        [],
        [],
        color="blue",
        marker="o",
        linestyle="None",
        markersize=10,
        label="CHD Absent",
        alpha=0.5,
    )
    red_dot = mlines.Line2D(
        [],
        [],
        color="red",
        marker="o",
        linestyle="None",
        markersize=10,
        label="CHD Present",
        alpha=0.5,
    )
    plt.legend(
        handles=[blue_dot, red_dot],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
    )
    plt.ylim(-4, 6)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/" + name_to_save)


if __name__ == "__main__":
    features_to_skip = ["chd"]
    df = pd.read_csv("HeartDisease.csv")
    chd_values = df["chd"].values

    data = standardize_data(df, skip_feature=features_to_skip)

    # plot_variance_explained(data)
    # plot_comp_coefficients(data)
    # plot_projected_data(data, chd_values)

    u, s, v = svd_transposed(data)

    eigenvectors = v[:, :2]

    params = {
        "legend.fontsize": 35,
        "axes.labelsize": 35,
        "axes.titlesize": 35,
        "xtick.labelsize": 35,
        "ytick.labelsize": 35,
        "axes.labelpad": 12.5,
    }
    pylab.rcParams.update(params)

    plt.figure(figsize=(18, 18), dpi=90)
    plt.grid(linewidth=3)

    attribute_names = [
        "sbp",
        "tobacco",
        "ldl",
        "adiposity",
        "famhist",
        "typea",
        "obesity",
        "alcohol",
        "age",
    ]
    # eigenvectors
    for i in range(eigenvectors.shape[0]):
        plt.arrow(0, 0, eigenvectors[i, 0], eigenvectors[i, 1], fc='black', ec='black', linewidth=3)
        scale1 = 1.15
        scale2 = 1.15
        if attribute_names[i] == "ldl":
            scale1 = 1.05
            scale2 = 1.22
        if attribute_names[i] == "typea":
            scale1 = 0.1
            scale2 = 1.25
        if attribute_names[i] == "adiposity":
            scale1 = 1.4
            scale2 = 1.35
        if attribute_names[i] == "obesity":
            scale1 = 1.6
            scale2 = 1.2
        if attribute_names[i] == "tobacco":
            scale1 = 1.6
            scale2 = 1.08
        if attribute_names[i] == "alcohol":
            scale1 = 1.1
            scale2 = 1.08
        plt.text(scale1 * eigenvectors[i, 0], scale2 * eigenvectors[i, 1], attribute_names[i], fontsize=35, bbox=dict(facecolor='red', alpha=0.5))

    circle = plt.Circle((0, 0), 1, color='blue', fill=False, linewidth=3)
    plt.gca().add_artist(circle)
    plt.axis('equal')
    plt.xlim(-1.15, 1.15)
    plt.ylim(-1.15, 1.15)
    plt.xlabel('PC1', fontsize=40)
    plt.ylabel('PC2', fontsize=40)
    plt.tight_layout()
    plt.savefig("images/" + "directions.png")
    plt.show()
