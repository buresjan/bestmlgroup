import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.pylab as pylab
import scipy as sc

def load_data(filename="HeartDisease.csv"):
    df = pd.read_csv(filename)
    return df

heart_disease_data = load_data()

heart_disease_data_without_chd = heart_disease_data.drop(columns=['chd'])

def plot_histograms_3x3(dataframe):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    for i, column in enumerate(dataframe.columns):
        row = i // 3
        col = i % 3
        axes[row, col].hist(dataframe[column], bins=20, color='skyblue', edgecolor='black')
        # axes[row, col].set_title(f"Distribution of '{column}'", fontsize=18)
        axes[row, col].set_xlabel(column, fontsize=18)
        axes[row, col].set_ylabel('Frequency', fontsize=18)
    plt.tight_layout()
    plt.savefig("!!!HISTOGRAMS!!!.png")
    plt.show()
