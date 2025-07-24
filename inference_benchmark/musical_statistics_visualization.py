"""
Helper functions for visualizing statistics.
"""
import os
from pathlib import Path
import glob
import json
import pretty_midi
import music21 as m21
from itertools import combinations
import numpy as np
import pandas as pd
import pretty_midi
import matplotlib.pyplot as plt
import seaborn as sns
import math
from tqdm import tqdm

def distribution_histograms_collections(
    file_path,
    title,
    metrics = [
        'duration_distribution',
        'velocity_distribution',
        'pitch_distribution',
        'pitch_distribution_no_octaves',
        'tension_distribution',
        'functionality_distribution'
    ],
    rows = 2,
    cols = 4,
    figsize = (9, 6),
):
    bins = 30
    data_dict = json.load(open(file_path, 'r'))
    plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=16)
    for i, metric in enumerate(metrics[:4], 1):
        plt.subplot(rows, cols, i)
        sns.histplot(data_dict[metric], kde=True, bins=bins, stat="density")
        plt.title(f"{metric}")
        plt.xlabel(metric)
        plt.ylabel("Frequency")
    
    # Plotting tension and functionality distributions
    plt.subplot(rows, cols, 5)
    sns.histplot(data_dict['tension_distribution'], kde=True, bins=np.linspace(0, 25, 26), stat="density")
    # x-axis limits for better visibility
    plt.xlim(0, 25)
    plt.ylim(0, 0.6)
    plt.title("Tension Distribution")

    plt.subplot(rows, cols, 6)
    sns.histplot(data_dict['functionality_distribution'], kde=True, bins=np.linspace(0, 100, 11), stat="density")
    plt.xlim(0, 100)
    plt.ylim(0, 0.2)
    plt.title("Functionality Distribution")

    plt.tight_layout()
    plt.show()


def distribution_histograms_by_metric(
    file_paths,
    file_name,
    title,
    metric,
    rows = 2,
    cols = 3,
    figsize = (9, 6),
    bins = 30,
    xlim = (0, 100),
    ylim = (0, 0.6)
):
    plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=16)
    for i, file_path in enumerate(file_paths, 1):
        title = file_path.replace('outputs/0722_full_run', '').replace('_dataset', '').replace('_skyline_top2', '')
        data_dict = json.load(open(Path(file_path) / file_name, 'r'))
        plt.subplot(rows, cols, i)
        sns.histplot(data_dict[metric], kde=True, bins=bins, stat="density")
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.title(f"{title}")
        plt.xlabel(metric)
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def distribution_over_time_by_metric(
    file_paths,
    file_name,
    title,
    metric,
    rows = 2,
    cols = 3,
    figsize = (9, 6),
    xlim = (0, 100),
    ylim = (0, 0.6)
):
    plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=16)
    
    for i, file_path in enumerate(file_paths, 1):
        title = file_path.replace('outputs/0722_full_run/', '').replace('_dataset', '').replace('_skyline_top2', '')
        data_dict = json.load(open(Path(file_path) / file_name, 'r'))

        df = pd.DataFrame(data_dict[metric]).rolling(window=5).mean().reset_index()
        # print(df)

        plt.subplot(rows, cols, i)
        sns.lineplot(df, x='index', y=0)
        plt.axvline(x=50, color='red', linestyle='--')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.title(f"{title}")
        plt.xlabel(metric)
        plt.ylabel("Score")
    plt.tight_layout()
    plt.show()