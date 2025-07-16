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
    metrics = [
        'duration_distribution',
        'velocity_distribution',
        'pitch_distribution',
        'pitch_distribution_no_octaves',
        'tension_distribution',
        'functionality_distribution'
    ],
    rows = 2,
    col = 3,
    figsize = (15, 10),
):
    data_dict = json.load(open(file_path, 'r'))
    plt.figure(figsize=figsize)
    for i, metric in enumerate(metrics, 1):
        plt.subplot(rows, col, i)
        sns.histplot(data_dict[metric], kde=True, bins=50)
        plt.title(f"{metric}")
        plt.xlabel(metric)
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()