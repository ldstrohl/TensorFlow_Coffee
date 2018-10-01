# Data Read for Coffee Experiment
import pandas as pd


def data_read(filename):
    data = pd.read_csv(filename)
    return data
    