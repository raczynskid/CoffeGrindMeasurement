import pandas as pd
import numpy as np

def particles_to_dataframe(points: np.array, sizes: np.array) -> pd.DataFrame:
    # reformat particle coordinates and sizes into a pandas dataframe
    return pd.DataFrame({'point_x': [point[0]for point in points], "points_y": [point[1]for point in points], 'point size': sizes})

def calculate_zscore(sizes: np.array) -> pd.DataFrame:
    # for each point calculate relationship to the mean
    size_series = pd.DataFrame({'point size': sizes})
    size_series.sort_values(by=["point size"], inplace=True)
    size_series["zscore"] = (size_series - size_series.mean())/size_series.std()
    return size_series

def drop_quantile(sizes: np.array, quantile: float=0.8) -> pd.Series:
    # calculate a top quantile for the data and drop it
    # only top quantile is dropped as the outliers to remove are clumps of particles
    size_series = pd.DataFrame({'point size': sizes})
    quantile_threshold = size_series["point size"].quantile(0.8)
    return size_series[size_series["point size"] < quantile_threshold]

def calculate_variance(sizes: pd.DataFrame) -> float:
    # calculate Variance using the point sizes list
    size_series = sizes["point size"]
    # calculate mean of dataset
    size_mean = size_series.mean()
    # for each point of dataset ((X - mean)^2) / number of datapoints
    return size_series.apply(lambda x: pow((x - size_mean), 2)).sum() / size_series.count()
    