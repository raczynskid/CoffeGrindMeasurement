import pandas as pd
import numpy as np

def particles_to_dataframe(points: np.array, sizes: np.array) -> pd.DataFrame:
    # reformat particle coordinates and sizes into a pandas dataframe
    return pd.DataFrame({'point_x': [point[0]for point in points], "points_y": [point[1]for point in points], 'point size': sizes})

def calculate_variance(point_list : pd.DataFrame) -> float:
    # calculate Variance using the point sizes list
    point_size = point_list["point size"]
    # calculate mean of dataset
    size_mean = point_size.mean()
    # for each point of dataset ((X - mean)^2) / number of datapoints
    return point_size.apply(lambda x: pow((x - size_mean), 2)).sum() / point_size.count()
    