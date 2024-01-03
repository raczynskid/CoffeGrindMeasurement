import pandas as pd
import numpy as np

def particles_to_dataframe(points: np.array, sizes: np.array) -> pd.DataFrame:
    # reformat particle coordinates and sizes into a pandas dataframe
    return pd.DataFrame({'point_x': [point[0]for point in points], "points_y": [point[1]for point in points], 'point size': sizes})
