# %%
from plotly import graph_objs as go
import pandas as pd

import numpy as np
import operator
import functools


class Database:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.operations = ['providesHandlingOperation',
                           'providesJoiningOperation',
                           'providesSeparationOperation']
        self.columns_rename = {'hasName': 'name',
                               'hasPrice': 'price',
                               'hasCycleTime': 'cycle_time',
                               'hasFootprint': 'footprint',
                               'hasMaximumVelocity': 'velocity',
                               'hasMaximumAcceleration': 'acceleration',
                               'isProcessTypeMD': 'module',
                               'hasReach': 'reach',
                               'providesHandlingOperation': 'operation_handling',
                               'providesJoiningOperation': 'operation_joining',
                               'providesSeparationOperation': 'operation_separation',
                               }
        self.columns_csv = ['isProcessTypeMD', 'hasName', 'hasPrice', 'hasFootprint', 'providesOperation', 'hasCycleTime', 'hasMaximumVelocity', 'hasMaximumAcceleration', 'hasReach']
        self.df = self._read_database()
        self.df = self._postprocessing(self.df)

    def _read_database(self):
        df = pd.read_excel(self.filepath, sheet_name='Resources', header=1, usecols=self.columns_csv)
        df = df.drop([0, 1], axis=0).reset_index(drop=True)
        df.hasFootprint[df.hasFootprint.isnull()] = '0x0'
        df.hasFootprint = df.hasFootprint.apply(lambda x: self._product_footprint(x))
        return df

    def _product_footprint(self, iterable) -> int:
        """
        return product of iterable

        Args:
            iterable (list): list to build product from

        Returns:
            int: product
        """
        prod = functools.reduce(operator.mul, map(int, iterable.split('x')))
        return prod * 1e-3 * 1e-3

    def _check_operation(self, data: str, operation_name: str) -> bool:
        """
        check if the data provides the operation for a given operation_name

        Args:
            data (str): the data which is checked if it provides the operation
            operation_name (str): name of the operation

        Returns:
            bool: 1 if operation is provided else 0
        """
        if type(data) is not str:
            return 0
        if data.count(operation_name):
            return 1
        else:
            return 0

    def calc_cycle_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        calculate the cycle time for a pandas dataframe based on 'velocity', 'acceleration' and 'reach' of the resource

        Args:
            df (pd.DataFrame): the resource database as dataframe

        Returns:
            pd.DataFrame: resource database as dataframe with new column 'cycle_time'
        """
        df['cycle_time'] = df.apply(lambda x: self.calc_time_for_distance(target_distance=x['reach'], velocity=x['velocity'], acceleration=x['acceleration']), axis=1)
        return df

    def define_operation(self, df: pd.DataFrame) -> pd.DataFrame:
        for operation in self.operations:
            df[self.columns_rename[operation]] = df.loc[:, 'providesOperation'].apply(lambda x: self._check_operation(x, operation))
        return df

    def calc_time_for_distance(self, target_distance: float, velocity: float, acceleration: float) -> float:
        """
        calculate the minimum time necessary to achieve the target distance

        Args:
            target_distance (float): target distance [mm]
            velocity (float): maximum velocity [mm/s]
            acceleration (float): maximum acceleration [mm/s^2]

        Returns:
            float: minimum time necessary to achieve the target distance [s]
        """
        # Time for acceleration ramp until the robot reaches the final velocity
        # t = v/a
        time_ramp = velocity / acceleration

        # Distance which the robot needs to reach the final velocity
        # s = v^2 / 2a
        # s = 0.5a * t^2 | t = v/a --> s = v^2 / 2a
        distance_for_target_vel = (velocity ** 2) / (2 * acceleration)

        # Time of the constant part of the movement
        time_constant = (target_distance - 2 * distance_for_target_vel) / velocity

        if distance_for_target_vel * 2 >= target_distance:
            # final time
            time = time_ramp * 2
            return time
        else:
            # final time
            time = 2 * time_ramp + time_constant
            return time

    def calc_distance_in_time(self, target_time: float, velocity: float, acceleration: float) -> float:
        """
        calculate the maximum distance possible in the given target time

        Args:
            target_time (float): available time [s]
            velocity (float): maximum velocity [mm/s]
            acceleration (float): maximum acceleration [mm/s^2]

        Returns:
            float: maximum distance possible in the given time [mm]
        """
        # Time for acceleration until the robot reaches the final velocity
        time_ramp = velocity / acceleration

        # If the acceleration time is less than the target time -> Max speed will be reached.
        if time_ramp * 2 <= target_time:
            # Distance which the robot needs to reach the final velocity
            # s = v^2 / 2a
            # s = 0.5a * t^2 | t = v/a --> s = v^2 / 2a
            distance_max_vel = (velocity ** 2) / (2 * acceleration)

            # Time of the constant part of the movement
            # minimum: target_time - 2 * time_ramp = 0
            distance_constant = velocity * (target_time - 2 * time_ramp)

            max_distance = distance_max_vel * 2 + distance_constant
        else:
            # $t_{target} / 2 -> one ramp for acceleration, one for braking
            time_ramp = target_time / 2

            # using s = 0,5a * t^2
            # with $t_{acceleration} = t_{braking}$:
            # s = 2 * 0,5a * t^2 = a * t^2
            max_distance = acceleration * time_ramp ** 2

        return max_distance

    def _postprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        df.rename(columns=self.columns_rename, inplace=True)
        df = self.define_operation(df=df)
        df = self.calc_cycle_time(df=df)
        return df


# %%
if __name__ == "__main__":
    d = Database("../data/database/Denso2021_SP1_ResourceDatabase_v0.18.xlsx")
    df = d.df
    print(df)

# %%
# x = np.linspace(1, 15, 15)
# l = [d.calc_distance_in_time(target_time=time, velocity=5e3, acceleration=1e3) for time in x]
# data = go.Scatter(x=x, y=l)
# fig = go.Figure(data=data)
# fig.update_layout(title="Distance in Time", xaxis_title="Time/s", yaxis_title="Distance/mm")
# fig.show()

# x = np.linspace(1, 40_000, 1000)
# l = [d.calc_time_for_distance(target_distance=distance, velocity=5e3, acceleration=1e3) for distance in x]
# data = go.Scatter(x=x, y=l)
# fig = go.Figure(data=data)
# fig.update_layout(title="Time for Distance", xaxis_title="Distance/mm", yaxis_title="Time/s")
# fig.show()