# %%
import pandas as pd
import numpy as np
import operator
import functools

class Database:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = self._read_database()
        self.operations = ['providesHandlingOperation',
                            'providesJoiningOperation',
                            'providesSeparationOperation']

    def _read_database(self):
        columns = ['isProcessTypeMD', 'hasName', 'hasPrice', 'hasFootprint', 'providesOperation', 'hasCycleTime']
        df = pd.read_excel(self.filepath, sheet_name='Resources', header=1, usecols=columns)
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

    def _calc_cycle_time(self, data) -> float:
        data = 0
        return np.random.uniform(low=4.0, high=30.0)

    def define_operation(self, df: pd.DataFrame) -> pd.DataFrame:
        for operation in self.operations:
            df[operation] = df.loc[:, 'providesOperation'].apply(lambda x: d._check_operation(x, operation))
        return df


#%%
if __name__ == "__main__":
    d = Database("../data/database/Denso2021_SP1_ResourceDatabase_v0.18.xlsx")
    df = d.df
    df = d.define_operation(df)
    print(df)

# %%
# 6AxisRobot "VP-5423" in Database pos: 341
df.loc[91, :]
df.loc[91, 'providesOperation'].split(';')


# %%
df['hasCycleTime'] = df['hasCycleTime'].apply(lambda x: np.random.uniform(low=4.0, high=30.0))

# %%
from plotly import graph_objs as go
data = go.Scatter(x=df.index, y=df.hasCycleTime)
fig = go.Figure(data=data)
fig.show()
