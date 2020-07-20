# %%
from database import Database
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

np.random.seed(12)

d = Database("data/database/Denso2021_SP1_ResourceDatabase_v0.18.xlsx")
df = d.df
df.head(10)

# %%
constraints = {'price': 100_000, 'footprint': 10.0, 'handling': 8, 'separation': 5, 'joining': 3, 'cycle_time': 12.0}

solver = pywraplp.Solver('balancing_robots', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
solver.Clear()

robot = [[]] * len(df)
# create variables
for i in range(len(df)):
    robot[i] = solver.IntVar(0, solver.infinity(), df['name'][i])

print('Number of variables:', solver.NumVariables())

def get_value_for_property(robot_name: str, property_name: str): 
    return df.loc[df.name == str(robot_name), property_name].values[0]

def fitness(robot_name: str):
    fitness = get_value_for_property(robot_name, 'price') / constraints['price'] + get_value_for_property(robot_name, 'footprint') / constraints['footprint']
    return fitness


# create objective function
solver.Minimize(solver.Sum([fitness(robot[i]) for i in range(len(robot))]))

# %%
def sum_property(property_name: str):
    return [robot[i] * df[property_name][i] for i in range(len(df))]


# create constraints
solver.Add(0 <= solver.Sum(sum_property('price')) <= constraints['price'])
solver.Add(0 <= solver.Sum(sum_property('footprint')) <= constraints['footprint'])
solver.Add(constraints['handling'] <= solver.Sum(sum_property('operation_handling')))
solver.Add(constraints['separation'] <= solver.Sum(sum_property('operation_separation')))
solver.Add(constraints['joining'] <= solver.Sum(sum_property('operation_joining')))


# %%
result_status = solver.Solve()
# The problem has an optimal solution.
# assert result_status == pywraplp.Solver.OPTIMAL

# The solution looks legit (when using solvers others than
# GLOP_LINEAR_PROGRAMMING, verifying the solution is highly recommended!).
assert solver.VerifySolution(1e-4, True)

print('Solution:')
print('Number of robots:', sum([robot[i].solution_value() for i in range(len(robot))]))

price = 0
footprint = 0
handling = 0
separation = 0
joining = 0
robots = []
print('-' * 65)
for i in range(len(df)):
    if robot[i].solution_value() > 0:
        c = df.loc[df.name == str(robot[i]), 'price'].values[0]
        s = df.loc[df.name == str(robot[i]), 'footprint'].values[0]
        handling += df.loc[df.name == str(robot[i]), 'operation_handling'].values[0] * robot[i].solution_value()
        separation += df.loc[df.name == str(robot[i]), 'operation_separation'].values[0] * robot[i].solution_value()
        joining += df.loc[df.name == str(robot[i]), 'operation_joining'].values[0] * robot[i].solution_value()
        robots.append(str(robot[i]))
        print('{:<10} | num: {:<5} | price: {:<5.2f} € | footprint: {:<5} m2'.format(str(robot[i]), robot[i].solution_value(), c, s))
        price += c
        footprint += s

print('-' * 65)
print('Price =', price, '€')
print('Footprint =', round(footprint, 3), 'm2')
print('Handling =', handling, '/', constraints['handling'])
print('Separation =', separation, '/', constraints['separation'])
print('Joining =', joining, '/', constraints['joining'])


# **Objective here is to minimize the fitness function**

# df.loc[df.name.isin(robots), :]
