#!/usr/bin/env python
# coding: utf-8


# %%
%%capture info
%%timeit
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

np.random.seed(12)
samples = 5000
probability = 0.2

index = ['name', 'cost', 'size', 'handling', 'positioning', 'joining']
names = ['robot' + str(i) for i in range(samples)]
costs = np.random.randint(1000, 10000, size=(samples)).tolist()
handling = np.random.binomial(1, probability, samples)
positioning = np.random.binomial(1, probability, samples)
joining = np.random.binomial(1, probability, samples)
sizes = np.round(np.random.rand((samples)) * 10, 2)

df = pd.DataFrame({'name': names, 'cost': costs, 'size': sizes, 'handling': handling, 'positioning': positioning, 'joining': joining})
df.head(10)


constraints = {'cost': 20000, 'size': 10.0, 'handling': 8, 'positioning': 5, 'joining': 3}


solver = pywraplp.Solver('simple_lp_programm', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
solver.Clear()

robot = [[]] * len(df)
# create variables
for i in range(len(df)):
    robot[i] = solver.IntVar(0, solver.infinity(), df['name'][i])
    
print('Number of variables:', solver.NumVariables())


# ## Fitness Function
# \begin{equation}
# f = \sum_{r \in \mathcal{R}} \frac{c_r}{C_{max}} + \frac{s_r}{S_{max}} \\
# \text{with:} \\
# \text{f = fitness} \\
# \text{c = cost} \\
# \text{s = size}
# \end{equation}

def fitness(robot):
    l = lambda x: df.loc[df.name==str(robot), x].values[0]
    f = l('cost') / constraints['cost'] + l('size') / constraints['size']
    return f


# create objective function
solver.Minimize(solver.Sum( [fitness(robot[i]) for i in range(len(robot))] ))


ll = lambda s: [robot[i] * df[s][i] for i in range(len(df))]


# create constraints
solver.Add(0 <= solver.Sum(ll('cost')) <= constraints['cost'])
solver.Add(0 <= solver.Sum(ll('size')) <= constraints['size'])
solver.Add(constraints['handling'] <= solver.Sum(ll('handling')))
solver.Add(constraints['positioning'] <= solver.Sum(ll('positioning')))
solver.Add(constraints['joining'] <= solver.Sum(ll('joining')))


result_status = solver.Solve()
# The problem has an optimal solution.
# assert result_status == pywraplp.Solver.OPTIMAL

# The solution looks legit (when using solvers others than
# GLOP_LINEAR_PROGRAMMING, verifying the solution is highly recommended!).
assert solver.VerifySolution(1e-4, True)

print('Solution:')
print('Number of robots:', sum([robot[i].solution_value() for i in range(len(robot))]))

cost = 0
size = 0
handling = 0
positioning = 0
joining = 0
robots = []
for i in range(len(df)):
    if robot[i].solution_value() > 0: 
        c = df.loc[df.name==str(robot[i]), 'cost'].values[0]
        s = df.loc[df.name==str(robot[i]), 'size'].values[0]
        handling += df.loc[df.name==str(robot[i]), 'handling'].values[0] * robot[i].solution_value()
        positioning += df.loc[df.name==str(robot[i]), 'positioning'].values[0] * robot[i].solution_value()
        joining += df.loc[df.name==str(robot[i]), 'joining'].values[0] * robot[i].solution_value()
        robots.append(str(robot[i]))
        print('{:<10} | num: {:<5} | cost: {:<5} € | size: {:<5} m2'.format(str(robot[i]), robot[i].solution_value(), c, s))
        cost += c
        size += s

print('Cost =', cost, '€')
print('Size =', round(size, 3), 'm2')
print('Handling =', handling, '/', constraints['handling'])
print('Positioning =', positioning, '/', constraints['positioning'])
print('Joining =', joining, '/', constraints['joining'])


# **Objective here is to minimize the fitness function**

# df.loc[df.name.isin(robots), :]


# %%
