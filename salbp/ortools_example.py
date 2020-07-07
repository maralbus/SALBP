
# %%
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
import numpy as np
# %%


class Example1:
    def create_data_model(self):
        """Stores the data for the problem."""
        data = {}
        data['constraint_coeffs'] = [
            [5, 7, 9, 2, 1],
            [18, 4, -9, 10, 12],
            [4, 7, 3, 8, 5],
            [5, 13, 16, 3, -7],
        ]
        data['bounds'] = [250, 285, 211, 315]
        data['obj_coeffs'] = [7, 8, 2, 9, 6]
        data['num_vars'] = 5
        data['num_constraints'] = 4
        return data

    def main(self):
        print('#' * 20)
        print('Example 1')
        print('#' * 20)
        solver = pywraplp.Solver('Example', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        # solver = pywraplp.Solver('Example', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
        inf = solver.infinity()
        data = self.create_data_model()
        variables = {}
        for j in range(data['num_vars']):
            variables[j] = solver.IntVar(0.0, inf, f"$x_{j}$")
        print("Number of variables:", solver.NumVariables())

    # Constraints
        for i in range(data['num_constraints']):
            constraint = [data['constraint_coeffs'][i][j] * variables[j] for j in range(data['num_vars'])]
            solver.Add(sum(constraint) <= data['bounds'][i])
        print("Number of contstraints:", solver.NumConstraints())

        # Objective
        objective = [data['obj_coeffs'][i] * variables[i] for i in range(data['num_vars'])]
        solver.Maximize(sum(objective))

        status = solver.Solve()
        if status == pywraplp.Solver.OPTIMAL:
            print('Solution status:')
            print('Objective value =', solver.Objective().Value())
            for j in range(len(variables)):
                print(f'{variables[j]} = {variables[j].solution_value()}')
            print(f'Problem solved in {solver.wall_time()} milliseconds')
            print(f'Problem solved in {solver.iterations()} iterations')
            print(f'Problem solved in {solver.nodes()} branch-and-bound nodes')
        else:
            print('The problem does not have an optimal solution.')


class Example2:
    def main(self):
        print('#' * 20)
        print('Example 2')
        print('#' * 20)
        # Data
        costs = [
            [90, 80, 75, 70],
            [35, 85, 55, 65],
            [125, 95, 90, 95],
            [45, 110, 95, 115],
            [50, 100, 90, 100],
        ]
        num_workers = len(costs)
        num_tasks = len(costs[0])

        # Solver
        # Create the mip solver with the CBC backend.
        solver = pywraplp.Solver('simple_mip_program',
                                 pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        # Variables
        # x[i, j] is an array of 0-1 variables, which will be 1
        # if worker i is assigned to task j.
        x = {}
        for i in range(num_workers):
            for j in range(num_tasks):
                x[i, j] = solver.IntVar(0, 1, '')

        # Constraints
        # Each worker is assigned to at most 1 task.
        for i in range(num_workers):
            solver.Add(solver.Sum([x[i, j] for j in range(num_tasks)]) <= 1)

        # Each task is assigned to exactly one worker.
        for j in range(num_tasks):
            solver.Add(solver.Sum([x[i, j] for i in range(num_workers)]) == 1)

        # Objective
        objective_terms = []
        for i in range(num_workers):
            for j in range(num_tasks):
                objective_terms.append(costs[i][j] * x[i, j])
        solver.Minimize(solver.Sum(objective_terms))

        # Solve
        status = solver.Solve()

        # Print solution.
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            print('Total cost = ', solver.Objective().Value(), '\n')
            for i in range(num_workers):
                for j in range(num_tasks):
                    # Test if x[i,j] is 1 (with tolerance for floating point arithmetic).
                    if x[i, j].solution_value() > 0.5:
                        print('Worker %d assigned to task %d.  Cost = %d' %
                              (i, j, costs[i][j]))


class Example3:
    def main(self):
        print('#' * 20)
        print('Example 3')
        print('#' * 20)
        # Data
        costs = [
            [90, 80, 75, 70],
            [35, 85, 55, 65],
            [125, 95, 90, 95],
            [45, 110, 95, 115],
            [50, 100, 90, 100],
        ]
        num_workers = len(costs)
        num_tasks = len(costs[0])

        # Model
        model = cp_model.CpModel()

        # Variables
        x = []
        for i in range(num_workers):
            t = []
            for j in range(num_tasks):
                t.append(model.NewBoolVar('x[%i,%i]' % (i, j)))
            x.append(t)

        # Constraints
        # Each worker is assigned to at most one task.
        for i in range(num_workers):
            model.Add(sum(x[i][j] for j in range(num_tasks)) <= 1)

        # Each task is assigned to exactly one worker.
        for j in range(num_tasks):
            model.Add(sum(x[i][j] for i in range(num_workers)) == 1)

        # Objective
        objective_terms = []
        for i in range(num_workers):
            for j in range(num_tasks):
                objective_terms.append(costs[i][j] * x[i][j])
        model.Minimize(sum(objective_terms))

        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        # Print solution.
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print('Total cost = %i' % solver.ObjectiveValue())
            print()
            for i in range(num_workers):
                for j in range(num_tasks):
                    if solver.BooleanValue(x[i][j]):
                        print('Worker ', i, ' assigned to task ', j, '  Cost = ',
                              costs[i][j])
        else:
            print('No solution found.')

# %%
ex1 = Example1()
ex1.main()
ex2 = Example2()
ex2.main()
ex3 = Example3()
ex3.main()

# %% 16.5 ms +- 97.2 us per loop (mean +- std. dev. of 7 runs, 100 loops each)
%%capture capt1
%timeit ex1.main()

# %% 1.75 ms +- 60.1 us per loop (mean +- std. dev. of 7 runs, 100 loops each)
%%capture capt2
%timeit ex2.main()

# %% 1.99 ms +- 51.4 us per loop (mean +- std. dev. of 7 runs, 100 loops each)
%%capture capt3
%timeit ex3.main()

# %%
cost = [[90, 76, 75, 70, 50, 74, 12, 68],
        [35, 85, 55, 65, 48, 101, 70, 83],
        [125, 95, 90, 105, 59, 120, 36, 73],
        [45, 110, 95, 115, 104, 83, 37, 71],
        [60, 105, 80, 75, 59, 62, 93, 88],
        [45, 65, 110, 95, 47, 31, 81, 34],
        [38, 51, 107, 41, 69, 99, 115, 48],
        [47, 85, 57, 71, 92, 77, 109, 36],
        [39, 63, 97, 49, 118, 56, 92, 61],
        [47, 101, 71, 60, 88, 109, 52, 90]]
# Variables
x = []
for i in range(len(cost)):
    t = []
    for j in range(len(cost[0])):
      t.append(1)
    x.append(t)
