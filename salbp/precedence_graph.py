"""
Created on May 13, 2019

@author: maa
@attention: SALBP scheduling and balancing test
@url: https://assembly-line-balancing.de/
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.0.0

#############################################################################################
"""

# %%
import numpy as np
import plotly as py
import plotly.graph_objs as go
from graphviz import Digraph


class PrecedenceGraph:
    def __init__(self):
        tags = ['num_tasks', 'cycle_time', 'task_time', 'precedence']
        self.data = {t: 0 for t in tags}

    def load_data(self, filepath: str) -> None:
        """
        param:
            filepath (str): path to file

        return: None
        """
        with open(filepath, 'r') as file:
            buffer = file.readlines()
            file.close()

        for i in range(len(buffer)):
            if buffer[i].replace('\n', '') in ['<number of tasks>']:
                self.data['num_tasks'] = int(buffer[i + 1])

            elif buffer[i].replace('\n', '') in ['<cycle time>']:
                self.data['cycle_time'] = int(buffer[i + 1])

            elif buffer[i].replace('\n', '') in ['<task times>']:
                task_time = {}
                for j in range(1, self.data['num_tasks'] + 1):
                    task, time = buffer[i + j].replace(',', ' ').split()
                    task, time = int(task), int(time)
                    task_time[task] = time / 10
                self.data['task_time'] = task_time

            elif buffer[i].replace('\n', '') in ['<precedence relations>']:
                j = 1
                precedence = {}
                successor = []
                while not buffer[i + j] in ['\n']:
                    p1, p2 = buffer[i + j].replace(',', ' ').split()
                    p1, p2 = int(p1), int(p2)
                    if p1 not in precedence.keys():
                        precedence[p1] = []
                    precedence[p1].append(p2)
                    j += 1
                self.data['precedence'] = precedence

    def no_predecessor(self, data: dict) -> list:
        """
        param:
            data (dict): data to search in
        return:
            list with nodes wihout predecessor
        """
        tasks = list(data.keys())
        successors = [x for s in data.values() for x in s]

        return list(set(tasks) - set(successors))

    def task_description(self, x: int) -> str:
        return 'task_' + str(x)

    def node_description(self, x: int) -> str:
        return 'task_' + str(x) + ':\n' + str(self.data['task_time'][x]) + ':\n' + str(self.data['task_type'][x])


# %%
if __name__ == "__main__":
    pg = PrecedenceGraph()
    pg.load_data('../data/small_data_set_n_20/instance_n=20_1.alb')
    print(pg.data)
    print('#' * 100)
    pg.no_predecessor(pg.data['precedence'])
    pg.data['task_type'] = {1: 'transfer',
                            2: 'transfer',
                            3: 'transfer',
                            4: 'transfer',
                            5: 'separation',
                            6: 'separation',
                            7: 'separation',
                            8: 'separation',
                            9: 'separation',
                            10: 'handling',
                            11: 'handling',
                            12: 'joining',
                            13: 'joining',
                            14: 'transfer',
                            15: 'transfer',
                            16: 'separation',
                            17: 'separation',
                            18: 'separation',
                            19: 'separation',
                            20: 'separation'}

    dot = Digraph()
    for i in pg.data['task_time']:
        dot.node(pg.task_description(i), pg.node_description(i))

    for t, successors in pg.data['precedence'].items():
        for successor in successors:
            dot.edge(pg.task_description(t), pg.task_description(successor))

# %%
    dot