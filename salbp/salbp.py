"""
Created on May 13, 2019

@author: maa
@attention: SALBP scheduling and balancing test
@url: https://assembly-line-balancing.de/
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.0.0

#############################################################################################

History:
- v1.0.3: update noSuccessor function to be more pythonic
- v1.0.2: get nodes without predecessor
- v1.0.1: collect data in dict
- v1.0.0: first init
"""

import plotly as py
import plotly.graph_objs as go
# import plotly.io as pio

import numpy as np


class SALBP:
    def __init__(self):
        tags = ['tasks', 'cycleTime', 'taskTime', 'precedence']
        self.data = {t: 0 for t in tags}

    def loadData(self, filepath: str) -> None:
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
                self.data['tasks'] = int(buffer[i + 1])

            elif buffer[i].replace('\n', '') in ['<cycle time>']:
                self.data['cycleTime'] = int(buffer[i + 1])

            elif buffer[i].replace('\n', '') in ['<task times>']:
                task_time = {}
                for j in range(1, self.data['tasks'] + 1):
                    task, time = buffer[i + j].replace(',', ' ').split()
                    task, time = int(task), int(time)
                    task_time[task] = time
                self.data['taskTime'] = task_time

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


    def plotPrecedance(self):
        """
        plot precedence graph
        """
        def noPredecessor(data:dict):
            """
            param:
                data (dict): data to search in
            return:
                list with nodes wihout predecessor
            """
            tasks = list(data.keys())
            successors = [x for s in data.values() for x in s]

            # for s in successors:
            #     if tasks.count(s):
            #         tasks.remove(s)
            return list(set(tasks) - set(successors))

        l = noPredecessor(self.data['precedence'])
        print(l)


if __name__ == "__main__":
    s = SALBP()
    s.loadData('/home/maa/git/SALBP/data/small_data_set_n_20/instance_n=20_1.alb')
    print(s.data)
    print('#' * 100)
    s.plotPrecedance()
