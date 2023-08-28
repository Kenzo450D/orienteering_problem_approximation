import numpy as np
import sys
from get_graph import get_graph_data


class RecursiveAlgorithm:
    def __init__(self, graph_data):
        self.costs = graph_data['costs']
        self.prizes = graph_data['prizes']
        self.depot_idx = graph_data['depot_idx']
        self.num_nodes = self.costs.shape[0]
        #self.num_customers = self.num_nodes - 1
        self.best_solution = None
        self.best_solution_cost = np.inf
        self.best_solution_weight = np.inf

    def get_path_cost(self, path):
        ''' Calculates the sum of each edges in the path.
        Input:
            path: list of vertices in the path
        Output:
            cost <int>: cost of the edges in the path
        '''
        cost = 0
        for i in range(1, len(path)):
            cost += self.costs[(path[i-1],path[i])]
        return cost

    def get_path_prize(self, path):
        ''' Calculates the sum of prizes of all vertices in the path
        Input:
            path: list of vertices in the path
        Output:
            prize <int>: prizes of the vertices in the path
        '''
        prize = sum([self.prizes[i] for i in path])
        return prize

    def recursive_greedy(self, s: int, t: int, b: int, r_set: set, iter:int=0, prefix:str=""):
        ''' Calculates orienteering path from s to t within budget b.
        '''
        c = self.costs 
        if c[s,t] > b: # line 3
            return None # no path from s to t within budget b
        path = [s, t] # line 5
        if iter == 0:
            return
        
        # -- get m
        m = self.get_path_prize(path)
        # print(f"{prefix}m: {m}")

        # -- iterate over all vertices
        v_set = set([i for i in range(0, self.num_nodes)])

        # -- remove the numbers in r_set
        v_set -= r_set

        # -- iterate over v_set
        for v_m in v_set: # line 8
            # try all budget split
            for b_val in range(2, b, 2): # line 9
                r_1 = r_set.union(set([v_m]))
                b_1 = b_val
                b_2 = b - b_val
                path_1 = self.recursive_greedy(s, v_m, b_1, r_1, iter-1, prefix+"\t") # line 10
                if path_1 is None:
                    continue
                r_2 = r_set.union(set(path_1))
                path_2 = self.recursive_greedy(v_m, t, b_2, r_2, iter-1, prefix+"\t") # line 11
                if path_2 is None:
                    continue
                if self.get_path_prize(set(path_1).union(set(path_2))) > m: # line 12
                    path = path_1 + path_2[1:]
                    m = self.get_path_prize(path)
        return path

def main(n):
    graph_data = get_graph_data(7)
    print ("graph_data: ", graph_data)
    print ("-"*100)
    rao = RecursiveAlgorithm(graph_data)

    r_set = set([graph_data['home'], graph_data['goal']])
    path = rao.recursive_greedy(graph_data['home'], graph_data['goal'], b=14, r_set=r_set, iter=3)
    print ("Calculated path: ", path)
    print ("Path prize: ", rao.get_path_prize(path))
    print ("Path cost: ", rao.get_path_cost(path))

if __name__=='__main__':
    main(7)






