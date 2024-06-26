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

    def get_ptp_prize_cost_difference(self, path, multiplier=10):
        ''' Calculates the difference between the sum of prizes and the sum of costs of all vertices in the path
        Input:
            path: list of vertices in the path
        Output:
            prize_cost_diff <int>: difference between the sum of prizes and the sum of costs of the vertices in the path
        '''
        prize = self.get_path_prize(path)
        cost = self.get_path_cost(path)
        prize_cost_diff = multiplier*prize - cost
        return prize_cost_diff, prize, cost

    def recursive_greedy(self, s: int, t: int, b: int, r_set: set, n_iter:int=0, prefix:str=""):
        ''' Calculates orienteering path from s to t within budget b.
        '''
        print (f"{prefix}s={s}\tt={t}\tb={b}\tr_set={r_set}\titer={n_iter}")
        c = self.costs 
        if c[s,t] > b: # line 3
            print(f"{prefix}cost: {c[s,t]} > budget: {b} ... Infeasible")
            return None # no path from s to t within budget b
        path = [s, t] # line 5
        if n_iter == 0:
            print (f'{prefix}Iter == 0 ... return')
            return path
        
        # -- print path
        print(f"{prefix}Initial Path: {path}")

        # -- get m
        m, prize, cost = self.get_ptp_prize_cost_difference(path)
        print(f"{prefix}Prize: {prize}\tCost: {cost}\tm: {m}")

        # -- iterate over all vertices
        v_set = set([i for i in range(0, self.num_nodes)])

        # -- remove the numbers in r_set
        v_set -= r_set
        print (f"{prefix}v_set: {v_set}")

        # -- iterate over v_set
        for v_m in v_set: # line 8
            # try all budget split
            for b_val in range(2, b, 2): # line 9
                r_1 = r_set.union(set([v_m]))
                b_1 = b_val
                b_2 = b - b_val
                path_1 = self.recursive_greedy(s, v_m, b_1, r_1, n_iter-1, prefix+"\t") # line 10
                if path_1 is None:
                    continue
                r_2 = r_set.union(set(path_1))
                path_2 = self.recursive_greedy(v_m, t, b_2, r_2, n_iter-1, prefix+"\t") # line 11
                if path_2 is None:
                    continue
                print (f"{prefix}path_1: {path_1}\tpath_2: {path_2}")
                path_new = path_1 + path_2[1:]
                m_new, prize_new, cost_new = self.get_ptp_prize_cost_difference(path_new)
                print (f"{prefix}New Path: {path_new}")
                print (f"{prefix}Prize New: {prize_new}\tCost New: {cost_new}\tm New: {m_new}")
                print (f"{prefix}Old Path: {path}")
                print (f"{prefix}Old prize: {prize}\tOld Cost: {cost}\tOld m: {m}")
                if m_new > m: # line 12
                    path = path_new
                    m = m_new
                    prize = prize_new
                    cost = cost_new
        return path

def main(n=7, budget=10, n_iterations=4):
    graph_data = get_graph_data(n)
    graph_data['prizes'][1] = 0
    graph_data['prizes'][2] = 0
    print ("graph_data: ", graph_data)
    print ("-"*100)
    rao = RecursiveAlgorithm(graph_data)

    r_set = set([graph_data['home'], graph_data['goal']])

    path = rao.recursive_greedy(graph_data['home'], graph_data['goal'], b=budget, r_set=r_set, n_iter=n_iterations)
    print ("Calculated path: ", path)
    print ("Path prize: ", rao.get_path_prize(path))
    print ("Path cost: ", rao.get_path_cost(path))

if __name__=='__main__':
    budget = 30
    n_vertices = 10 #7,10
    n_iter = 4
    main(n_vertices, budget, )






