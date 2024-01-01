import numpy as np
import gurobipy as gp
import sys
from get_graph import get_graph_data

class OrienteeringMILP:
    def __init__(self, graph_data):
        self.costs = graph_data['costs']
        self.prizes = graph_data['prizes']
        self.depot_idx = graph_data['depot_idx']
        self.num_nodes = self.costs.shape[0]
        self.num_customers = self.num_nodes - 1
        self.best_solution = None
        self.best_solution_cost = np.inf
        self.best_solution_weight = np.inf

    def get_path_prize(self, path):
        ''' Calculates the sum of prizes of all vertices in the path
        Input:
            path: list of vertices in the path
        Output:
            prize <int>: prizes of the vertices in the path
        '''
        prize = sum([self.prizes[i] for i in path])
        return prize

    def solve_milp_path(self, s:int, t:int, b:int):
        ''' Calculates an orienteering path from s to t. Path is within time budget of b
        '''
        model = gp.Model("Orienteering")

        # Variables
        # x_{i,j} \in {0,1}
        x = model.addVars(self.num_nodes, self.num_nodes, vtype=gp.GRB.BINARY, name="x")
        # y_i \in {0,1}
        y = model.addVars(self.num_nodes, vtype=gp.GRB.BINARY, name="y")
        # u_i \in R
        u = model.addVars(self.num_nodes, vtype=gp.GRB.CONTINUOUS, name="u")

        # Objective Function
        model.setObjective(gp.quicksum(self.prizes[i] * y[i] for i in range(self.num_nodes) if i!= s or i!=t), gp.GRB.MAXIMIZE)

        # Constraints
        # Constraint 1: Budget constraint
        model.addConstr(gp.quicksum(self.costs[i, j] * x[i, j] for i in range(self.num_nodes) for j in range(self.num_nodes)) <= b, name="time_budget")

        # Constraint 2: Inbound edge to end vertex should be one
        model.addConstr(gp.quicksum((x[i,t] for i in range(self.num_nodes) if i!=t)) == 1, name="incoming_edges_end")

        # Constraint 3: Outgoing edge from start vertex should be one
        model.addConstr((gp.quicksum(x[s,i] for i in range(self.num_nodes) if i!= s)) == 1, name="outgoing_edges_start")

        # Constraint 4: Inbound edge to the start vertex should be zero
        model.addConstr(gp.quicksum((x[i,s] for i in range(self.num_nodes) if i!=s)) == 0, name="incoming_edges_start")

        # Constraint 5: Outgoing edge from end vertex should be zerp
        model.addConstr((gp.quicksum(x[t,i] for i in range(self.num_nodes) if i!= t)) == 0, name="outgoing_edges_end")

        # Constraint 5: Inbound edge to end vertex should be y[i]
        model.addConstrs((gp.quicksum(x[i, j] for j in range(self.num_nodes) if j!= i) == y[i] for i in range(self.num_nodes) if i!= s and i!=t), name="outgoing_edges")
        model.addConstrs((gp.quicksum(x[i, j] for i in range(self.num_nodes) if i!= j) == y[j] for j in range(self.num_nodes) if j!= s and j!=t), name="incoming_edges")

        # Constraint 6: Subtour elimination
        M = self.num_nodes - 1
        for i in range(0, self.num_nodes):
            for j in range(0, self.num_nodes):
                if i != j and i!=s and j!=s and i!=t and j!=t:
                    model.addConstr(u[i] - u[j] + M * x[i, j] <= M - 1, f"mtz_{i}_{j}")

        model.addConstrs((2 <= u[i] for i in range(self.num_nodes)), name="subtour_elimination_2_lower")
        model.addConstrs((u[i] <= self.num_nodes for i in range(self.num_nodes)), name="subtour_elimination_2_upper")


        # Solve the model
        # model.computeIIS()
        # sys.exit(0)
        model.optimize()
        # print (model.display())

        # Print variables
        # print("Variables:")
        # for var in model.getVars():
            # print(f"{var.VarName} = {var.x}")

        path = []
        # Check the optimization status
        if model.status == gp.GRB.OPTIMAL:
            print("Optimal solution found.")
            _, path = self.get_path_edges(x, s, t)

        elif model.status == gp.GRB.INFEASIBLE:
            print("The model is infeasible.")
        else:
            print("Optimization did not converge to optimality.")

        return path


    def solve_milp(self, s:int, b:int):
        ''' Calculates an orienteering tour from s to s within time budget of b
        '''
        model = gp.Model("Orienteering")

        # Variables
        x = model.addVars(self.num_nodes, self.num_nodes, vtype=gp.GRB.BINARY, name="x")
        y = model.addVars(self.num_nodes, vtype=gp.GRB.BINARY, name="y")

        # u_i \in R

        u = model.addVars(self.num_nodes, vtype=gp.GRB.CONTINUOUS, name="u")

        # Objective function
        model.setObjective(gp.quicksum(self.prizes[i] * y[i] for i in range(self.num_nodes)), gp.GRB.MAXIMIZE)

        # Constraints
        # Constraint 1: Budget constraint
        model.addConstr(gp.quicksum(self.costs[i, j] * x[i, j] for i in range(self.num_nodes) for j in range(self.num_nodes)) <= b, name="time_budget")

        # Constraint 2: Inbound edge to start vertex should be one
        model.addConstr(gp.quicksum((x[i,s] for i in range(self.num_nodes) if i!=s)) == 1, name="incoming_edges_start")

        # Constraint 3: Outgoing edge from start vertex should be one
        model.addConstr((gp.quicksum(x[s,i] for i in range(self.num_nodes) if i!= s)) == 1, name="outgoing_edges_start")

        # Constraint 4: Inbound edge to end vertex should be y[i]
        model.addConstrs((gp.quicksum(x[i, j] for j in range(self.num_nodes) if j!= i) == y[i] for i in range(self.num_nodes) if i!= s), name="outgoing_edges")
        model.addConstrs((gp.quicksum(x[i, j] for i in range(self.num_nodes) if i!= j) == y[j] for j in range(self.num_nodes) if j!= s), name="incoming_edges")

        # Constraint 6: Subtour elimination
        M = self.num_nodes - 1
        for i in range(0, self.num_nodes):
            for j in range(0, self.num_nodes):
                if i != j and i!=s and j!=s:
                    model.addConstr(u[i] - u[j] + M * x[i, j] <= M - 1, f"mtz_{i}_{j}")

        model.addConstrs((2 <= u[i] for i in range(self.num_nodes)), name="subtour_elimination_2_lower")
        model.addConstrs((u[i] <= self.num_nodes for i in range(self.num_nodes)), name="subtour_elimination_2_upper")


        # Solve the model
        model.optimize()
        # print (model.display())
        # Print variables
        # print("Variables:")
        # for var in model.getVars():
        #     print(f"{var.VarName} = {var.x}")

        path = []
        # Check the optimization status
        if model.status == gp.GRB.OPTIMAL:
            print("Optimal solution found.")
            _, path = self.get_tour_edges(x, s)

        elif model.status == gp.GRB.INFEASIBLE:
            print("The model is infeasible.")
        else:
            print("Optimization did not converge to optimality.")

        return path

    def get_tour_edges(self, x, s):
        ''' Extracts the path from the model
        Args:
            x: model variable
            s: start vertex
        '''
        path_edges = []
        path_vertices = []
        # the path starts from vertex 's', so we start from s
        # -- find first edge:
        next_vertex = None
        for j in range(self.num_nodes):
            if x[s, j].X > 0.5:
                path_edges.append((s, j))
                path_vertices.append(s)
                next_vertex = j
                break
        # -- find the remaining edges
        while next_vertex != s:
            for j in range(self.num_nodes):
                if x[next_vertex, j].X > 0.5:
                    path_edges.append((next_vertex, j))
                    path_vertices.append(next_vertex)
                    next_vertex = j
                    break
        path_vertices.append(s)
        return path_edges, path_vertices

    def get_path_edges(self, x, s, t):
        ''' Extracts the path from the model
        Args:
            x: model variable
            s: start vertex
            t: goal vertex
        '''
        path_edges = []
        path_vertices = []
        # the path starts from vertex 's', so we start from s
        # -- find first edge:
        next_vertex = None
        for j in range(self.num_nodes):
            if x[s, j].X > 0.5:
                path_edges.append((s, j))
                path_vertices.append(s)
                next_vertex = j
                break
        # -- find the remaining edges
        while next_vertex != t:
            for j in range(self.num_nodes):
                if x[next_vertex, j].X > 0.5:
                    path_edges.append((next_vertex, j))
                    path_vertices.append(next_vertex)
                    next_vertex = j
                    break
        path_vertices.append(t)
        return path_edges, path_vertices

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

def main(n):
    graph_data = get_graph_data(7)
    print ("graph_data: ", graph_data)
    print ("-"*100)
    o_milp = OrienteeringMILP(graph_data)
    print ("Number of nodes: ", o_milp.num_nodes)

    r_set = set([graph_data['home'], graph_data['goal']])
    path = o_milp.solve_milp_path(graph_data['home'], graph_data['goal'], b=100)
    # path = o_milp.solve_milp(graph_data['home'], b=100)
    print ("Calculated path: ", path)
    print ("Path prize: ", o_milp.get_path_prize(path))
    print ("Path cost: ", o_milp.get_path_cost(path))

if __name__=='__main__':
    main(7)

