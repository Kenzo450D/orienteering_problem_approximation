import numpy as np

def get_graph_data(v:int=5, type:str="exploration"):
    ''' Returns the graph data for the given graph.
    Input:
        v <int>: number of vertices
        type <str>: type of graph (exploration, geometric)
    '''
    data = {}
    if v == 5:
        data['costs'] = np.array([[  0, 10,  6, 10, 12],
                                [ 10,  0, 12,  9,  5],
                                [  6, 12,  0,  6,  8],
                                [ 10,  9,  6,  0,  6],
                                [ 12,  5,  8,  6,  0]])
        data['prizes'] = np.array([50, 100, 150, 100, 50])
        data['depot_idx'] = 0
        #data['penalties'] = np.array([-50, -50, -50, -50, -50])
    elif v == 7 and type == "geometric":
        data['costs'] = np.array([[0, 2, 2, 2, 4, 4, 4],
                                  [2, 0, 4, 2, 2, 4, 4],
                                  [2, 4, 0, 2, 4, 2, 4],
                                  [2, 2, 2, 0, 2, 2, 2],
                                  [4, 2, 4, 2, 0, 4, 2],
                                  [4, 4, 2, 2, 4, 0, 2],
                                  [4, 4, 4, 2, 2, 2, 0]])
        data['prizes'] = np.array([0, 50, 50, 50, 50, 50, 0])
        data['depot_idx'] = 0
        data['home'] = 0
        data['goal'] = 6
    elif v == 7 and type == "exploration":
        data['costs'] = np.array([[  0,  5, 14, 18, 20, 21, 21],
                                  [  5,  0,  9, 13, 15, 16, 16],
                                  [ 14,  9,  0, 22, 24, 25, 25],
                                  [ 20, 13, 22,  0,  2,  3, 3],
                                  [ 21, 15, 24,  2,  0,  5, 5],
                                  [ 21, 16, 25,  3,  5,  0, 6],
                                  [ 21, 16, 45,  3,  5,  6, 0]])
        data['prizes'] = np.array([0, 0, 10000, 10000, 10000, 10000, 10000])
        data['depot_idx'] = 0
        data['home'] = 1
        data['goal'] = 0
    elif v == 10:
        #                          0  1  2  3  4  5  6  7  8  9
        data['costs'] = np.array([[0, 2, 2, 2, 4, 4, 4, 6, 6, 6],  # 0
                                  [2, 0, 4, 2, 2, 4, 4, 4, 6, 6],  # 1
                                  [2, 4, 0, 2, 4, 2, 4, 6, 4, 6],  # 2
                                  [2, 2, 2, 0, 2, 2, 2, 4, 4, 4],  # 3
                                  [4, 2, 4, 2, 0, 4, 2, 2, 4, 4],  # 4
                                  [4, 4, 2, 2, 4, 0, 2, 4, 2, 4],  # 5
                                  [4, 4, 4, 2, 2, 2, 0, 2, 2, 2],  # 6
                                  [6, 4, 6, 4, 2, 4, 2, 0, 4, 2],  # 7
                                  [6, 6, 4, 4, 4, 2, 2, 4, 0, 2],  # 8
                                  [6, 6, 6, 4, 4, 4, 2, 2, 2, 0]]) # 9
        data['prizes'] = np.array([0, 50, 50, 50, 50, 50, 50, 50, 50, 0])
        data['depot_idx'] = 0
        data['home'] = 0
        data['goal'] = 9
    else:
        # -- get graph
        data = None
    return data


