import numpy as np

def get_graph_data(v:int=5):
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
    elif v == 7:
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
    elif v == 10:
        #                          0  1  2  3  4  5  6  7  8  9
        data['costs'] = np.array([[0, 2, 2, 2, 4, 4, 4, 6, 6, 6],
                                  [2, 0, 4, 2, 2, 4, 4, 4, 6, 6],
                                  [2, 4, 0, 2, 4, 2, 4, 6, 6, 6],
                                  [2, 2, 2, 0, 2, 2, 2, 4, 4, 4],
                                  [4, 2, 4, 2, 0, 4, 2, 2, 4, 4],
                                  [4, 4, 2, 2, 4, 0, 2, 4, 2, 4],
                                  [4, 4, 4, 2, 2, 2, 0, 2, 2, 2],
                                  [6, 4, 6, 4, 2, 4, 2, 0, 4, 2],
                                  [6, 6, 4, 4, 4, 2, 2, 4, 0, 2],
                                  [6, 6, 6, 4, 4, 4, 2, 2, 2, 0]])
        data['prizes'] = np.array([0, 50, 50, 50, 50, 50, 50, 50, 50, 0])
        data['depot_idx'] = 0
        data['home'] = 0
        data['goal'] = 9
    else:
        # -- get graph
        data = None
    return data


