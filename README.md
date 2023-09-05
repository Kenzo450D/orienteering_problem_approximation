# Recursive Greedy Algorithm for Orienteering Problem

The orienteering problem is vehicle routing problem where the goal is to maximize the prize collected in a path in a weighted graph. The vehicles has budget constraints, where the sum of the path edges should not exceed the budget of the vehicle.

This repository containts a python code that implements the recursive greedy algorithm for orienteering problem for a single vehicle or robot. 

The algorithm is based on Chekuri and Pal (2005) algorithm. 


## Usage

To execute the algorithm, use the following command:

```bash
python3 recursive_algorithm.py
```


# Recursive Greedy Algorithm for Profitable Tour Problem

The approximation algorithm for the profitable tour problem (PTP) is similar to that of the orienteering problem. Instead of just considering the tour prize, we consider the prize and the cost. We have multiplied the prize along the tour by a multiplier 10, and subtracted the path cost from it to get the net profit along each path. The goal is to find a prize which gives the maximum profit given a budget constraint.



## Usage

To execute the Profitable Tour Problem solver, use the following command:

```bash
python3 recursive_algorithm_ptp.py
```

# Citing the code

If you find this code useful for your work or research, please consider starring this GitHub repository and citing it in your work. Your support is greatly appreciated.

# Author

**Sayantan Datta** 
- Email: sayantan.knz@gmail.com