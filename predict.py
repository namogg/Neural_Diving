import tensorflow as tf
import ml_collections
import numpy as np
import tensorflow.compat.v2 as tf
import sys 
import math
import pyscipopt as scip
import matplotlib
import matplotlib.pyplot  as plt
sys.path.append("./")
from neural_lns import solvers 
from neural_lns import solving_utils
from neural_lns import mip_utils
from neural_lns import calibration
from neural_lns import sampling
from neural_lns import solution_data
from neural_lns import data_utils
from neural_lns import calibration , event
from graph_nets import graphs
from pyscipopt import Model, Eventhdlr, SCIP_RESULT, SCIP_EVENTTYPE, SCIP_PARAMSETTING, SCIP_STAGE
from predict_config import get_solver_config,configure_solver, get_mip
matplotlib.use("Agg")
"Create MIP instance"

"""
Get config
"""
import ml_collections
# def get_param():
#     solving_params = ml_collections.ConfigDict()
#     solving_params.solver_config = get_solver_config()
#     solving_params.sampler = sampling.RandomSampler(model_path="neural_lns/tmp/models")
#     def min_objective(a, b):
#         return min(a, b)
#     solving_params.objective = min_objective
#     solving_params.sol_data = solution_data.SolutionData(objective_type = min_objective)
#     solving_params.timer = calibration.Timer()
#     solving_params.solver = solvers.NeuralDivingSolver(solving_params.solver_config,solving_params.sampler)
#     solving_params.mip = get_mip()
#     return solving_params



def plot_array_as_line(data, xlabel="Node", ylabel="Gap", title="Gap Plot"):
    """
    Plots a 1D array as a line plot.

    Args:
    data (list or numpy.ndarray): The 1D array to plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    title (str): Title for the plot.
    """
    x = np.arange(len(data))  # Create x-coordinates based on array indices
    plt.plot(x, data, label="Data")
    plt.ylim(0, 6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend()
    plt.grid(True)
    plt.savefig("E:/plot/plot_SCIP.png")

def main():
    mip,scip_mip = get_mip("E:/benchmark/30n20b8.mps")
    #scip_mip.setPresolve(SCIP_PARAMSETTING.OFF)
    #scip_mip.setIntParam("limits/solutions", 1)
    #scip_mip.optimize()
    #scip_mip.setIntParam("parallel/maxnthreads", 12)
    node_event = event.NodeEvent()
    scip_mip.includeEventhdlr(node_event, "NodeEvent", "python event handler to catch Node Solved")
    solver_config = get_solver_config(mip,scip_mip)
    solving_params = configure_solver(mip)
    sampler = sampling.RepeatedCompetitionSampler("E:/neural_lns/tmp/models")

    # Create a new concrete function with the updated input signature
    solver =  solvers.NeuralDivingSolver(solver_config)
    solver._sampler = sampler
    sol_data, solve_stats = solvers.run_solver(mip,solving_params,solver)
    plot_array_as_line(node_event.gaps)
    print(sol_data)

if __name__ == "__main__":
    main()



