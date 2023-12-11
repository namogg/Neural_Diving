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
from predict_config import get_diving_config,configure_solver, get_mip_from_file, get_lns_config
from script import tsp
matplotlib.use("Agg")
"Create MIP instance"

"""
Get config
"""
import ml_collections

scip_mip = get_mip_from_file("E:/TSP/ga_for_tsp-main/data/a280.tsp")


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

def load_tsp(path): 
    return tsp.get_tsp(path)

def main():
    #scip_mip.setPresolve(SCIP_PARAMSETTING.OFF)
    #scip_mip.setIntParam("limits/solutions", 1)
    #scip_mip.setIntParam("parallel/maxnthreads", 12)
    node_event = event.NodeEvent()
    scip_mip.includeEventhdlr(node_event, "NodeEvent", "python event handler to catch Node Solved")
    scip_mip.setParam("limits/time", 8200)
    #scip_mip.optimize()
    #plot_array_as_line(node_event.gaps)
    #print(sol_data)
    diving_config = get_diving_config(scip_mip)
    lns_config = get_lns_config(diving_config)
    solving_params = configure_solver(scip_mip)
    sampler = sampling.RepeatedCompetitionSampler("E:/neural_lns/tmp/models")

    # Create a new concrete function with the updated input signature
    
    #solver =  solvers.NeuralDivingSolver(diving_config)
    solver = solvers.NeuralNSSolver(lns_config)
    solver._solver_config.diving_model = "E:/neural_lns/tmp/models"
    solver._sampler = sampler
    
    sol_data, solve_stats = solvers.run_solver(scip_mip,solving_params,solver)
    plot_array_as_line(node_event.gaps)
    print(sol_data)
    return sol_data

if __name__ == "__main__":
    main()