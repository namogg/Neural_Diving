import tensorflow as tf
import ml_collections
import numpy as np
import tensorflow.compat.v2 as tf
import sys 
import math
sys.path.append("./")
from neural_lns import solvers
from neural_lns import solving_utils
from neural_lns import mip_utils
from neural_lns import calibration
from neural_lns import sampling
from neural_lns import solution_data
from neural_lns import data_utils
from neural_lns import calibration
"Create MIP instance"
def get_mip():
    mip = mip_utils.MPModel()
    var1 = mip_utils.MPVariable(lower_bound=0, upper_bound=1, objective_coefficient=1, is_integer=True, name="var1")
    var2 = mip_utils.MPVariable(lower_bound=0, upper_bound=1, objective_coefficient=2, is_integer=True, name="var2")
    var3 = mip_utils.MPVariable(lower_bound=0, upper_bound=1, objective_coefficient=3, is_integer=True, name="var3")
    mip.variable.extend([var1, var2, var3])

    constraint1 = mip_utils.MPConstraint(var_index=[0, 1], coefficient=[1, 2], lower_bound=-math.inf, upper_bound=5, name="constraint1")
    constraint2 = mip_utils.MPConstraint(var_index=[1, 2], coefficient=[2, 3], lower_bound=-math.inf, upper_bound=6, name="constraint2")
    mip.constraint.extend([constraint1, constraint2])

    solution = mip_utils.MPSolutionResponse(objective_value=0.0, variable_value=[0.0, 0.0, 0.0], status_str="", status=mip_utils.MPSolverResponseStatus.NOT_SOLVED)
    return mip
"""
Get config
"""
import ml_collections
def get_param():
    solving_params = ml_collections.ConfigDict()
    solving_params.solver_config = get_solver_config()
    solving_params.sampler = sampling.RandomSampler(model_path="neural_lns/tmp/models")
    def min_objective(a, b):
        return min(a, b)
    solving_params.objective = min_objective
    solving_params.sol_data = solution_data.SolutionData(objective_type = min_objective)
    solving_params.timer = calibration.Timer()
    solving_params.solver = solvers.NeuralDivingSolver(solving_params.solver_config,solving_params.sampler)
    solving_params.mip = get_mip()
    return solving_params

def get_solver_config():
    solver_config = ml_collections.ConfigDict()
    solver_config.solver_name = 'neural_ns'
    solver_config.predict_config = ml_collections.ConfigDict()
    solver_config.predict_config.sampler_config = ml_collections.ConfigDict()
    solver_config.scip_solver_config = ml_collections.ConfigDict()
    solver_config.scip_solver_config.params = ml_collections.ConfigDict()
    solver_config.predict_config.sampler_config.name = 'random'
    solver_config.predict_config.extract_features_scip_config = ml_collections.ConfigDict({
        'seed': 42,
        'time_limit_seconds': 60 * 10,
        'separating_maxroundsroot': 0,   # No cuts
        'conflict_enable': False,        # No additional cuts
        'heuristics_emphasis': 'off',    # No heuristics
        'mip' : get_mip()
    })
    solver_config.preprocessor_configs = None
    solver_config.enable_restart = False
    solver_config.num_solve_steps = 5
    solver_config.perc_unassigned_vars = 0.5
    solver_config.temperature = 0.9
    solver_config.write_intermediate_sols = False
    return solver_config


solving_config = get_solver_config()
solving_params = get_param()
solver = solving_utils.Solver()
solver.solve(solving_params)