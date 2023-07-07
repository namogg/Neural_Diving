import tensorflow as tf
import ml_collections
import numpy as np
import tensorflow.compat.v2 as tf
import sys 
import math
import pyscipopt as scip
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

    # Variables
    var1 = mip_utils.MPVariable(lower_bound=0, upper_bound=1, objective_coefficient=7, is_integer=True, name="var1")
    var2 = mip_utils.MPVariable(lower_bound=-math.inf, upper_bound=math.inf, objective_coefficient=-9, is_integer=False, name="var2")
    mip.variable.extend([var1, var2])

    # Constraints
    constraint1 = mip_utils.MPConstraint(var_index=[0, 1], coefficient=[-1, 3], lower_bound=-math.inf, upper_bound=6, name="constraint1")
    constraint2 = mip_utils.MPConstraint(var_index=[0, 1], coefficient=[7, 1], lower_bound=-math.inf, upper_bound=35, name="constraint2")
    mip.constraint.extend([constraint1, constraint2])

    mip.maximize = True

    return mip
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

def get_solver_config():
    solver_config = ml_collections.ConfigDict()
    solver_config.solver_name = 'neural_ns'
    solver_config.predict_config = ml_collections.ConfigDict()
    solver_config.predict_config.sampler_config = ml_collections.ConfigDict()
    solver_config.scip_solver_config = ml_collections.ConfigDict()
    solver_config.scip_solver_config.params = ml_collections.ConfigDict()
    solver_config.predict_config.sampler_config.name = 'random'
    solver_config.predict_config.sampler_config = ml_collections.ConfigDict()
    solver_config.predict_config.sampler_config.params = ml_collections.ConfigDict()
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
def configure_solver() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    # Preprocessor configuration (optional)
    config.preprocessor_configs = None  # Add preprocessor configurations if needed

    # Additional solver-specific configurations
    # Add solver-specific configurations here

    # Write intermediate solutions
    config.write_intermediate_sols = True  # Set to True or False as desired
    return config

def main():
    mip = get_mip()
    solver_config = get_solver_config()
    solving_params = configure_solver()
    sampler = sampling.RepeatedCompetitionSampler("neural_lns/tmp/models")
    solver =  solvers.NeuralDivingSolver(solver_config)
    solver._sampler = sampler
    solvers.run_solver(mip,solving_params,solver)

if __name__ == "__main__":
    main()