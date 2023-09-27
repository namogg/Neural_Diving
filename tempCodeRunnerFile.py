# def get_param():
#     solving_params = ml_collections.ConfigDict()
#     solving_params.solver_config = get_solver_config()
#     solving_params.sampler = sampling.RandomSampler(model_path="neural_lns/tmp/models")
#     def min_objective(a, b):
#         return min(a, b)
#     solving_params.objective = min_objective
#     solving_params.sol_data = solution_data.SolutionData(objective_type = min_objective)
#     solving_params.timer = calibration.Timer()
#     solving_params.solver = solvers.Ne