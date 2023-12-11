import ml_collections
import mip_utils
import os
from typing import List
from script import tsp

def get_mip_from_file(path):
    if '.tsp' in os.path.basename(path): 
        scip_model,_ = tsp.get_tsp(path)
    else:
        scip_model = mip_utils.read_lp(path)
    #mip = mip_utils.convert_pyscipmodel_to_mip(scip_model)
    return scip_model


def get_lns_config(diving_config): 
    solver_config = ml_collections.ConfigDict()
    solver_config.diving_config = diving_config
    solver_config.perc_unassigned_vars = 0.7
    solver_config.predict_config = diving_config.predict_config
    solver_config.scip_solver_config = diving_config.scip_solver_config
    solver_config.temperature = 0.2
    return solver_config


def get_diving_config(scip_mip,train = False):
    """
    - Config cho solver: SCIP, NeuralDiving, Sampler
    """
    solver_config = ml_collections.ConfigDict()
    solver_config.solver_name = 'neural_ns'
    solver_config.predict_config = ml_collections.ConfigDict()
    solver_config.predict_config.sampler_config = ml_collections.ConfigDict()
    solver_config.scip_solver_config = ml_collections.ConfigDict()
    solver_config.scip_solver_config.params = ml_collections.ConfigDict()
    solver_config.scip_solver_config.params.scip_mip =  scip_mip
    #solver_config.scip_solver_config.params.mip =  mip
    solver_config.predict_config.sampler_config.name = 'competition'
    solver_config.predict_config.sampler_config.params = ml_collections.ConfigDict()
    solver_config.predict_config.extract_features_scip_config = ml_collections.ConfigDict({
        'seed': 42,
        'time_limit_seconds': 60 * 10,
        'separating_maxroundsroot': 0,   # No cuts
        'conflict_enable': False,        # No additional cuts
        'heuristics_emphasis': 'off',    # No heuristics
        #'mip' : solver_config.scip_solver_config.params.mip,
        'scip_mip' : solver_config.scip_solver_config.params.scip_mip,
        'train': train
    })
    solver_config.predict_config.percentage_predict_vars = 0.1
    solver_config.preprocessor_configs = None
    solver_config.enable_restart = False
    solver_config.num_solve_steps = 5
    solver_config.perc_unassigned_vars = 0
    solver_config.temperature = 0.9
    solver_config.write_intermediate_sols = False
    return solver_config

def configure_solver(scip_mip) -> ml_collections.ConfigDict:
    """
    Config các biến để giải: MIP, preprocessor
    """
    config = ml_collections.ConfigDict()
    #config.mip = mip
    config.scip_mip = scip_mip
    # Preprocessor configuration (optional)
    config.preprocessor_configs = None 
    config.write_intermediate_sols = True  # Set to True or False as desired
    return config


def get_mips_from_folder(folder_path: str, extensions: List[str]):
    mips = []
    # Duyệt qua tất cả các tệp trong thư mục
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Kiểm tra phần mở rộng của tệp
            _, file_extension = os.path.splitext(file)
            if file_extension in extensions:
                file_path = os.path.join(root, file)
                mip,_ = get_mip_from_file(file_path)
                mips.append(mip)
    return mips
