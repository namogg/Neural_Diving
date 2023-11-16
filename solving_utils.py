# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Common utilities for Solver."""
import sys
sys.path.append("./")
import abc
import enum
from typing import Any, Dict, Optional, List
import numpy as np
import ml_collections
import tensorflow as tf
import matplotlib.pyplot  as plt
from neural_lns import mip_utils
from mip_utils import MPSolverResponseStatus
import scipy.sparse as sp
from pyscipopt import  Eventhdlr, SCIP_EVENTTYPE, SCIP_STATUS, SCIP_PARAMSETTING,SCIP_STAGE
from event import MyEvent, NodeEvent
import pyscipopt as scip
    
handler = MyEvent()
class Converter:
    def vtype_int(var):
        """Retrieve the variables type (BINARY, INTEGER, IMPLINT or CONTINUOUS)"""
        vartype = var.vtype()
        if vartype == "BINARY":
            return 0  # BINARY
        elif vartype == "INTEGER":
            return 1  # INTEGER
        elif vartype == "CONTINUOUS":
            return 2  # CONTINUOUS
        elif vartype == "IMPLINT":
            return 3  # IMPLINT
        
    def getBasisStatus_int(var):
        stat = var.getBasisStatus()
        if stat == "lower":
                return 0
        elif stat == "basic":
            return 1
        elif stat == "upper":
            return 2
        elif stat == "zero":
            return 3
        else:
            raise Exception('SCIP returned unknown base status!')
        
    def map_status(status):
        status_mapping = {
            SCIP_STATUS.OPTIMAL: MPSolverResponseStatus.OPTIMAL.value,
            SCIP_STATUS.INFEASIBLE: MPSolverResponseStatus.INFEASIBLE.value,
            SCIP_STATUS.UNBOUNDED: MPSolverResponseStatus.UNBOUNDED.value,
            SCIP_STATUS.NODELIMIT: MPSolverResponseStatus.STOPPED.value,
            SCIP_STATUS.TOTALNODELIMIT : MPSolverResponseStatus.STOPPED.value,
            SCIP_STATUS.STALLNODELIMIT : MPSolverResponseStatus.STOPPED.value,
            SCIP_STATUS.USERINTERRUPT  : MPSolverResponseStatus.STOPPED.value,
            SCIP_STATUS.MEMLIMIT : MPSolverResponseStatus.STOPPED.value,
            SCIP_STATUS.GAPLIMIT: MPSolverResponseStatus.STOPPED.value,
            SCIP_STATUS.BESTSOLLIMIT: MPSolverResponseStatus.FEASIBLE.value,
            SCIP_STATUS.RESTARTLIMIT: MPSolverResponseStatus.STOPPED.value,
            SCIP_STATUS.SOLLIMIT: MPSolverResponseStatus.STOPPED.value,
            SCIP_STATUS.UNKNOWN: MPSolverResponseStatus.UNKNOWN.value,
            SCIP_STATUS.MEMLIMIT: MPSolverResponseStatus.FAILED.value,
            SCIP_STATUS.BESTSOLLIMIT: MPSolverResponseStatus.BESTSOLLIMIT.value
        }

        # Ánh xạ từng giá trị từ PY_SCIP_STATUS vào MPSolverResponseStatus
        if status in status_mapping:
            return status_mapping[status]
        else:
            return MPSolverResponseStatus.UNKNOWN.value
    
class SolverState(enum.Enum):
  INIT = 0
  MODEL_LOADED = 1
  FINISHED = 2


class Solver(abc.ABC):
  """Wrapper around a given classical MIP solver.

  This class contains the API needed to communicate with a MIP solver, e.g.
  SCIP.
  """
  def __init__(self):
        self.solutions = []

  def load_model(self,scip_mip) -> SolverState:
        """Loads a MIP model into the solver."""
        # Implement the logic to load the MIP model here
        # You can access the MIP model using the 'mip' parameter
        # Return the state of the solver after loading the model
        # For example:
        if type(scip_mip) is scip.Model:
            self.scip_mip = scip_mip
        else: 
            self.scip_mip = scip_mip[0]
        status = mip_utils.MPSolverResponseStatus.NOT_SOLVED
        return status

  
  def solve(
      self, solving_params: ml_collections.ConfigDict
    ) -> mip_utils.MPSolverResponseStatus:
    """Solves the loaded MIP model."""
    #solving_params.scip_mip
    scip_mip = self.scip_mip
    #scip_mip.dropEvent(SCIP_EVENTTYPE.FIRSTLPSOLVED,handler)
    #scip_mip.setIntParam("parallel/maxnthreads", 12)
    # scip_mip.setRealParam("limits/gap", 1e-9) 
    # scip_mip.setBoolParam("timing/clocktype", 1)
    # start_time = time.time()
    scip_mip.setIntParam("limits/solutions", 1)
    #node_event = NodeEvent()
    #scip_mip.includeEventhdlr(node_event, "NodeEvent", "python event handler to catch Node Solved")
    scip_mip.solveConcurrent()
    status = Converter.map_status(scip_mip.getStatus())
    solutions = scip_mip.getSols()
    for sol in solutions: 
        obj_val = scip_mip.getObjVal(sol)
        sol_mip = mip_utils.convert_pyscipopt_solution_to_MPResponse(scip_mip,sol,status)
        self.add_solution(sol_mip)
    return status
  
  def get_best_solution(self) -> Optional[Any]:
    if self.solutions:
        # Tìm lời giải tốt nhất dựa trên giá trị hàm mục tiêu
        best_solution = min(self.solutions, key=lambda s: s.objective_value)
        return best_solution
    else:
        raise ValueError("No solutions found")
    
  def add_solution(self, solution: Any) -> bool:
    # Create a dictionary to map variable names to their values
    solution
    self.solutions.append((solution))

    return solution

  def extract_lp_features_at_root(
      self, solving_params: ml_collections.ConfigDict) -> Dict[str, Any]:
    """Returns a dictionary of root node features."""
    mip = solving_params.mip
    scip_mip = solving_params.scip_mip
    if(solving_params.train):
         scip_mip.setPresolve(SCIP_PARAMSETTING.OFF)
    #scip_mip.setPresolve(SCIP_PARAMSETTING.OFF)
    #scip_mip.hideOutput()
    #scip_mip.includeEventhdlr(handler, "FIRSTLPSOLVED", "python event handler to catch FIRSTLPEVENT")
    #scip_mip.optimize()
    constraint_features, edge_features, variable_features, edge_indices = FeatureExtractor.extract_state(scip_mip)
    features = {}
    
    features['variable_features'] = tf.convert_to_tensor(variable_features['values'], dtype=tf.float64)
    features['binary_variable_indices'] = np.array(FeatureExtractor.extract_binary_indices(scip_mip))
    features['model_maximize'] = mip.maximize
    features['variable_names'] = tf.convert_to_tensor(FeatureExtractor.extract_name_feature(scip_mip))
    features['constraint_features'] = tf.convert_to_tensor(constraint_features['values'], dtype=tf.float64)
    features['best_solution_labels'] = tf.convert_to_tensor(0, dtype=tf.float64)
    features['variable_lbs'] = FeatureExtractor.extract_lower_bounds(scip_mip)
    features['edge_indices'] = FeatureExtractor.extract_edge_indices(edge_indices)
    features['all_integer_variable_indices'] = tf.convert_to_tensor(FeatureExtractor.extract_integer_indices(scip_mip), dtype=tf.int64)
    features['edge_features_names'] = tf.convert_to_tensor("coef_normalized")
    features['variable_feature_names'] = tf.convert_to_tensor("Variable features: (age, avg_inc_val, basis_status_0, basis_status_1, basis_status_2, basis_status_3, coef_normalized, has_lb, has_ub, inc_val, reduced_cost, sol_frac, sol_is_at_lb, sol_is_at_ub, sol_val, type_0, type_1, type_2, type_3')")
    features['constraint_feature_names'] = tf.convert_to_tensor("age, bias, dualsol_val_normalized, is_tight, obj_cosine_similarity")
    features['variable_ubs'] = tf.convert_to_tensor(FeatureExtractor.extract_upper_bounds(scip_mip))
    features['edge_features'] = tf.convert_to_tensor(edge_features['values'], dtype=tf.float64)
    #handler.eventexit()
    return features


class FeatureExtractor():
    """
    Variable features: (age,avg_inc_val,basis_status_0,basis_status_1,basis_status_2,basis_status_3,coef_normalized,has_lb,has_ub,inc_val,reduced_cost,sol_frac,sol_is_at_lb,sol_is_at_ub,sol_val,type_0,type_1,type_2,type_3'],
        dtype=object)
    Constrain features: numpy= (age,bias,dualsol_val_normalized,is_tight,obj_cosine_similarity')
    Edge features: (coef_normalized')
    """
    def extract_integer_indices(scip_model):
        integer_indices = []
        vars = scip_model.getVars(scip_model.getLPColsData())
        for i,var in enumerate(vars):
            if var.vtype() == 'INTEGER':
                integer_indices.append(i)
        return integer_indices

    def extract_name_feature(scip_model):
        name_features = []
        vars = scip_model.getVars(scip_model.getLPColsData())
        for var in vars:
            name_features.append(var.name)
        return name_features

    def extract_binary_indices(scip_model):
        binary_indices = []
        vars = scip_model.getVars(scip_model.getLPColsData())
        for i,var in enumerate(vars):
            if var.vtype() == 'BINARY':
                binary_indices.append(i)
        return binary_indices

    def extract_lower_bounds(scip_model):
        lower_bounds = []
        vars = scip_model.getVars(scip_model.getLPColsData())
        for var in vars:
            lower_bounds.append(var.getLbLocal())
        return lower_bounds

    def extract_upper_bounds(scip_model):
        upper_bounds = []
        vars = scip_model.getVars(scip_model.getLPColsData())
        for var in vars:
            upper_bounds.append(var.getUbLocal())
        return upper_bounds
    
    def extract_edge_indices(edge_indices):
        # edge_indices = []
        # index_map = {}
        # for index, var in enumerate(model.getVars()):
        #     index_map[var.name] = index
        # # Iterate over constraints in the model
        # for constraint_index, constraint in enumerate(model.getConss()):
        #     for key in model.getValsLinear(constraint).keys():
        #         # Get the variables related to the constraint
        #         variables_index = index_map[key]
        #         edge_indices.append([constraint_index, variables_index])
        if len(edge_indices) > 0:
            edge_indices_np = np.array(edge_indices, dtype=np.int64)
        else:
            # Create an empty NumPy array with the desired shape
            edge_indices_np = np.empty((0, 2), dtype=np.int64)
        return np.transpose(edge_indices_np)
    
    def extract_state(model, buffer=None):
        """
        Compute a bipartite graph representation of the solver. In this
        representation, the variables and constraints of the MILP are the
        left- and right-hand side nodes, and an edge links two nodes iff the
        variable is involved in the constraint. Both the nodes and edges carry
        features.

        Parameters
        ----------
        model : pyscipopt.scip.Model
            The current model.
        buffer : dict
            A buffer to avoid re-extracting redundant information from the solver
            each time.
        Returns
        -------
        variable_features : dictionary of type {'names': list, 'values': np.ndarray}
            The features associated with the variable nodes in the bipartite graph.
        edge_features : dictionary of type ('names': list, 'indices': np.ndarray, 'values': np.ndarray}
            The features associated with the edges in the bipartite graph.
            This is given as a sparse matrix in COO format.
        constraint_features : dictionary of type {'names': list, 'values': np.ndarray}
            The features associated with the constraint nodes in the bipartite graph.
        """
        if buffer is None or model.getNNodes() == 1:
            buffer = {}

        # update state from buffer if any
        s = FeatureExtractor.get_state(model)
        #s = model.getState(buffer['scip_state'] if 'scip_state' in buffer else None)
        buffer['scip_state'] = s

        if 'state' in buffer:
            obj_norm = buffer['state']['obj_norm']
        else:
            obj_norm = np.linalg.norm(s['col']['coefs'])
            obj_norm = 1 if obj_norm <= 0 else obj_norm

        row_norms = s['row']['norms']
        row_norms[row_norms == 0] = 1

        # Column features
        n_cols = len(s['col']['types'])

        if 'state' in buffer:
            col_feats = buffer['state']['col_feats']
        else:
            col_feats = {}
            col_feats['type'] = np.zeros((n_cols, 4))  # BINARY INTEGER IMPLINT CONTINUOUS
            col_feats['type'][np.arange(n_cols), s['col']['types']] = 1
            col_feats['coef_normalized'] = s['col']['coefs'].reshape(-1, 1) / obj_norm
        col_feats['empty_1'] = np.empty((n_cols, 1))
        col_feats['empty_2'] = np.empty((n_cols, 1))
        col_feats['empty_3'] = np.empty((n_cols, 1))
        col_feats['empty_4'] = np.empty((n_cols, 1))
        col_feats['empty_5'] = np.empty((n_cols, 1))
        col_feats['empty_6'] = np.empty((n_cols, 1))
        col_feats['empty_7'] = np.empty((n_cols, 1))
        col_feats['has_lb'] = ~np.isnan(s['col']['lbs']).reshape(-1, 1)
        col_feats['has_ub'] = ~np.isnan(s['col']['ubs']).reshape(-1, 1)
        col_feats['sol_is_at_lb'] = s['col']['sol_is_at_lb'].reshape(-1, 1)
        col_feats['sol_is_at_ub'] = s['col']['sol_is_at_ub'].reshape(-1, 1)
        col_feats['sol_frac'] = s['col']['solfracs'].reshape(-1, 1)
        col_feats['sol_frac'][s['col']['types'] == 3] = 0  # continuous have no fractionality
        col_feats['basis_status'] = np.zeros((n_cols, 4))  # LOWER BASIC UPPER ZERO
        col_feats['basis_status'][np.arange(n_cols), s['col']['basestats']] = 1
        col_feats['reduced_cost'] = s['col']['redcosts'].reshape(-1, 1) / obj_norm
        col_feats['age'] = s['col']['ages'].reshape(-1, 1) / (s['stats']['nlps'] + 5)
        col_feats['sol_val'] = s['col']['solvals'].reshape(-1, 1)
        col_feats['inc_val'] = s['col']['incvals'].reshape(-1, 1)
        col_feats['avg_inc_val'] = s['col']['avgincvals'].reshape(-1, 1)

        col_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in col_feats.items()]
        col_feat_names = [n for names in col_feat_names for n in names]
        col_feat_vals = np.concatenate(list(col_feats.values()), axis=-1)

        variable_features = {
            'names': col_feat_names,
            'values': col_feat_vals,}

        # Row features

        if 'state' in buffer:
            row_feats = buffer['state']['row_feats']
            has_lhs = buffer['state']['has_lhs']
            has_rhs = buffer['state']['has_rhs']
        else:
            row_feats = {}
            has_lhs = np.nonzero(~np.isnan(s['row']['lhss']))[0]
            has_rhs = np.nonzero(~np.isnan(s['row']['rhss']))[0]
            row_feats['obj_cosine_similarity'] = np.concatenate((
                -s['row']['objcossims'][has_lhs],
                +s['row']['objcossims'][has_rhs])).reshape(-1, 1)
            row_feats['bias'] = np.concatenate((
                -(s['row']['lhss'] / row_norms)[has_lhs],
                +(s['row']['rhss'] / row_norms)[has_rhs])).reshape(-1, 1)

        row_feats['is_tight'] = np.concatenate((
            s['row']['is_at_lhs'][has_lhs],
            s['row']['is_at_rhs'][has_rhs])).reshape(-1, 1)

        row_feats['age'] = np.concatenate((
            s['row']['ages'][has_lhs],
            s['row']['ages'][has_rhs])).reshape(-1, 1) / (s['stats']['nlps'] + 5)

        # # redundant with is_tight
        # tmp = s['row']['basestats']  # LOWER BASIC UPPER ZERO
        # tmp[s['row']['lhss'] == s['row']['rhss']] = 4  # LOWER == UPPER for equality constraints
        # tmp_l = tmp[has_lhs]
        # tmp_l[tmp_l == 2] = 1  # LHS UPPER -> BASIC
        # tmp_l[tmp_l == 4] = 2  # EQU UPPER -> UPPER
        # tmp_l[tmp_l == 0] = 2  # LHS LOWER -> UPPER
        # tmp_r = tmp[has_rhs]
        # tmp_r[tmp_r == 0] = 1  # RHS LOWER -> BASIC
        # tmp_r[tmp_r == 4] = 2  # EQU LOWER -> UPPER
        # tmp = np.concatenate((tmp_l, tmp_r)) - 1  # BASIC UPPER ZERO
        # row_feats['basis_status'] = np.zeros((len(has_lhs) + len(has_rhs), 3))
        # row_feats['basis_status'][np.arange(len(has_lhs) + len(has_rhs)), tmp] = 1

        tmp = s['row']['dualsols'] / (row_norms * obj_norm)
        row_feats['dualsol_val_normalized'] = np.concatenate((
                -tmp[has_lhs],
                +tmp[has_rhs])).reshape(-1, 1)

        row_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in row_feats.items()]
        row_feat_names = [n for names in row_feat_names for n in names]
        row_feat_vals = np.concatenate(list(row_feats.values()), axis=-1)

        constraint_features = {
            'names': row_feat_names,
            'values': row_feat_vals,}

        # Edge features
        if 'state' in buffer:
            edge_row_idxs = buffer['state']['edge_row_idxs']
            edge_col_idxs = buffer['state']['edge_col_idxs']
            edge_feats = buffer['state']['edge_feats']
        else:
            coef_matrix = sp.csr_matrix(
                (s['nzrcoef']['vals'] / row_norms[s['nzrcoef']['rowidxs']],
                (s['nzrcoef']['rowidxs'], s['nzrcoef']['colidxs'])),
                shape=(len(s['row']['nnzrs']), len(s['col']['types'])))
            coef_matrix = sp.vstack((
                -coef_matrix[has_lhs, :],
                coef_matrix[has_rhs, :])).tocoo(copy=False)

            edge_row_idxs, edge_col_idxs = coef_matrix.row, coef_matrix.col
            edge_feats = {}

            edge_feats['coef_normalized'] = coef_matrix.data.reshape(-1, 1)

        edge_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in edge_feats.items()]
        edge_feat_names = [n for names in edge_feat_names for n in names]
        edge_feat_indices = np.vstack([edge_row_idxs, edge_col_idxs])
        edge_feat_vals = np.concatenate(list(edge_feats.values()), axis=-1)

        edge_features = {
            'names': edge_feat_names,
            'indices': edge_feat_indices,
            'values': edge_feat_vals,}

        if 'state' not in buffer:
            buffer['state'] = {
                'obj_norm': obj_norm,
                'col_feats': col_feats,
                'row_feats': row_feats,
                'has_lhs': has_lhs,
                'has_rhs': has_rhs,
                'edge_row_idxs': edge_row_idxs,
                'edge_col_idxs': edge_col_idxs,
                'edge_feats': edge_feats,
            }
        return constraint_features, edge_features, variable_features, edge_feat_indices

    def get_state(model, prev_state=None):
        cols = model.getLPColsData()
        rows = model.getLPRowsData()
        ncols = model.getNLPCols()
        nrows = model.getNLPRows()
        update = prev_state is not None

        col_types = np.empty(shape=(ncols,), dtype=np.int32)
        col_coefs = np.empty(shape=(ncols,), dtype=np.float32)
        col_lbs = np.empty(shape=(ncols,), dtype=np.float32)
        col_ubs = np.empty(shape=(ncols,), dtype=np.float32)
        col_basestats = np.empty(shape=(ncols,), dtype=np.int32)
        col_redcosts = np.empty(shape=(ncols,), dtype=np.float32)
        col_ages = np.empty(shape=(ncols,), dtype=np.int32)
        col_solvals = np.empty(shape=(ncols,), dtype=np.float32)
        col_solfracs = np.empty(shape=(ncols,), dtype=np.float32)
        col_sol_is_at_lb = np.empty(shape=(ncols,), dtype=np.int32)
        col_sol_is_at_ub = np.empty(shape=(ncols,), dtype=np.int32)
        col_incvals = np.empty(shape=(ncols,), dtype=np.float32)
        col_avgincvals = np.empty(shape=(ncols,), dtype=np.float32)

        sol = model.getBestSol()
        for i in range(ncols):
            col_i = cols[i].getLPPos()
            var = cols[i].getVar()

            lb = cols[i].getLb()
            ub = cols[i].getUb()
            solval = cols[i].getPrimsol()

            if not update:

                # Variable type
                col_types[col_i] = Converter.vtype_int(var)

                # Objective coefficient
                col_coefs[col_i] = cols[i].getObjCoeff()

            # Lower bound
            if model.isInfinity(abs(lb)):
                col_lbs[col_i] = np.nan
            else:
                col_lbs[col_i] = lb

            # Upper bound
            if model.isInfinity(abs(ub)):
                col_ubs[col_i] = np.nan
            else:
                col_ubs[col_i] = ub

            # Basis status
            col_basestats[col_i] = Converter.getBasisStatus_int(cols[i])

            # Reduced cost
            col_redcosts[col_i] = Converter.getBasisStatus_int(cols[i])

            # Age
            col_ages[col_i] = 0 #cols[i].age

            # LP solution value
            col_solvals[col_i] = solval
            col_solfracs[col_i] = model.feasFrac(solval)
            col_sol_is_at_lb[col_i] = model.isEQ(solval, lb)
            col_sol_is_at_ub[col_i] = model.isEQ(solval, ub)

            # Incumbent solution value
            if sol is None:
                col_incvals[col_i] = np.nan
                col_avgincvals[col_i] = np.nan
            else:
                col_incvals[col_i] = model.getSolVal(sol, var)
                col_avgincvals[col_i] = 0
        row_lhss = np.empty(shape=(nrows,), dtype=np.float32)
        row_rhss = np.empty(shape=(nrows,), dtype=np.float32)
        row_nnzrs = np.empty(shape=(nrows,), dtype=np.int32)
        row_dualsols = np.empty(shape=(nrows,), dtype=np.float32)
        row_basestats = np.empty(shape=(nrows,), dtype=np.int32)
        row_ages = np.empty(shape=(nrows,), dtype=np.int32)
        row_activities = np.empty(shape=(nrows,), dtype=np.float32)
        row_objcossims = np.empty(shape=(nrows,), dtype=np.float32)
        row_norms = np.empty(shape=(nrows,), dtype=np.float32)
        row_is_at_lhs = np.empty(shape=(nrows,), dtype=np.int32)
        row_is_at_rhs = np.empty(shape=(nrows,), dtype=np.int32)

        if not update:
            row_lhss = np.empty(shape=(nrows,), dtype=np.float32)
            row_rhss = np.empty(shape=(nrows,), dtype=np.float32)
            row_nnzrs = np.empty(shape=(nrows,), dtype=np.int32)
            row_dualsols = np.empty(shape=(nrows,), dtype=np.float32)
            row_basestats = np.empty(shape=(nrows,), dtype=np.int32)
            row_ages = np.empty(shape=(nrows,), dtype=np.int32)
            row_activities = np.empty(shape=(nrows,), dtype=np.float32)
            row_objcossims = np.empty(shape=(nrows,), dtype=np.float32)
            row_norms = np.empty(shape=(nrows,), dtype=np.float32)
            row_is_at_lhs = np.empty(shape=(nrows,), dtype=np.int32)
            row_is_at_rhs = np.empty(shape=(nrows,), dtype=np.int32)
            row_is_local = np.empty(shape=(nrows,), dtype=np.int32)
            row_is_modifiable = np.empty(shape=(nrows,), dtype=np.int32)
            row_is_removable = np.empty(shape=(nrows,), dtype=np.int32)
        else:
            row_lhss = prev_state['row']['lhss']
            row_rhss = prev_state['row']['rhss']
            row_nnzrs = prev_state['row']['nnzrs']
            row_dualsols = prev_state['row']['dualsols']
            row_basestats = prev_state['row']['basestats']
            row_ages = prev_state['row']['ages']
            row_activities = prev_state['row']['activities']
            row_objcossims = prev_state['row']['objcossims']
            row_norms = prev_state['row']['norms']
            row_is_at_lhs = prev_state['row']['is_at_lhs']
            row_is_at_rhs = prev_state['row']['is_at_rhs']
            row_is_local = prev_state['row']['is_local']
            row_is_modifiable = prev_state['row']['is_modifiable']
            row_is_removable = prev_state['row']['is_removable']

        nnzrs = 0
        for i in range(nrows):
            lhs = rows[i].getLhs()
            rhs = rows[i].getRhs()
            cst = rows[i].getConstant()
            activity = model.getRowLPActivity(rows[i])

            if not update:
                row_nnzrs[i] = rows[i].getNLPNonz()
                nnzrs += row_nnzrs[i]

                if model.isInfinity(abs(lhs)):
                    row_lhss[i] = np.nan
                else:
                    row_lhss[i] = lhs - cst

                if model.isInfinity(abs(rhs)):
                    row_rhss[i] = np.nan
                else:
                    row_rhss[i] = rhs - cst

                row_is_local[i] = rows[i].isLocal()
                row_is_modifiable[i] = rows[i].isModifiable()
                row_is_removable[i] = rows[i].isRemovable()

                # SCIPlpRecalculateObjSqrNorm(scip.set, scip.lp)
                # prod = rows[i].sqrnorm * scip.lp.objsqrnorm
                # row_objcossims[i] = rows[i].objprod / SQRT(prod) if SCIPisPositive(scip, prod) else 0.0

                row_norms[i] = rows[i].getNorm()

            row_dualsols[i] = model.getRowDualSol(rows[i])
            row_basestats[i] = Converter.getBasisStatus_int(rows[i])
            row_ages[i] = 0
            row_activities[i] = activity - cst
            row_is_at_lhs[i] = model.isEQ(activity, lhs)
            row_is_at_rhs[i] = model.isEQ(activity, rhs)


        coef_colidxs = np.empty(shape=(nnzrs,), dtype=np.int32)
        coef_rowidxs = np.empty(shape=(nnzrs,), dtype=np.int32)
        coef_vals = np.empty(shape=(nnzrs,), dtype=np.float32)

        if not update:
            j = 0
            for i in range(nrows):
                row_cols = rows[i].getCols()
                row_vals = rows[i].getVals()
                for k in range(row_nnzrs[i]):
                    coef_colidxs[j + k] = row_cols[k].getLPPos()
                    coef_rowidxs[j + k] = i
                    coef_vals[j + k] = row_vals[k]
                j += row_nnzrs[i]
        state = {
                'col': {
                    'types':        col_types,
                    'coefs':        col_coefs,
                    'lbs':          col_lbs,
                    'ubs':          col_ubs,
                    'basestats':    col_basestats,
                    'redcosts':     col_redcosts,
                    'ages':         col_ages,
                    'solvals':      col_solvals,
                    'solfracs':     col_solfracs,
                    'sol_is_at_lb': col_sol_is_at_lb,
                    'sol_is_at_ub': col_sol_is_at_ub,
                    'incvals':      col_incvals,
                    'avgincvals':   col_avgincvals,
                },
                'row': {
                    'lhss':          row_lhss,
                    'rhss':          row_rhss,
                    'nnzrs':         row_nnzrs,
                    'dualsols':      row_dualsols,
                    'basestats':     row_basestats,
                    'ages':          row_ages,
                    'activities':    row_activities,
                    'objcossims':    row_objcossims,
                    'norms':         row_norms,
                    'is_at_lhs':     row_is_at_lhs,
                    'is_at_rhs':     row_is_at_rhs,
                    'is_local':      row_is_local,
                    'is_modifiable': row_is_modifiable,
                    'is_removable':  row_is_removable,
                },
                'nzrcoef': {
                    'colidxs': coef_colidxs,
                    'rowidxs': coef_rowidxs,
                    'vals':    coef_vals,
                },
                'stats': {
                    'nlps': model.getNLPs(),
                },
            }
        return state