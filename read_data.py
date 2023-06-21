import sys 
sys.path.append("./")
from neural_lns import solvers
from neural_lns import mip_utils
from neural_lns import calibration
from neural_lns import sampling
from neural_lns import solution_data
from neural_lns import data_utils
from neural_lns import calibration
import tensorflow as tf
import ml_collections
import numpy as np
import tensorflow.compat.v2 as tf

import math

def read_tfrecord(file_path):
    dataset = tf.data.TFRecordDataset(file_path)

    # Apply the decode function to each record
    parsed_dataset = dataset.map(data_utils.decode_fn)

    return parsed_dataset
dataset = read_tfrecord("neural_lns/data/example.tfrecord")
print(dataset)
for example in dataset:
    variable_features = example['variable_features']
    constraint_features = example['constraint_features']
    edge_indices = example['edge_indices']
    edge_features = example['edge_features']
    all_integer_variable_indices = example['all_integer_variable_indices']
    best_solution_labels = example['best_solution_labels']
    binary_variable_indices = example['binary_variable_indices']
    constraint_feature_names = example['constraint_feature_names']
    edge_features_names = example['edge_features_names']
    model_maximize = example['model_maximize']
    variable_feature_names = example['variable_feature_names']
    variable_lbs = example['variable_lbs']
    variable_names = example['variable_names']
    variable_ubs = example['variable_ubs']

