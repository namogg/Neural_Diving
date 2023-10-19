import sys
sys.path.append("./")
from neural_lns import data_utils, local_branching_data_generation, predict_config, mip_utils
import tensorflow as tf

from typing import List


mip_list = []
mip,scip_mip = predict_config.get_mip("E:/benchmark/30n20b8.mps")
mip_list.append(mip)
config = predict_config.get_solver_config(mip,scip_mip,True)

def createtf_record(filename,mip_list: List):
    with tf.io.TFRecordWriter(filename) as writer:
        for mip in mip_list:
            features = data_utils.get_features(mip,config.predict_config.extract_features_scip_config)
            features['variable_features'] = tf.cast(features['variable_features'],dtype=tf.float32)
            features['constraint_features'] = tf.cast(features['constraint_features'],dtype=tf.float32)
            features['edge_features'] = tf.cast(features['edge_features'],dtype=tf.float32)
            #features['edge_indices'] = tf.convert_to_tensor(features['edge_indices'],dtype = tf.int64)
            #features['best_solution_labels'] = tf.cast(features['best_solution_labels'],dtype=tf.float32)
            example = data_utils.serialized_example(features)
            writer.write(example)

createtf_record("neural_lns/data/example1.tfrecord",mip_list)