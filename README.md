# An implementation of Neural Diving 

### Overview

This project implement Neural Diving, a model from DeepMind to solve large scale Mixed Integer Programming Problem. Neural Diving use a Graph Neural Network to generate partial variable assignment and create sub-problem, the remaining variable can be assign by using SCIP. The code is inspired by Large Neighborhood Search Algorithm from DeepMind. To install SCIP, please follow this [installation](https://github.com/scipopt/PySCIPOpt/blob/master/INSTALL.md) guide.

Overview of the component

*   __calibration.py__: Abstract timer for MIP solving.
*   __config_train.py__: Configuration file for training parameters.
*   __data_utils.py__: Utility functions for feature extraction.
*   __layer_norm.py__: Model layer normalisation and dropout utilities.
*   __light_gnn.py__: The GNN model used for training.
*   __local_branching_data_generation.py__: Library with functions required to
    generate imitation data.
*   __local_branching_expert.py__: Expert for Neural Large Neighbourhood Search
    based on local branching.
*   __mip_utils.py__: MIP utility functions.
*   __preprocessor.py__: Abstract APIs for MIP preprocessor.
*   __sampling.py__: Sampling strategies for Neural LNS.
*   __solution_data.py__: SolutionData classes used to log solution process.
*   __solvers.py__: Neural diving and neural neighbourhood selection
    implementations.
*   __solving_utils.py__: Common utilities for `solvers.py`.
*   __train.py__: Training script for neural neighbourhood selection model.
*   __data__: Directory with example tfrecord file to run training. The example dataset is derivative of open-sourced [NN Verification Dataset](https://github.com/deepmind/deepmind-research/tree/master/neural_mip_solving).

## Installation

To install the dependencies of this implementation, please run:

```
pip install -U pip
pip install -r requirements.txt
```


## Usage

1. Implement the interfaces provided in `calibration.py`, `preprocessor.py` and
   `solving_utils.py` for the timer, preprocessor / presolver and solver
   respectively.
2. Specify valid training and validation paths in `config_train.py` (i.e.
   <dataset_absolute_training_path> and <dataset_absolute_validation_path>).
3. Train the neural neighbourhood selection model using:

   ```
   cd <parent-directory-of-neural_lns>
   python3 -m neural_lns.train
   ```

## Reference 
Paper: [Solving Mixed Integer Programs Using Neural Networks](https://arxiv.org/abs/2012.13349)
Reference Code: [neural_lns](https://github.com/google-deepmind/neural_lns)

## Disclaimer

