import torch

# ---- Neural Network params ----
trial_length = 600
num_trials_nn = 1

device_nn = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Everything with "???" indicates an option with a missing value.
# Our utility functions will fill in these details using the
# environment information
cfg_nn_dict = {
    # dynamics model configuration
    "dynamics_model": {
        "_target_": "mbrl.models.GaussianMLP",
        "device": device_nn,
        "num_layers": 4,
        "ensemble_size": 7,
        "hid_size": 200,
        "in_size": "???",
        "out_size": "???",
        "deterministic": True,
        "propagation_method": "fixed_model",
        # can also configure activation function for GaussianMLP
        "activation_fn_cfg": {
            "_target_": "torch.nn.SiLU"}
    },
    # options for training the dynamics model
    "algorithm": {
        "learned_rewards": True,
        "target_is_delta": True,
        "normalize": True,
    },
    # these are experiment specific options
    "overrides": {
        "trial_length": trial_length,
        "num_steps": num_trials_nn * trial_length,
        "model_batch_size": 256,
        "validation_ratio": 1/6
        # 0.05
    }
}


# ---- Symbolic Regression params ----
# Symbolic Regression

trial_length = 600
num_trials_sr = 1
ensemble_size_sr = 1

device_sr = "cpu"

# Everything with "???" indicates an option with a missing value.
# Our utility functions will fill in these details using the
# environment information
cfg_sr_dict = {
    # dynamics model configuration
    "dynamics_model": {
        "_target_": "symbolic_mbrl.symbolic_model.SymbolicModel",
        "symbols": "add,sub,mul,div,constant,variable,sin,exp,abs",
        "population_size": 5000,
        "generations": 10000,
        "max_length": 20,
        "max_depth": 10,
        "in_size": "???",
        "out_size": "???",
        "device": device_sr,
        "deterministic": True,
        "propagation_method": None,
        "ensemble_size": ensemble_size_sr,
        "learn_reward": True
    },
    # options for training the dynamics model
    "algorithm": {
        "learned_rewards": True,
        "target_is_delta": True,
        "normalize": True,
    },
    # these are experiment specific options
    "overrides": {
        "trial_length": trial_length,
        "num_steps": num_trials_sr * trial_length,
        "model_batch_size": 1,
        "validation_ratio": 1/6
    }
}

agent_cfg_dict_simple1dmpd_sr = {
    # this class evaluates many trajectories and picks the best one
    "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
    "planning_horizon": 3,
    "replan_freq": 1,
    "verbose": False,
    "action_lb": "???",
    "action_ub": "???",
    # this is the optimizer to generate and choose a trajectory
    "optimizer_cfg": {
        "_target_": "mbrl.planning.CEMOptimizer",
        "num_iterations": 10,
        "elite_ratio": 0.1,
        "population_size": 999,
        "alpha": 0.1,
        "device": device_sr,
        "lower_bound": "???",
        "upper_bound": "???",
        "return_mean_elites": True,
        "clipped_normal": True
    }
}

agent_cfg_dict_simple1dmpd_nn = {
    # this class evaluates many trajectories and picks the best one
    "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
    "planning_horizon": 3,
    "replan_freq": 1,
    "verbose": False,
    "action_lb": "???",
    "action_ub": "???",
    # this is the optimizer to generate and choose a trajectory
    "optimizer_cfg": {
        "_target_": "mbrl.planning.CEMOptimizer",
        "num_iterations": 10,
        "elite_ratio": 0.1,
        "population_size": 999,
        "alpha": 0.1,
        "device": device_nn,
        "lower_bound": "???",
        "upper_bound": "???",
        "return_mean_elites": True,
        "clipped_normal": True
    }
}

agent_cfg_dict_cartpole_nn = {
    # this class evaluates many trajectories and picks the best one
    "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
    "planning_horizon": 15,
    "replan_freq": 1,
    "verbose": False,
    "action_lb": "???",
    "action_ub": "???",
    # this is the optimizer to generate and choose a trajectory
    "optimizer_cfg": {
        "_target_": "mbrl.planning.CEMOptimizer",
        "num_iterations": 5,
        "elite_ratio": 0.1,
        "population_size": 350,
        "alpha": 0.1,
        "device": device_nn,
        "lower_bound": "???",
        "upper_bound": "???",
        "return_mean_elites": True,
        "clipped_normal": False
    }
}
