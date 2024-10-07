import numpy as np
import torch
from pyoperon.sklearn import SymbolicRegressor
from mbrl.models import Ensemble


class SymbolicModel(Ensemble):
    def __init__(self, symbols, population_size, generations, max_length, max_depth, in_size, out_size,
                 ensemble_size, device, propagation_method, deterministic, learn_reward):
        super().__init__(ensemble_size, device, propagation_method, deterministic)
        self.in_size = in_size
        self.out_size = out_size
        self.ensemble_size = ensemble_size
        self.learn_reward = learn_reward
        sr_params = {'population_size': population_size,
                     'allowed_symbols': symbols,
                     'optimizer_iterations': 10,
                     'generations': generations,
                     'n_threads': 32,
                     'max_length': max_length,
                     'max_depth': max_depth}
        self.reg_next_obs = []
        for _ in range(self.out_size):
            self.reg_next_obs.append(SymbolicRegressor(**sr_params))
        if self.learn_reward:
            self.reg_reward = SymbolicRegressor(**sr_params)

    def forward(self, x, rng, propagation_indices):
        next_obs = np.zeros((x.shape[0], self.out_size))
        for i in range(self.out_size):
            next_obs[:, i] = self.reg_next_obs[i].predict(x)
        if self.learn_reward:
            obs_act_next_obs = np.hstack((x, next_obs.reshape(-1, 1)))
            reward = self.reg_reward.predict(obs_act_next_obs)
            preds = np.vstack((next_obs, reward)).T
        else:
            preds = next_obs
        return torch.from_numpy(preds), None

    def loss(self, model_in, target):
        return self.reg.score(model_in, target)

    def eval_score(self, model_in, target):
        return self.reg.score(model_in, target)

    def sample_propagation_indices(
            self, batch_size: int, _rng: torch.Generator) -> torch.Tensor:
        model_len = self.num_members
        if batch_size % model_len != 0:
            raise ValueError(
                "To use SymbolicModel's ensemble propagation, the batch size must "
                "be a multiple of the number of models in the ensemble."
            )
        # rng causes segmentation fault, see https://github.com/pytorch/pytorch/issues/44714
        return torch.randperm(batch_size, device=self.device)


class SymbolicModelTrainer:
    def __init__(self, dynamics_model: SymbolicModel):
        self.dynamics_model = dynamics_model

    # def train(self, X_train, y_train, X_val, y_val):
    def train(self, dataset_train, dataset_val):
        train_scores = []
        val_scores = []
        out_size = self.dynamics_model.model.out_size
        X_train = np.hstack((dataset_train.transitions.obs,
                            dataset_train.transitions.act,
                            dataset_train.transitions.next_obs))
        X_val = np.hstack((dataset_val.transitions.obs,
                          dataset_val.transitions.act,
                          dataset_val.transitions.next_obs))
        y_train_reward = dataset_train.transitions.rewards
        y_val_reward = dataset_val.transitions.rewards
        y_train_next_obs = dataset_train.transitions.next_obs
        y_val_next_obs = dataset_val.transitions.next_obs

        # fitting next obs
        for i in range(out_size):
            reg_next_obs = self.dynamics_model.model.reg_next_obs[i]
            reg_next_obs.fit(X_train[:, :-out_size],
                             y_train_next_obs[:, i])
            train_scores.append(reg_next_obs.score(
                X_train[:, :-out_size], y_train_next_obs[:, i]))
            val_scores.append(reg_next_obs.score(
                X_val[:, :-out_size], y_val_next_obs[:, i]))
            print(reg_next_obs.get_model_string(reg_next_obs.model_))

        # fitting reward
        if self.dynamics_model.model.learn_reward:
            reg_reward = self.dynamics_model.model.reg_reward
            reg_reward.fit(X_train, y_train_reward)
            train_scores.append(reg_reward.score(X_train, y_train_reward))
            val_scores.append(reg_reward.score(X_val, y_val_reward))
            print(reg_reward.get_model_string(reg_reward.model_))

        return train_scores, val_scores
