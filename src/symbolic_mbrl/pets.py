import mbrl.util.common as common_util
import mbrl.models as models
from symbolic_mbrl.symbolic_model import SymbolicModelTrainer
from functools import partial
import numpy as np


def pets(env, agent, dynamics_model, num_trials, cfg, ensemble_size, replay_buffer, method):
    if method == "SR":
        # Create a trainer for the model
        model_trainer = SymbolicModelTrainer(dynamics_model)
        train_function = model_trainer.train
    elif method == "NN":
        # Create a trainer for the model
        model_trainer = models.ModelTrainer(
            dynamics_model, optim_lr=7.5e-4, weight_decay=3e-5)
        train_function = partial(model_trainer.train, num_epochs=2000,
                                 patience=25, silent=True)
    else:
        raise ValueError("The only usable methods are SR and NN")
    # Main PETS loop
    all_rewards = []
    all_mse = []
    num_data_to_be_added = 10
    for i in range(num_trials):
        obs, _ = env.reset()
        agent.reset()

        terminated = False
        total_reward = 0.0
        steps_trial = 0
        if i == 0:
            data_length = cfg.overrides.trial_length
            val_ratio = cfg.overrides.validation_ratio
            train_length = (1-val_ratio)*data_length

        else:
            train_length += num_data_to_be_added
            data_length += num_data_to_be_added
            val_ratio = 1 - train_length / data_length
        while not terminated:
            # --------------- Model Training -----------------
            if steps_trial == 0:
                dynamics_model.update_normalizer(
                    replay_buffer.get_all())  # update normalizer stats

                dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
                    replay_buffer,
                    batch_size=cfg.overrides.model_batch_size,
                    val_ratio=val_ratio,
                    ensemble_size=ensemble_size,
                    shuffle_each_epoch=True,
                    bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
                )

                _, hist_val = train_function(
                    dataset_train, dataset_val)
                if method == "SR":
                    all_mse.append(np.mean(hist_val))
                elif method == "NN":
                    all_mse.append(hist_val[-1])

            if steps_trial <= num_data_to_be_added:
                # --- Doing env step using the agent and adding to model dataset ---
                next_obs, reward, terminated, _, _ = common_util.step_env_and_add_to_buffer(
                    env, obs, agent, {}, replay_buffer)
            else:
                action = agent.act(obs)
                next_obs, reward, terminated, _, _ = env.step(
                    action)

            obs = next_obs
            total_reward += reward
            steps_trial += 1

            # if steps_trial == cfg.overrides.trial_length:
            if steps_trial == 200:
                break

        all_rewards.append(total_reward)
        print(total_reward)
    return all_rewards, all_mse
