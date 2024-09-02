import mbrl.util.common as common_util


# def train_callback(_model, _total_calls, _epoch, tr_loss, val_score, _best_val):
def train_callback(train_losses, val_scores, tr_loss, val_score):
    train_losses.append(tr_loss)
    val_scores.append(val_score.mean().item())


def pets(env, agent, dynamics_model, num_trials, cfg, ensemble_size, replay_buffer, method):
    model_trainer = None
    # Create a trainer for the model
    # model_trainer = models.ModelTrainer(dynamics_model, optim_lr=7.5e-4, weight_decay=3e-5)
    # model_trainer = SymbolicModelTrainer(dynamics_model)
    # added_data = []
    # Main PETS loop
    all_rewards = [0]
    for trial in range(num_trials):
        print(trial)
        obs, _ = env.reset()
        agent.reset()

        terminated = False
        total_reward = 0.0
        steps_trial = 0
        while not terminated:
            # --------------- Model Training -----------------
            if steps_trial == 0:
                dynamics_model.update_normalizer(
                    replay_buffer.get_all())  # update normalizer stats

                dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
                    replay_buffer,
                    batch_size=cfg.overrides.model_batch_size,
                    val_ratio=cfg.overrides.validation_ratio,
                    ensemble_size=ensemble_size,
                    shuffle_each_epoch=True,
                    bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
                )

                print(dataset_train.num_stored, dataset_val.num_stored)

                # process and standardize the data
                # X_train = np.hstack((dataset_train.transitions.obs, dataset_train.transitions.act))
                # X_val = np.hstack((dataset_val.transitions.obs, dataset_val.transitions.act))
                # y_train = dataset_train.transitions.rewards
                # y_val = dataset_val.transitions.rewards
                # mean_X_train = np.mean(X_train, axis = 0)
                # std_X_train = np.std(X_train, axis = 0)
                # mean_y_train = np.mean(y_train, axis = 0)
                # std_y_train = np.std(y_train, axis = 0)
                # X_train_norm = (X_train - mean_X_train)/std_X_train
                # y_train_norm = (y_train - mean_y_train)/std_y_train
                # X_val_norm = (X_val - mean_X_train)/std_X_train
                # y_val_norm = (y_val - mean_y_train)/std_y_train

                # dynamics_model.model.update_mean_std(mean_X_train, std_X_train, mean_y_train, std_y_train)
                dynamics_model.model.update_mean_std(0, 1, 0, 1)

                # model_trainer.train(
                #    dataset_train,
                #    dataset_val=dataset_val,
                #   num_epochs=2000,
                #    patience=25,
                #    callback=train_callback,
                #    silent=True)

                # train_r2, val_r2 = model_trainer.train(X_train_norm, y_train_norm, X_val_norm, y_val_norm)
                train_r2, val_r2 = model_trainer.train(dataset_train, dataset_val)
                print(train_r2, val_r2)

            # --- Doing env step using the agent and adding to model dataset ---
            next_obs, reward, terminated, _, _ = common_util.step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer)

            # added_data.append([next_obs, reward, terminated, truncated])

            obs = next_obs
            total_reward += reward
            steps_trial += 1

            if steps_trial == cfg.overrides.trial_length:
                break

        all_rewards.append(total_reward)
