from symbolic_mbrl.config_files import *
import mbrl.env.cartpole_continuous as cartpole_env
import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns
import omegaconf
import numpy as np
import mbrl.util.common as common_util
import mbrl.models as models
import mbrl.planning as planning
from symbolic_mbrl.pets import pets
import matplotlib.pyplot as plt


def main(method, device):
    # extract the proper cfg files
    if method == "SR":
        cfg_dict = cfg_sr_dict
        num_trials = num_trials_sr

    elif method == "NN":
        cfg_dict = cfg_nn_dict
        num_trials = num_trials_nn

    agent_cfg_dict = agent_cfg_dict_cartpole
    agent_cfg_dict["optimizer_cfg"]["device"] = device
    seed = 0
    env = cartpole_env.CartPoleEnv(render_mode="rgb_array")
    env.reset(seed)
    rng = np.random.default_rng(seed=0)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # This functions allows the model to evaluate the true rewards given an observation
    reward_fn = reward_fns.cartpole
    # This function allows the model to know if an observation should make the episode end
    term_fn = termination_fns.cartpole

    cfg = omegaconf.OmegaConf.create(cfg_dict)
    trial_length = cfg.overrides.trial_length
    ensemble_size = cfg.dynamics_model.ensemble_size

    # Create a 1-D dynamics model for this environment
    dynamics_model = common_util.create_one_dim_tr_model(
        cfg, obs_shape, act_shape)

    # Create a gym-like environment to encapsulate the model
    model_env = models.ModelEnv(env, dynamics_model, term_fn,
                                reward_fn, generator=generator)

    dtype = np.float32
    replay_buffer = common_util.create_replay_buffer(
        cfg, obs_shape, act_shape, rng=rng, obs_type=dtype,
        action_type=dtype, reward_type=dtype,)

    # generate data
    common_util.rollout_agent_trajectories(
        env,
        trial_length,  # initial exploration steps
        planning.RandomAgent(env),
        {},  # keyword arguments to pass to agent.act()
        replay_buffer=replay_buffer,
        trial_length=trial_length
    )

    agent_cfg = omegaconf.OmegaConf.create(agent_cfg_dict)

    agent = planning.create_trajectory_optim_agent_for_model(
        model_env,
        agent_cfg,
        num_particles=20
    )

    all_rewards, all_mse = pets(env, agent, dynamics_model, num_trials,
                                cfg, ensemble_size, replay_buffer, method)

    return all_rewards, all_mse


if __name__ == "__main__":
    num_runs = 3

    all_reward_sr = np.zeros((10, num_runs))
    all_reward_nn = np.zeros((10, num_runs))
    all_mse_sr = np.zeros((10, num_runs))
    all_mse_nn = np.zeros((10, num_runs))

    # for i in range(num_runs):
    #    all_reward_sr[:, i], all_mse_sr[:, i] = main("SR", device_sr)
    #    all_reward_nn[:, i], all_mse_nn[:, i] = main("NN", device_nn)

    # np.save("cartpole_rewards_sr.npy", all_reward_sr)
    # np.save("cartpole_mse_sr.npy", all_mse_sr)
    # np.save("cartpole_rewards_nn.npy", all_reward_nn)
    # np.save("cartpole_mse_nn.npy", all_mse_nn)

    num_training_data = 10*np.arange(1, 11)

    all_reward_sr = np.load("cartpole_rewards_sr.npy")
    all_reward_nn = np.load("cartpole_rewards_nn.npy")
    all_mse_sr = np.load("cartpole_mse_sr.npy")
    all_mse_nn = np.load("cartpole_mse_nn.npy")

    # mean std reward across multiple runs
    mean_reward_sr = np.mean(all_reward_sr, axis=1)
    std_reward_sr = np.std(all_reward_sr, axis=1)
    mean_reward_nn = np.mean(all_reward_nn, axis=1)
    std_reward_nn = np.std(all_reward_nn, axis=1)

    # mean std mse across multiple runs
    mean_mse_sr = np.mean(all_mse_sr, axis=1)
    std_mse_sr = np.std(all_mse_sr, axis=1)
    mean_mse_nn = np.mean(all_mse_nn, axis=1)
    std_mse_nn = np.std(all_mse_nn, axis=1)

    # --- PLOTS ---
    width = 443.57848
    fontsize = 8
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['font.sans-serif'] = 'Dejavu Sans'
    plt.rcParams['font.family'] = 'sans-serif'

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Figure width in inches
    fig_width_in = width * inches_per_pt

    fig_dim = (fig_width_in, 3.)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=fig_dim)

    ax[0].plot(num_training_data,
               mean_reward_sr, c="#e41a1c", label="SR-PETS")
    ax[0].fill_between(num_training_data, mean_reward_sr - std_reward_sr,
                       mean_reward_sr + std_reward_sr, color="#e41a1c", alpha=0.5)
    ax[0].plot(num_training_data,
               mean_reward_nn, c="#4daf4a", label="NN-PETS")
    ax[0].fill_between(num_training_data, mean_reward_nn - std_reward_nn,
                       mean_reward_nn + std_reward_nn, color="#4daf4a", alpha=0.5)
    ax[0].set_xlabel("# Training Data")
    ax[0].set_ylabel("Reward")
    ax[0].legend()
    ax[0].set_xlim(10, 100)
    ax[0].text(-0.3, 1., "A",
               transform=ax[0].transAxes, size=fontsize, weight="bold")

    ax[1].plot(num_training_data,
               mean_mse_sr, c="#e41a1c", label="SR-PETS")
    ax[1].plot(num_training_data,
               mean_mse_nn, c="#4daf4a", label="NN-PETS")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("# Training Data")
    ax[1].set_ylabel("Validation MSE")
    ax[1].legend()
    ax[1].set_xlim(10, 100)
    ax[1].text(-0.35, 1., "B",
               transform=ax[1].transAxes, size=fontsize, weight="bold")

    fig.tight_layout()

    plt.savefig("cartpole.pdf", dpi=300)
