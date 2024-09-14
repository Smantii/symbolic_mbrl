import numpy as np
import torch
from gymnasium import spaces
import gymnasium as gym
from symbolic_mbrl.config_files import *
import omegaconf
import mbrl.util.common as common_util
import mbrl.models as models
import mbrl.planning as planning
from symbolic_mbrl.pets import pets
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def reward_fn(a, next_obs):
    return torch.cos(2 * torch.pi * next_obs) * torch.exp(torch.abs(next_obs) / 3)


class Simple1DMDP(gym.Env):
    def __init__(self):
        super(Simple1DMDP, self).__init__()

        # define the action space
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32)

        # define the observation space: continuous single dimension for position
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(1,), dtype=np.float32)
        # Initialize state and episode length
        self.state = 0.0
        self.episode_length = 10
        self.current_step = 0

    def reset(self, seed=None):
        # reset the state to 0 and the step counter
        self.state = 0.0
        self.current_step = 0
        return torch.FloatTensor([self.state]), {}

    def step(self, action):
        # update state based on action
        self.state += action
        # clip in the case we go outside of [-10,10]
        # set truncated parameter
        truncated = np.abs(self.state) > 10
        self.state = np.clip(self.state, -10., 10.)

        # calculate reward
        # reward = torch.cos(2 * torch.pi * self.state) * torch.exp(torch.abs(self.state) / 3)
        reward = reward_fn(torch.from_numpy(action),
                           torch.from_numpy(self.state))
        # increment step counter
        self.current_step += 1
        # check if episode is terminated
        terminated = self.current_step >= self.episode_length

        # set placeholder for info
        info = {}
        return torch.FloatTensor([self.state]).flatten(), reward, terminated, truncated, info

    def render(self, mode='human'):
        # simple print rendering
        print(f"Step: {self.current_step}, State: {self.state}")


def main(method, device):
    # extract the proper cfg files
    if method == "SR":
        cfg_dict = cfg_sr_dict
        agent_cfg_dict = agent_cfg_dict_simple1dmpd_sr
        num_trials = num_trials_sr
    elif method == "NN":
        cfg_dict = cfg_nn_dict
        agent_cfg_dict = agent_cfg_dict_simple1dmpd_nn
        num_trials = num_trials_nn

    # Register the custom environment
    gym.envs.registration.register(
        id='Simple1DMDP-v0',
        entry_point=Simple1DMDP
    )

    env = gym.make('Simple1DMDP-v0')

    seed = 0
    env.reset()
    rng = np.random.default_rng(seed=seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    cfg = omegaconf.OmegaConf.create(cfg_dict)
    trial_length = cfg.overrides.trial_length
    ensemble_size = cfg.dynamics_model.ensemble_size

    # Create a 1-D dynamics model for this environment
    dynamics_model = common_util.create_one_dim_tr_model(
        cfg, obs_shape, act_shape)
    dynamics_model.set_elite([0, 1, 2])

    # Create a gym-like environment to encapsulate the model
    def term_fn(a, next_obs): return False
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

    pets(env, agent, dynamics_model, num_trials,
         cfg, ensemble_size, replay_buffer, method)

    # --- PLOTS ---
    num_data = 999
    data = torch.zeros((num_data, 2))
    data[:, 0] = torch.linspace(-10, 10, num_data)
    if method == "SR":
        data = torch.hstack((data, data[:, 0].reshape(-1, 1)))
        reward = dynamics_model.model.reg_reward.predict(data)
    elif method == "NN":
        dynamics_model.model.eval()
        with torch.no_grad():
            reward = dynamics_model.model(
                data.to("cuda"), propagation_indices=torch.arange(num_data))[0][:, 1]
            reward = reward.to("cpu")
    plt.plot(data[:, 0], reward, label="Predicted reward")
    plt.plot(data[:, 0], reward_fn(
        data[:, 1], data[:, 0]), label="True reward")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main("SR", device_sr)
