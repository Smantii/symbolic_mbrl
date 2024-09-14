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


def main(method, device):
    # extract the proper cfg files
    if method == "SR":
        assert False
    elif method == "NN":
        cfg_dict = cfg_nn_dict
        agent_cfg_dict = agent_cfg_dict_cartpole_nn
        num_trials = num_trials_nn

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

    pets(env, agent, dynamics_model, num_trials,
         cfg, ensemble_size, replay_buffer, method)

    # --- PLOTS ---


if __name__ == "__main__":
    main("NN", device_nn)
