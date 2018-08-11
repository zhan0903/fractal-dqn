from fractalai.model import RandomDiscreteModel
from fractalai.environment import ExternalProcess, ParallelEnvironment, AtariEnvironment
from fractalai.fractalmc import FractalMC


name = "Pong-v0"
render = False # It is funnier if the game is displayed on the screen
clone_seeds = True  # This will speed things up a bit
max_steps = 1e6  # Play until the game is finished.
n_repeat_action = 1  # Atari games run at 20 fps, so taking 4 actions per seconds is more
reward_limit = 21
render_every = 2

#dt means time step
dt_mean = 3
dt_std = 2
min_dt = 3

max_samples = 3000#3000  # Let see how well it can perform using at most 300 samples per step
max_walkers = 100#100 # Let's set a really small number to make everthing faster
time_horizon = 30#30  # 50 frames should be enough to realise you have been eaten by a ghost


if __name__ == "__main__":
    # 16个并行环境，创建16个进程
    env = ParallelEnvironment(name=name,env_class=AtariEnvironment,
                              blocking=False, n_workers=4, n_repeat_action=n_repeat_action)  # We will play an Atari game
    model = RandomDiscreteModel(max_wakers=max_walkers,
                                n_actions=env.n_actions)# The Agent will take discrete actions at random
    # print(env.unwrapped.get_action_meanings())

    fmc = FractalMC(model=model, env=env, max_walkers=max_walkers,
                    reward_limit=reward_limit, render_every=render_every,
                    time_horizon=time_horizon, dt_mean=dt_mean, dt_std=dt_std, accumulate_rewards=True, min_dt=min_dt)
    fmc.run_agent(render=False, print_swarm=False)
    # # train dqn model
    #
    # # train dqn model
    # num_episodes = 500
    # for i_episode in range(num_episodes):
    #     dqn_agent.optimize_model()
    #     if i_episode % dqn_agent.target_update == 0:
    #         dqn_agent.target_net.load_state_dict(dqn_agent.policy_net.state_dict())
    #
    # # text dqn model
    # env = gym.make('Pong-v0').unwrapped
    # observation = env.reset()

    # fmc.render_game(sleep=1/40)
