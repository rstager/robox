import gym

gym.envs.registration.register(
    id='RoboschoolReacherTest3-v1',
    entry_point='gym_nreacher:RoboschoolReacher',
    max_episode_steps=150,
    reward_threshold=18.0,
    kwargs={'xml': 'reacher3.xml'}
)
gym.envs.registration.register(
    id='RoboschoolReacherTestflat4-v1',
    entry_point='gym_nreacher:RoboschoolReacher',
    max_episode_steps=150,
    reward_threshold=18.0,
    kwargs={'xml': 'reacherflat4.xml'}
)
gym.envs.registration.register(
    id='RoboschoolReacherTest4-v1',
    entry_point='gym_nreacher:RoboschoolReacher',
    max_episode_steps=150,
    reward_threshold=18.0,
    kwargs={'xml': 'reacher4.xml'}
)
gym.envs.registration.register(
    id='RoboschoolReacherTest2-v1',
    entry_point='gym_nreacher:RoboschoolReacher',
    max_episode_steps=150,
    reward_threshold=18.0,
    kwargs={'xml': 'reacher.xml'}
)
gym.envs.registration.register(
    id='RoboschoolReacherTest5-v1',
    entry_point='gym_nreacher:RoboschoolReacher',
    max_episode_steps=400,
    reward_threshold=18.0,
    kwargs={'xml': 'reacher5.xml'}
)