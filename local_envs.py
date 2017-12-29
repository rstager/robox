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
    max_episode_steps=150,
    reward_threshold=18.0,
    kwargs={'xml': 'reacher5.xml'}
)
gym.envs.registration.register(
    id='RoboschoolReacherTest8-v1',
    entry_point='gym_nreacher:RoboschoolReacher',
    max_episode_steps=150,
    reward_threshold=18.0,
    kwargs={'xml': 'reacher8.xml'}
)
gym.envs.registration.register(
    id='RoboschoolReacherI3-v1',
    entry_point='gym_nreacher:RoboschoolReacherImage',
    max_episode_steps=150,
    reward_threshold=18.0,
    kwargs={'xml': 'reacher3.xml','cameras':[(320,240)]}
)
gym.envs.registration.register(
    id='RoboschoolReacherI8-v1',
    entry_point='gym_nreacher:RoboschoolReacherImage',
    max_episode_steps=150,
    reward_threshold=18.0,
    kwargs={'xml': 'reacher8.xml','cameras':[(320,240)]}
)
gym.envs.registration.register(
    id='RoboschoolReacherI5-v1',
    entry_point='gym_nreacher:RoboschoolReacher',
    max_episode_steps=150,
    reward_threshold=18.0,
    kwargs={'xml': 'reacherI5.xml','cameras':[(320,240)]}
)
gym.envs.registration.register(
    id='KukaI-v1',
    entry_point='gym_kuka:KukaCamGymEnv',
    max_episode_steps=350,
    kwargs={'renders':True}
)

gym.envs.registration.register(
    id='BulletReacher8-v1',
    entry_point='gym_bullet_nreacher:BulletReacher',
    max_episode_steps=150,
    reward_threshold=18.0,
    kwargs={'xml': 'reacher8.xml'}
)