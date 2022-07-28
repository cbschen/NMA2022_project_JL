from gym.envs.registration import register
# import gym
# import gym.envs.registry

# env_name = 'gym_whackamole_simple/WhackAMole-v0'
# if env_name in gym.envs.registry.env_specs:
#     del gym.envs.registry.env_specs[env_name]

register(
    id='gym_whackamole_simple/WhackAMole-v0',
    entry_point='gym_whackamole_simple.envs:WhackAMole2',
    max_episode_steps = None,
)
