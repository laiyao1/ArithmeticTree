from gym.envs.registration import register

register(
	id = 'multiplier-openroad-v0',
	entry_point = 'multiplier_env.multiplier_openroad_env:MultiplierEnv'
)
