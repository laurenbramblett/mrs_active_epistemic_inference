from gymnasium.envs.registration import register

register(
    id='MAMGEnv-v0',
    entry_point='mamg_env:MAMGEnv',
)