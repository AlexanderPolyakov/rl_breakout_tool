from gymnasium.envs.registration import register

register(
    id='breakwall',
    entry_point='breakwall_clone.envs:BreakWall',
)

