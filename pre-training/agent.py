import flappy_bird_gymnasium
import gymnasium
env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False) # use_lidar: [False, True]


"""
ACTIONS SPACE:
0 - do nothing 
1 - flap
"""
"""
REWARDS:
+ 0.1 - every frame it stays alive
+ 1.0 - successfully passing a pipe
- 1.0 - dying
- 0.5 - touch top of the screen
"""

obs, _ = env.reset()
while True:
    # NEXT ACTION:
    # feed the observation to your agent here
    action = env.action_space.sample()

    # processing:
    obs, reward, terminated, _, info = env.step(action)
    
    # checking if the player is still alive
    if terminated:
        break

env.close() 