import flappy_bird_gymnasium
import gymnasium
import torch
from dqn import DQN


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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
class Agent:
  def run(self, is_training = True, render = False):
    env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False) # use_lidar: [False, True]
    
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    policy_dqn = DQN(num_states, num_actions).to_device(DEVICE)
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