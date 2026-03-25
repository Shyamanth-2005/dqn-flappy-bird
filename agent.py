import yaml
import torch
import gymnasium
import itertools
import flappy_bird_gymnasium
from dqn import DQN
from experience_replay import ReplayMemory
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
  def __init__(self,hyperparameter_set):
    with open('hyperparameters.yaml', 'r') as file:
      all_hyperparameter_sets = yaml.safe_load(file)
      hyperparameter = all_hyperparameter_sets[hyperparameter_set]
      # print(hyperparameter)
      
      self.replay_memory_size = hyperparameter['replay_memory_size'] # size of the replay memory
      self.mini_batch_size = hyperparameter['mini_batch_size'] # size of the mini batch for training
      self.epsilon_init = hyperparameter['epsilon_init'] # 1 = 100 % random action (full exploration)
      self.epsilon_decay = hyperparameter['epsilon_decay'] # decay rate of epsilon per episode
      self.epsilon_min = hyperparameter['epsilon_min'] # minimum value of epsilon (minimum exploration)
  
  def run(self, is_training = True, render = False):
    #env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False) # use_lidar: [False, True]
    env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
    
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    
    rewards_per_episode = []
    policy_dqn = DQN(num_states, num_actions).to_device(DEVICE)
    
    if is_training:
      memory = ReplayMemory(self.replay_memory_size)
    
    for episode in itertools.count():
      state, _ = env.reset()
      terminated = False
      episode_reward = 0.0
      
      
      while not terminated:
          # NEXT ACTION:
          # feed the observation to your agent here
          action = env.action_space.sample()

          # processing:
          next_state, reward, terminated, _, info = env.step(action)
          
          # accumulate reward for the episode
          episode_reward += reward
          
          if is_training:
            memory.append((state, action, next_state, reward, terminated))
          
          # move to next state
          state = next_state
      
      rewards_per_episode.append(episode_reward)

