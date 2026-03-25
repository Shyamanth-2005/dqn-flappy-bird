import yaml
import torch
import random
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
    epsilon_history = []
    
    
    policy_dqn = DQN(num_states, num_actions).to(DEVICE)
    
    if is_training:
      memory = ReplayMemory(self.replay_memory_size)
      
      epsilon = self.epsilon_init
      
    
    for episode in itertools.count():
      state, _ = env.reset()
      state = torch.tensor(state, dtype = torch.float, device = DEVICE)
      
      
      
      
      terminated = False
      episode_reward = 0.0
      
      
      while not terminated:
          # NEXT ACTION:
          # feed the observation to your agent here
          if is_training and random.random() < epsilon:
            action = env.action_space.sample()
            action = torch.tensor(action, dtype = torch.int64, device = DEVICE)
          else:
            # we don't need gradient as we are just do evaluation
            with torch.no_grad():
              # tensor([1,2,3,..]) => tensor([[1,2,3, ...]])
              
              # so we need to add an extra dim at the begining 
              action = policy_dqn(state.unsqueeze(dim = 0)).squeeze().argmax()

          # processing:
          next_state, reward, terminated, _, info = env.step(action.item())
          
          # accumulate reward for the episode
          episode_reward += reward
          
          # convert next_state and rewards to tensor
          next_state = torch.tensor(next_state, dtype = torch.float, device = DEVICE)
          reward = torch.tensor(reward, dtype = torch.float, device = DEVICE)
          
          if is_training:
            memory.append((state, action, next_state, reward, terminated))
          
          # move to next state
          state = next_state
      
      rewards_per_episode.append(episode_reward)
     
      epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
      epsilon_history.append(epsilon)

if __name__ == "__main__":
  agent = Agent("cartpole1")
  agent.run(render = True)
