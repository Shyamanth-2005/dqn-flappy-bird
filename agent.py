import yaml
import torch
import random
import itertools
import gymnasium
from dqn import DQN
from torch import nn
import flappy_bird_gymnasium
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
      hyperparameters = all_hyperparameter_sets[hyperparameter_set]
      # print(hyperparameter)
      
      self.replay_memory_size = hyperparameters['replay_memory_size'] # size of the replay memory
      self.mini_batch_size = hyperparameters['mini_batch_size'] # size of the mini batch for training
      self.epsilon_init = hyperparameters['epsilon_init'] # 1 = 100 % random action (full exploration)
      self.epsilon_decay = hyperparameters['epsilon_decay'] # decay rate of epsilon per episode
      self.epsilon_min = hyperparameters['epsilon_min'] # minimum value of epsilon (minimum exploration)
      
      self.network_sync_rate = hyperparameters['network_sync_rate'] # no of steps (rate) to copy the next policy network to the target network 
      self.learning_rate_a = hyperparameters['learning_rate_a'] # learning rate -> how fast or slow the model learns
      self.discount_factor_g = hyperparameters['discount_factor_g'] # decides how imp future rewards are
      
      
      self.loss_fn = nn.MSELoss()
      self.optimizer = None
  
  def optimize(self, mini_batch, policy_dqn, target_dqn):
    """ 
    q learning formula
    q[state, action]  = q[state, action] + learning+rate * (reward + discount_factor * max(q[next_state, action]) - a[state, action])
    
    dqn target formula
    q[state, action] = reward if next_state is terminal else
    reward + discount_factor * max(q[next_state,actions])
    """
    # transpose the list of experiences and seperate each element
    states, actions, next_states, rewards, terminations = zip(*mini_batch)

    # stack tensors to create batch tensors
    # tensor([1, 2, 3])

    states = torch.stack(states)
    actions = torch.stack(actions)
    next_states = torch.stack(next_states)
    rewards = torch.tensor(rewards).float().to(DEVICE)
    terminations = torch.tensor(terminations).float().to(DEVICE)

    with torch.no_grad():
       # calculate target q values (expected returns )
       target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(next_states).max(dim = 1)[0]
       """
       target_dqn(next_states) => tensor([[1,2,3], [4,5,6]])
        .max(dim = 1) => torch.return_types.max(values = tensor([3, 6]), indices = tensor([3, 0, 0 ,1]))
          [0] => tensor([3, 6])
       """
    # calculate q values from the current policy
    current_q = policy_dqn(states).gather(dim = 1, index = actions.unsqueeze(dim = 1)).squeeze()
    """
      policy_dqn(states) => tensor([1,2,3]. [4,5,6])
        actions.
    """


      
    # compute loss
    loss = self.loss_fn(current_q, target_q)
      
    # optimize the model
    self.optimizer.zero_grad() # clear gradients
    loss.backward() # compute gradients (backwardpropagation) or direction of weight updat
    self.optimizer.step() # update network parameters ie. weights and bais
      
    
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
      
      # tagret network for better estimates
      # why two networks when policy network trains we copy a set of 
      # it to the target network so we can stablize the training (instead of moving targets)
      # and get better results
      target_dqn = DQN(num_states, num_actions).to(DEVICE)
      target_dqn.load_state_dict(policy_dqn.state_dict())
      
      # track number of steps taken , used for syncing policy => target network
      step_count = 0
      
      # policy network optimizer , "Adam" optimizer
      self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr = self.learning_rate_a)
      
      
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
            # save experience into memory
            memory.append((state, action, next_state, reward, terminated))

            # increment step counter
            step_count += 1
          # move to next state
          state = next_state
      
      # keep track of the rewards collected per episode
      rewards_per_episode.append(episode_reward)
     
      epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
      epsilon_history.append(epsilon)
      
      # if enough experience has been collected
      if len(memory) > self.mini_batch_size:
        
        # sample from memory
        mini_batch = memory.sample(self.mini_batch_size)
        
        self.optimize(mini_batch, policy_dqn, target_dqn)
        
        # copy policy network to target network after a certain number of steps
        if step_count > self.network_sync_rate:
          target_dqn.load_state_dict(policy_dqn.state_dict())
          step_count = 0
        
      
      
      

if __name__ == "__main__":
  agent = Agent("cartpole1")
  agent.run(render = True)
