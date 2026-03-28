import os
import yaml
import torch
import random
import argparse
import itertools
import matplotlib
import numpy as np
import gymnasium as gym
import flappy_bird_gymnasium
import matplotlib.pyplot as plt


from dqn import DQN
from torch import nn
from datetime import datetime, timedelta
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
# for printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok = True)

# Agg: used to generate  plots as images and save them to a file instead of rendering  to screen
matplotlib.use("Agg")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu" # force cpu



class Agent:
  def __init__(self, hyperparameter_set):
    self.hyperparameter_set = hyperparameter_set  # store the hyperparameter set name
    
    with open('hyperparameters.yaml', 'r') as file:
      all_hyperparameter_sets = yaml.safe_load(file)
      hyperparameters = all_hyperparameter_sets[hyperparameter_set]
      # print(hyperparameter)
    self.env_id             = hyperparameters['env_id']
    self.learning_rate_a    = hyperparameters['learning_rate_a']        # learning rate (alpha)
    self.discount_factor_g  = hyperparameters['discount_factor_g']      # discount rate (gamma)
    self.network_sync_rate  = hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
    self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
    self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
    self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
    self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
    self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
    self.stop_on_reward     = hyperparameters['stop_on_reward']         # stop training after reaching this number of rewards
    self.fc1_nodes          = hyperparameters['fc1_nodes']
    self.env_make_params    = hyperparameters.get('env_make_params',{}) # Get optional environment-specific parameters, default to empty dict
    self.enable_double_dqn  = hyperparameters['enable_double_dqn']      # double dqn on/off flag
    self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']     # dueling dqn on/off flag


    self.loss_fn = nn.MSELoss()
    self.optimizer = None
    # path to run info
    self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
    self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
    self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')
  
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
    if is_training:
      start_time = datetime.now()
      last_graph_update_time = start_time

      log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
      print(log_message)
      with open(self.LOG_FILE, 'w') as file:
        file.write(log_message + '\n')
        
    # env = gym.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False) # use_lidar: [False, True]
    # env = gym.make("CartPole-v1", render_mode="human" if render else None)
    env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)
    
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    
    rewards_per_episode = []
    
    
    
    policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(DEVICE)
    
    if is_training:
      epsilon = self.epsilon_init
      memory = ReplayMemory(self.replay_memory_size)
      
      
      # tagret network for better estimates
      # why two networks when policy network trains we copy a set of 
      # it to the target network so we can stablize the training (instead of moving targets)
      # and get better results
      target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(DEVICE)
      target_dqn.load_state_dict(policy_dqn.state_dict())
      

      
      # policy network optimizer , "Adam" optimizer
      self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr = self.learning_rate_a)
      
      epsilon_history = []
      
      # track number of steps taken , used for syncing policy => target network
      step_count = 0
      
      # best reward
      best_reward = -9999999
    else:
      # load learned policy
      policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, weights_only=True, map_location=DEVICE))
      
      # switch model to evaluation mode
      policy_dqn.eval()
    
    # train indefinitely, manually  stop the run when ur satisfied with the results
    for episode in itertools.count():
      state, _ = env.reset()
      state = torch.tensor(state, dtype = torch.float, device = DEVICE)
      
      
      
      
      terminated = False
      episode_reward = 0.0
      
      
      while (not terminated and episode_reward  < self.stop_on_reward):
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
      
      # save model when new best reward is obtained.
      if is_training:
        if episode_reward > best_reward:
          log_message =  f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
          print(log_message)
          with open(self.LOG_FILE, 'a') as file:
            file.write(log_message + '\n')

          torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
          best_reward = episode_reward

        # update graph every x seconds
        current_time = datetime.now()
        if current_time - last_graph_update_time > timedelta(seconds = 10):
          self.save_graph(rewards_per_episode, epsilon_history)
          last_graph_update_time = current_time

        # if enough experience has been collected
        if len(memory) > self.mini_batch_size:

          # sample from memory
          mini_batch = memory.sample(self.mini_batch_size)

          self.optimize(mini_batch, policy_dqn, target_dqn)

          # decay epsilon
          epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
          epsilon_history.append(epsilon)

          # copy policy network to target network after a certain number of steps
          if step_count > self.network_sync_rate:
            target_dqn.load_state_dict(policy_dqn.state_dict())
            step_count = 0
  
  def save_graph(self, rewards_per_episode, epsilon_history):
    # save plots
    fig = plt.figure(1)

    # plot average rewards (Y-axis) vs episodes (X-axis)
    mean_rewards = np.zeros(len(rewards_per_episode))
    for x in range(len(mean_rewards)):
      mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
    plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
    # plt.xlabel('Episodes')
    plt.ylabel('Mean Rewards')
    plt.plot(mean_rewards)

    # plot epsilon decay (Y-axis) vs episodes (X-axis)
    plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
    # plt.xlabel('Time Steps')
    plt.ylabel('Epsilon Decay')
    plt.plot(epsilon_history)

    plt.subplots_adjust(wspace=1.0, hspace=1.0)

    # save plots
    fig.savefig(self.GRAPH_FILE)
    plt.close(fig)  
      
      
      

if __name__ == "__main__":
  # parser command line inputs
  parser = argparse.ArgumentParser(description = "train or test model.")
  parser.add_argument("hyperparameters", help = "")
  parser.add_argument("--train", help = "training mode", action = "store_true")
  args = parser.parse_args()
  
  dql = Agent(hyperparameter_set = args.hyperparameters)

  if args.train:
    dql.run(is_training = True)
  else:
    dql.run(is_training = False, render = True)