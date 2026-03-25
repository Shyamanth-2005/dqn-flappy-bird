import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
  
  def __init__(self,state_dim, action_dim, hidden_dim = 256):
    super(DQN, self).__init__()
    
    
    # hidden layer
    self.fc1 = nn.Linear(state_dim, hidden_dim)
    
    # output layer
    self.fc2 = nn.Linear(hidden_dim, action_dim)
    
  def forward(self, x):
    # x - input states (12 info)
    # activation function 
    x = F.relu(self.fc1(x)) # layer 1
    return self.fc2(x) # output layer to calculate the q value of expect return
  

if __name__ == "__main__":
  state_dim = 12
  action_dim = 2
  model = DQN(state_dim, action_dim)
  state = torch.randn(1, state_dim)
  print(model(state))



  