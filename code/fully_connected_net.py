import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# NN
class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()

      self.fc1 = nn.Linear(8, 16)
      self.relu = nn.ReLU()

      self.fc2 = nn.Linear(16, 1)
      self.sigmoid = nn.Sigmoid()

      self.dropout = nn.Dropout(0.05)

    def forward(self, x):
      x = self.fc1(x)
      x = self.relu(x)
      x = self.dropout(x)
      
      x = self.fc2(x)
      x = self.sigmoid(x)

      return x

class Net_1(nn.Module):
    def __init__(self):
      super(Net_1, self).__init__()

      self.fc1 = nn.Linear(8, 1)
      self.sigmoid = nn.Sigmoid()


    def forward(self, x):
      x = self.fc1(x)
      x = self.sigmoid(x)

      return x




class Net_2(nn.Module):
    def __init__(self):
      super(Net_2, self).__init__()

      self.fc1 = nn.Linear(2, 16)
      self.relu = nn.SELU()

      self.fc2 = nn.Linear(16, 16)
      self.fc3 = nn.Linear(16, 1)

      self.sigmoid = nn.Sigmoid()

      self.dropout = nn.Dropout(0.05)

    def forward(self, x):
      x = self.fc1(x)
      x = self.relu(x)
      x = self.dropout(x)

      x = self.fc2(x)
      x = self.relu(x)
      x = self.dropout(x)
      
      x = self.fc3(x)
      x = self.sigmoid(x)

      return x
    
class dataset(Dataset):
  def __init__(self,x,y):
    self.x = torch.tensor(x,dtype=torch.float32)
    self.y = torch.tensor(y,dtype=torch.float32)
    self.length = self.x.shape[0]
 
  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]
  def __len__(self):
    return self.length
  
def predict(model, samples):
    #model.eval()

    predictions = []
    
    samples_tensor = torch.tensor(samples).type(torch.FloatTensor)
    
    with torch.no_grad():
        predictions = model(samples_tensor)

    return predictions.cpu().numpy()