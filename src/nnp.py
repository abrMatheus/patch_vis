import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, cosine_similarity

from src.functs import min_max_norm


class NNP(nn.Module):
    def __init__(self, input_dim, proj_dim = 2, lr=0.001):
        super().__init__()

        self.input_dim = input_dim
        self.proj_dim = proj_dim
        
        self.isize = 2048

        self.mlp = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.input_dim, self.isize),
            nn.SiLU(),
            nn.Linear(self.isize, self.isize),
            nn.SiLU(),
            nn.Linear(self.isize, self.isize),
            nn.SiLU(),
            nn.Linear(self.isize, self.proj_dim),
            nn.Sigmoid(),
        )

        # self.criterion = torch.nn.MSELoss()
        self.criterion = torch.nn.L1Loss()
        self.set_optim(lr)
    
    def set_optim(self, lr=0.001):
        self.optim = optim.Adam(self.mlp.parameters(), lr=lr)

    def forward(self, x):
        return self.mlp(x)


    def predict(self, patches):
        self.mlp.cuda()
        Xtr = torch.tensor(patches).float().cuda()
        self.mlp.eval()
        with torch.no_grad():
            pred = self.forward(Xtr)
        return pred.detach().cpu().numpy()


    def fit(self, input_patches, X_embedded, epochs=300, batch_size=128):

        acc_loss = []

        Xtr = torch.tensor(input_patches).float()
        Xpj = torch.tensor(X_embedded).float()

        train_ds = torch.utils.data.TensorDataset(Xtr, Xpj)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        self.mlp.cuda()

        for epoch in range(epochs):  # loop over the dataset multiple times
        
            running_loss = 0.0
            for i, data in enumerate(train_dl, 0):
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
        
                self.optim.zero_grad()
    
                outputs = self.forward(inputs)
                # loss = F.l1_loss(outputs, labels)
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optim.step()
        
                # print statistics
                running_loss += loss.item()
                
            acc_loss.append(running_loss/(i+1))
            print(f"Epoch {epoch}/{epochs}", flush=True, end='\r')
        
        print('Finished Training                 ')


        pred = self.predict(input_patches)
        return pred, X_embedded, acc_loss


    # def save_to_folder(self):




class NNPInv(nn.Module):
    def __init__(self, input_dim, proj_dim = 2, lr=0.001):
        super().__init__()

        self.input_dim = input_dim
        self.proj_dim = proj_dim
        
        self.isizes = [128,256,512,1024]

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.isizes[0]),
            nn.SiLU(),
            nn.Dropout(),

            nn.Linear(self.isizes[0], self.isizes[1]),
            nn.SiLU(),
            nn.Dropout(),

            nn.Linear(self.isizes[1], self.isizes[2]),
            nn.SiLU(),
            nn.Dropout(),

            nn.Linear(self.isizes[2], self.isizes[3]),
            nn.SiLU(),
            nn.Dropout(),

            nn.Linear(self.isizes[3], self.proj_dim),
            nn.Sigmoid(),
        )

        # self.criterion = torch.nn.MSELoss()
        self.criterion = torch.nn.L1Loss()
        self.set_optim(lr)
    
    def set_optim(self, lr=0.001):
        self.optim = optim.Adam(self.mlp.parameters(), lr=lr)

    def forward(self, x):
        return self.mlp(x)


    def predict(self, X):
        self.mlp.cuda()
        Xtr = torch.tensor(X).float().cuda()
        self.mlp.eval()
        with torch.no_grad():
            pred = self.forward(Xtr)
        return pred.detach().cpu().numpy()


    def fit(self, X_embedded, X, epochs=300, batch_size=128):

        acc_loss = []

        Xtr = torch.tensor(X_embedded).float()
        Xgoal = torch.tensor(X).float()

        train_ds = torch.utils.data.TensorDataset(Xtr, Xgoal)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        self.mlp.cuda()

        for epoch in range(epochs):  # loop over the dataset multiple times
        
            running_loss = 0.0
            for i, data in enumerate(train_dl, 0):
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
        
                self.optim.zero_grad()
    
                outputs = self.forward(inputs)
                # loss = F.l1_loss(outputs, labels)
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optim.step()
        
                # print statistics
                running_loss += loss.item()
                
            acc_loss.append(running_loss/(i+1))
            print(f"Epoch {epoch}/{epochs}", flush=True, end='\r')
        
        print('Finished Training                 ')


        pred = self.predict(X_embedded)
        return pred, X_embedded, acc_loss


    # def save_to_folder(self):
