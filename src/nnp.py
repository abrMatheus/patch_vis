import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE

from src.functs import min_max_norm


class NNP(nn.Module):
    def __init__(self, input_dim, proj_dim = 2):
        super().__init__()

        self.input_dim = input_dim
        self.proj_dim = proj_dim
        
        self.isize = 256

        self.norm_layer = nn.BatchNorm1d(input_dim, affine=False)
        self.norm_layer.eval()

        self.net = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.input_dim, self.isize),
            nn.ReLU(),
            nn.Linear(self.isize, self.isize),
            nn.ReLU(),
            nn.Linear(self.isize, self.isize),
            nn.ReLU(),
            nn.Linear(self.isize, self.proj_dim),
            nn.Sigmoid(),
        )

        self.criterion = torch.nn.MSELoss()
        self.set_optim()
    
    def set_optim(self):
        self.optim = optim.Adam(self.net.parameters(), lr=0.001)

    def forward(self, inputs):
        with torch.no_grad():
            x = self.norm_layer(inputs)
        return self.net(x)

    def learn_normalization_param(self, input_patches):
        self.norm_layer.running_mean = torch.tensor(input_patches.mean(0))
        self.norm_layer.running_var  = torch.tensor(input_patches.std(0)**2)

    def predict(self, patches):
        print(patches.dtype)
        X =  torch.tensor(patches).float().detach().clone().requires_grad_(False)

        self.net.eval()
        with torch.no_grad():
            pred = self.net(X)
        return pred


    def fit(self, input_patches, labels, epochs=300):

        self.learn_normalization_param(input_patches)

        acc_loss = []

        X_embedded = TSNE(n_components=2, learning_rate='auto',init='random',
                           perplexity=20).fit_transform(input_patches)

        X_embedded = min_max_norm(X_embedded)


        Xtr = torch.tensor(input_patches).float()
        Xpj = torch.tensor(X_embedded).float()

        train_ds = torch.utils.data.TensorDataset(Xtr, Xpj)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1024, shuffle=True)

        for epoch in range(epochs):  # loop over the dataset multiple times
        
            running_loss = 0.0
            for i, data in enumerate(train_dl, 0):
                inputs, labels = data
        
                self.optim.zero_grad()
    
                outputs = self.net(inputs)
                # loss = F.l1_loss(outputs, labels)
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optim.step()
        
                # print statistics
                running_loss += loss.item()
                
            acc_loss.append(running_loss/(i+1))
            print(f"Epoch {epoch}/{epochs}", flush=True, end='\r')
        
        print('Finished Training                 ')


        pred = self.predict(Xtr)
        return pred, X_embedded, acc_loss


    # def save_to_folder(self):
