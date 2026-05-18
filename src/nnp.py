import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, cosine_similarity

from src.functs import min_max_norm


class NNP(nn.Module):
    def __init__(self, input_dim, proj_dim = 2):
        super().__init__()

        self.input_dim = input_dim
        self.proj_dim = proj_dim
        
        self.isize = 256

        self.register_buffer('mean', torch.zeros(self.input_dim))
        self.register_buffer('std', torch.ones(self.input_dim))

        self.mlp = nn.Sequential(
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
        self.optim = optim.Adam(self.mlp.parameters(), lr=0.001)

    def forward(self, x):
        with torch.no_grad():
            x = (x - self.mean) / (self.std + 1e-8)
        return self.mlp(x)

    def learn_normalization_param(self, input_patches):
        self.smean = input_patches.mean(0)
        self.sdev = input_patches.std(0)
        with torch.no_grad():
            self.mean.copy_(torch.tensor(input_patches.mean(0)))
            self.std.copy_(torch.tensor(input_patches.std(0)))

    def predict(self, patches):
        X =  torch.tensor(patches).float().requires_grad_(False)

        self.mlp.eval()
        with torch.no_grad():
            pred = self.forward(X)
        return pred


    def fit(self, input_patches, labels, epochs=300, use_metric="euclidean"):

        self.learn_normalization_param(input_patches)

        acc_loss = []

        if use_metric=="euclidean_distance":
            pair_metric = euclidean_distances(input_patches)
        elif use_metric =="cosine_distance":
            pair_metric = cosine_distances(input_patches)
        elif use_metric == "cosine_similarity":
            pair_metric = cosine_similarity(input_patches)
        
        X_embedded = TSNE(n_components=2, learning_rate='auto',init='random',
                            perplexity=20, metric='precomputed').fit_transform(pair_metric)

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
