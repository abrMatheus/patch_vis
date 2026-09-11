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


    def predict_no_grad(self, patches):
        self.mlp.cuda()
        Xtr = torch.tensor(patches).float().cuda()
        self.mlp.eval()
        with torch.no_grad():
            pred = self.forward(Xtr)
        return pred


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


        pred = self.predict_no_grad(input_patches)
        return pred, X_embedded, acc_loss


    # def save_to_folder(self):


    def fit_val(self, X_train, P_train, X_val, P_val, epochs=300, batch_size=128):

        tr_loss = []
        va_loss = []

        Xtr = torch.tensor(X_train).float()
        Ptr = torch.tensor(P_train).float()
        Xva = torch.tensor(X_val).float()
        Pva = torch.tensor(P_val).float()

        train_ds = torch.utils.data.TensorDataset(Xtr, Ptr)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_ds = torch.utils.data.TensorDataset(Xva, Pva)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True)

        self.mlp.cuda()

        for epoch in range(epochs):  # loop over the dataset multiple times
        
            self.mlp.train()
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
                
            tr_loss.append(running_loss/(i+1))
            print(f"Epoch {epoch}/{epochs}", flush=True, end='\r')
        

            with torch.no_grad():
                self.mlp.eval()
                running_loss = 0.0
                for i, data in enumerate(val_dl, 0):
                    inputs, labels = data
                    inputs = inputs.cuda()
                    labels = labels.cuda()
            
                    outputs = self.forward(inputs)
                    
                    loss = self.criterion(outputs, labels)
            
                    # print statistics
                    running_loss += loss.item()
                    
                va_loss.append(running_loss/(i+1))

        print('Finished Training                 ')


        pred = self.predict_no_grad(X_train)
        return pred, tr_loss, va_loss



class NNPInv(nn.Module):
    def __init__(self, input_dim, proj_dim = 2, lr=0.001):
        super().__init__()

        self.input_dim = input_dim
        self.proj_dim = proj_dim
        
        self.isizes = [128,256,512,1024]

        dropout_rate=0.1
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.isizes[0]),
            # nn.SiLU(),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(self.isizes[0], self.isizes[1]),
            # nn.SiLU(),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(self.isizes[1], self.isizes[2]),
            # nn.SiLU(),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(self.isizes[2], self.isizes[3]),
            # nn.SiLU(),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(self.isizes[3], self.proj_dim),
            nn.Sigmoid(),
        )

        # self.criterion = torch.nn.MSELoss()
        self.criterion = torch.nn.L1Loss()
        self.set_optim(lr)

        self.__init_weights()
    
    def set_optim(self, lr=0.001):
        self.optim = optim.Adam(self.mlp.parameters(), lr=lr)

    def forward(self, x):
        return self.mlp(x)

    def __init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                module.bias.data.zero_()

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

        self.mlp.eval()
        with torch.no_grad():
            pred = self.mlp(Xtr.cuda()).detach().cpu().numpy()
        # pred = self.predict(X_embedded)
        return pred, X_embedded, acc_loss


    # def save_to_folder(self):

    def fit_val(self, Pred_train, X_train, Pred_val, X_val, epochs=300, batch_size=128):

        tr_loss = []
        va_loss = []

        Xtr = torch.tensor(X_train).float()
        Ptr = torch.tensor(Pred_train).float()
        Xva = torch.tensor(X_val).float()
        Pva = torch.tensor(Pred_val).float()

        train_ds = torch.utils.data.TensorDataset(Ptr, Xtr)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_ds = torch.utils.data.TensorDataset(Pva, Xva)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True)

        self.mlp.cuda()

        for epoch in range(epochs):  # loop over the dataset multiple times
        
            self.mlp.train()
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
                
            tr_loss.append(running_loss/(i+1))
            print(f"Epoch {epoch}/{epochs}", flush=True, end='\r')
        

            with torch.no_grad():
                self.mlp.eval()
                running_loss = 0.0
                for i, data in enumerate(val_dl, 0):
                    inputs, labels = data
                    inputs = inputs.cuda()
                    labels = labels.cuda()
            
                    outputs = self.forward(inputs)
                    
                    loss = self.criterion(outputs, labels)
            
                    # print statistics
                    running_loss += loss.item()
                    
                va_loss.append(running_loss/(i+1))

        print('Finished Training                 ')


        pred = self.predict(Pred_train)
        return pred, tr_loss, va_loss
