import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
#import pyreadr
devicenum = 1
device=torch.device('cuda:'+str(devicenum) if torch.cuda.is_available() else 'cpu')

## Data loading
parser = argparse.ArgumentParser() 
parser.add_argument('--wdir',type=str,default="/home/luoqinhuan/MDN/data/hapmap3/data_simulation/")
parser.add_argument('--cell_data_name',type=str,default="data2CE_total.csv")
parser.add_argument('--gene_data_name',type=str,default="data2GE_total.csv")
parser.add_argument('--cell_model_name',type=str,default="CellEmbedding.pkl")
parser.add_argument('--gene_model_name',type=str,default="GeneEmbedding.pkl")
parser.add_argument('--out_cell_name',type=str,default="CellEmbedding.csv")
parser.add_argument('--out_gene_name',type=str,default="GeneEmbedding.csv")
parser.add_argument('--var_dim', type=int, default=0)
parser.add_argument('--split_size', type=int, default=1000)
args = parser.parse_args()
## Cell Embedding Data loading
cell_path = os.path.join(args.wdir, args.cell_data_name)
gene_path = os.path.join(args.wdir, args.gene_data_name)
data = pd.read_csv(cell_path,header= 0,index_col=0)
#data = list(data.items())[0][1]
print("Data loaded!")
data = pd.DataFrame(data).transpose()
## DataLoader
#labels = pd.read_csv("/home/luoqinhuan/Denoise/Pancreas/label.csv",index_col=0)

#data_train, data_test, labels_train, labels_test = train_test_split(data,labels,test_size=0.3, random_state=42)
data_train, data_test = train_test_split(data, test_size=0.3, random_state=42)
#labels_train.to_csv('/home/luoqinhuan/pretrained_data/labels_train.txt',index=0,header=0)
#labels_test.to_csv('/home/luoqinhuan/pretrained_data/labels_test.txt',index=0,header=0)
data_train = np.array(data_train)
data_test = np.array(data_test)

data_train = data_train.astype(float)
data_test = data_test.astype(float)
data_train = torch.from_numpy(data_train)
data_test = torch.from_numpy(data_test)
train_dataset = TensorDataset(data_train)
test_dataset = TensorDataset(data_test)
train_data = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_data = DataLoader(dataset=test_dataset, batch_size=256)

# Encoder and decoder use the DC-GAN architecture
class Encoder(torch.nn.Module):
    def __init__(self, input_dim, z_dim):
        super(Encoder, self).__init__()
        self.model = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1024),
            nn.Dropout(0.5), 
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 128),
            nn.Dropout(0.5), 
            torch.nn.LeakyReLU(),
            #torch.nn.Linear(512, 128),
            #torch.nn.LeakyReLU(),
            torch.nn.Linear(128, z_dim)
        ])
        
    def forward(self, x):
        #print("Encoder")
        #print(x.size())
        for layer in self.model:
            x = layer(x)
            #print(x.size())
        return x
      
class Decoder(torch.nn.Module):
    def __init__(self,input_dim, z_dim):
        super(Decoder, self).__init__()
        self.model = torch.nn.ModuleList([
            torch.nn.Linear((z_dim + args.var_dim), 128),
            #nn.Dropout(0.5), 
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1024),
            nn.Dropout(0.5), 
            torch.nn.ReLU(),
            #torch.nn.Linear(512, 2048),
            #torch.nn.ReLU(),
            torch.nn.Linear(1024,input_dim),
            nn.Dropout(0.5), 
            torch.nn.ReLU()
        ])
        
    def forward(self, x):
        #print("Decoder")
        #print(x.size())
        for layer in self.model:
            x = layer(x)
            #print(x.size())
        return x
       
def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

def l1_regularization(model, l1_alpha):
    l1_loss = []
    for module in model.modules():
        if type(module) is torch.nn.BatchNorm2d:
            l1_loss.append(torch.abs(module.weight).sum())
    return l1_alpha * sum(l1_loss)

class Model(torch.nn.Module):
    def __init__(self,input_dim,z_dim):
        super(Model, self).__init__()
        self.encoder = Encoder(input_dim,z_dim)
        self.decoder = Decoder(input_dim,z_dim)
        
    def forward(self, x,batch_info):
        z = self.encoder(x)
        z_con = torch.cat([z, batch_info], dim=1)
        x_reconstructed = self.decoder(z_con)
        return z, x_reconstructed
    
# Ensemble model 
class Ensemble(nn.Module):
    def __init__(self, model, n_models):
        super(Ensemble, self).__init__()
        self.models = nn.ModuleList([model for _ in range(n_models)])
        
    def forward(self, x,batch_info):
        zs, x_recons = [], []
        for model in self.models:
            z, x_recon = model(x,batch_info)
            zs.append(z)
            x_recons.append(x_recon)
            

        z = torch.mean(torch.stack(zs), dim=0)
        x_recon = torch.mean(torch.stack(x_recons), dim=0)
        return  z,x_recon
    
# Augmentations function
def add_gaussian_noise(x, mean=0., var=0.1):
    noise = torch.randn(x.size()) * var + mean
    return x + noise
def drop_genes(x, p=0.1):  
    mask = torch.rand(x.shape[1]) > p  
    return x[:, mask]  
def drop_cells(x, p=0.1):  
    mask = torch.rand(x.shape[0]) > p  
    return x[mask, :] 

class GAN(nn.Module):
    def __init__(self,z_dim):
        super(GAN, self).__init__()
        
        self.fc1 = nn.Linear(z_dim, 4196)
        self.fc2 = nn.Linear(4196, 512)
        self.fc3 = nn.Linear(512, 1)     
        
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))
def train_ensemble_AddNoise(
    dataloader,
    input_dim,
    var_dim,
    z_dim=2,
    n_epochs=10,
    print_every=100,
    batch_size = 64,
    lambda_l1 = 0.1,
    #mmd_anneal_rate = 1.08,
    #mmd_weight = 1,
    beta = 20,
    gamma = 10,
    n_models = 10,
    noise_rate = 0.2
):
    epoch_list=[]
    loss_list=[]
    model = Model(input_dim,z_dim).to(device)
    ensemble = Ensemble(model, n_models)
    gan = GAN(z_dim).to(device)
    bce_loss = nn.BCEWithLogitsLoss()
    #if use_cuda:
        #model = model.cuda()
    #print(model)
    optimizer = torch.optim.Adam(ensemble.parameters())
    gan_opt = torch.optim.Adam(gan.parameters(), lr=1e-3)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.95)
    i = -1
    for epoch in range(n_epochs):
        loss_record = 0
        #mmd_weight = mmd_weight * mmd_anneal_rate  
        for label,data in enumerate(dataloader):
            i += 1
            optimizer.zero_grad()
            data = data[0]
            batch_info = data[:,(data.shape[1] - args.var_dim):data.shape[1]]
            data = data[:,:(data.shape[1] - args.var_dim)]
            x = data.to(torch.float32)
            batch_info = batch_info.to(torch.float32)
            if torch.rand(1) < noise_rate:  
                x= add_gaussian_noise(x)
            true_samples = Variable(
                torch.randn(batch_size, z_dim),
                requires_grad=False
            )
            true_samples = F.softmax(true_samples)
            z, x_reconstructed = ensemble(x.to(device),batch_info.to(device))
            adv_loss = bce_loss(gan(z), torch.zeros_like(gan(z))) * gamma
            mmd = compute_mmd(true_samples.to(device), z) * beta
            nll = (x_reconstructed - x.to(device)).pow(2).mean()
            loss = nll + mmd + l1_regularization(model,lambda_l1) + adv_loss 
            loss_record = loss_record + loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
            ## Training GAN
            gan_opt.zero_grad() 
            gan_loss = bce_loss(gan(true_samples.to(device)), torch.ones_like(gan(true_samples.to(device)))) + bce_loss(gan(z.detach()), torch.zeros_like(gan(z)))
            gan_loss.backward() 
            gan_opt.step() 
            if i % print_every == 0:
                print("Negative log likelihood is {:.5f}, mmd loss is {:.5f}".format(
                    nll.data.item(), mmd.data.item()))
        loss_list.append(loss_record)
        epoch_list.append(epoch)
        print(epoch)
    return ensemble

z_dim = 32
model = train_ensemble_AddNoise(train_data,input_dim = (data.shape[1] - args.var_dim) ,var_dim = args.var_dim,z_dim=z_dim, n_epochs=100,batch_size = 256,lambda_l1 = 1e-2, beta=0.5,noise_rate = 0,n_models= 10,gamma=0.1)

torch.save(model,os.path.join(args.wdir, args.cell_model_name))
data = np.array(data)
data = torch.from_numpy(data)
data = data.to(torch.float32)
data_splits = torch.split(data, args.split_size, dim=0)  
embeddings = []
model.eval
for split in data_splits:
    X = split.to(torch.float32).to(device)
    batch_info = X[:,(data.shape[1] - args.var_dim):data.shape[1]]
    x = X[:,:(data.shape[1] - args.var_dim)]
    x = x.to(torch.float32)
    batch_info = batch_info.to(torch.float32)
    z, x_reconstructed = model(x.to(device),batch_info.to(device))
    embeddings.append(z.cpu().detach())
      
embeddings = torch.cat(embeddings, dim=0)
np.savetxt(os.path.join(args.wdir, args.out_cell_name),embeddings.cpu().detach().numpy(),fmt='%.2f',delimiter=',')



## Gene Embedding Model
class Encoder(torch.nn.Module):
    def __init__(self, input_dim, z_dim):
        super(Encoder, self).__init__()
        self.model = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 2048),
            nn.Dropout(0.5), 
            torch.nn.LeakyReLU(),
            torch.nn.Linear(2048, 256),
            nn.Dropout(0.5), 
            torch.nn.LeakyReLU(),
            #torch.nn.Linear(512, 128),
            #nn.Dropout(0.2),
            #torch.nn.LeakyReLU(),
            torch.nn.Linear(256, z_dim)
        ])
        
    def forward(self, x):
        #print("Encoder")
        #print(x.size())
        for layer in self.model:
            x = layer(x)
            #print(x.size())
        return x
    
    
class Decoder(torch.nn.Module):
    def __init__(self,input_dim, z_dim):
        super(Decoder, self).__init__()
        self.model = torch.nn.ModuleList([
            torch.nn.Linear(z_dim, 256),
            nn.Dropout(0.5), 
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2048),
            nn.Dropout(0.5), 
            torch.nn.ReLU(),
            #torch.nn.Linear(128, 256),
            #nn.Dropout(0.2), 
            #torch.nn.ReLU(),
            torch.nn.Linear(2048,input_dim),
            nn.Dropout(0.5), 
            torch.nn.ReLU()
        ])
        
    def forward(self, x):
        #print("Decoder")
        #print(x.size())
        for layer in self.model:
            x = layer(x)
            #print(x.size())
        return x
    

class Model(torch.nn.Module):
    def __init__(self,input_dim,z_dim):
        super(Model, self).__init__()
        self.encoder = Encoder(input_dim,z_dim)
        self.decoder = Decoder(input_dim,z_dim)
        
    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return z, x_reconstructed
    
# Ensemble model 
class Ensemble(nn.Module):
    def __init__(self, model, n_models):
        super(Ensemble, self).__init__()
        self.models = nn.ModuleList([model for _ in range(n_models)])
        
    def forward(self, x):
        zs, x_recons = [], []
        for model in self.models:
            z, x_recon = model(x)
            zs.append(z)
            x_recons.append(x_recon)
            

        z = torch.mean(torch.stack(zs), dim=0)
        x_recon = torch.mean(torch.stack(x_recons), dim=0)
        
        return  z,x_recon

class GAN(nn.Module):
    def __init__(self,z_dim):
        super(GAN, self).__init__()
        
        self.fc1 = nn.Linear(z_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)     
        
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))
    
def train_ensemble_AddNoise(
    dataloader,
    input_dim,
    z_dim=2,
    n_epochs=10,
    print_every=100,
    batch_size = 64,
    lambda_l1 = 0.1,
    #mmd_anneal_rate = 1.08,
    #mmd_weight = 1,
    beta = 20,
    n_models = 10,
    noise_rate = 0.2,
    gamma = 1
):
    epoch_list=[]
    loss_list=[]
    model = Model(input_dim,z_dim).to(device)
    ensemble = Ensemble(model, n_models)
    gan = GAN(z_dim).to(device)
    #if use_cuda:
        #model = model.cuda()
    #print(model)
    optimizer = torch.optim.Adam(ensemble.parameters())
    bce_loss = nn.BCEWithLogitsLoss()
    gan_opt = torch.optim.Adam(gan.parameters(), lr=1e-3)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.95)
    i = -1
    for epoch in range(n_epochs):
        loss_record = 0
        #mmd_weight = mmd_weight * mmd_anneal_rate  
        for label,data in enumerate(dataloader):
            i += 1
            optimizer.zero_grad()
            data = data[0]
            x = data.to(torch.float32)
            if torch.rand(1) < noise_rate:  
                x= add_gaussian_noise(x)
            z, x_reconstructed = ensemble(x.to(device))
            true_samples = Variable(
                torch.randn(batch_size, z_dim),
                requires_grad=False
            )
            true_samples = F.softmax(true_samples)
            adv_loss = bce_loss(gan(z), torch.zeros_like(gan(z))) * gamma
            mmd = compute_mmd(true_samples.to(device), z) * beta
            nll = (x_reconstructed - x.to(device)).pow(2).mean()
            loss = nll + mmd + l1_regularization(model,lambda_l1) + adv_loss 
            loss_record = loss_record + loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
            ## Training GAN
            gan_opt.zero_grad() 
            gan_loss = bce_loss(gan(true_samples.to(device)), torch.ones_like(gan(true_samples.to(device)))) + bce_loss(gan(z.detach()), torch.zeros_like(gan(z)))
            gan_loss.backward() 
            gan_opt.step() 
            if i % print_every == 0:
                print("Negative log likelihood is {:.5f}, mmd loss is {:.5f}".format(
                    nll.data.item(), mmd.data.item()))
        loss_list.append(loss_record)
        epoch_list.append(epoch)
        #print(epoch)
    #plt.savefig("/home/luoqinhuan/pretrained_data/training_result_50_L32_10x_3layers_L1_01_b10_E10.png")
    return ensemble

## Gene Embedding Data loading
data = pd.read_csv(gene_path,header= 0,index_col=0)
#data = list(data.items())[0][1]
print("Data loaded!")
data = pd.DataFrame(data)
## DataLoader
data_train, data_test = train_test_split(data,test_size=0.1, random_state=42)
#labels_train.to_csv('/home/luoqinhuan/pretrained_data/labels_train.txt',index=0,header=0)
#labels_test.to_csv('/home/luoqinhuan/pretrained_data/labels_test.txt',index=0,header=0)
data_train = np.array(data_train)
data_test = np.array(data_test)

data_train = data_train.astype(float)
data_test = data_test.astype(float)
data_train = torch.from_numpy(data_train)
data_test = torch.from_numpy(data_test)
train_dataset = TensorDataset(data_train)
test_dataset = TensorDataset(data_test)
train_data = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_data = DataLoader(dataset=test_dataset, batch_size=256)

z_dim = 20
model = train_ensemble_AddNoise(train_data,input_dim = data.shape[1] ,z_dim=z_dim, n_epochs=50,batch_size = 256,lambda_l1 = 1e-2,beta= 2,noise_rate = 0.5,n_models= 10,gamma=2)
#torch.save(model,"/home/luoqinhuan/Denoise/CellEmbedding_L32_e30_L1_1000_b5_AN02_E5.pkl")
data = np.array(data)
data = torch.from_numpy(data)
data = data.to(torch.float32)
data_splits = torch.split(data, args.split_size, dim=0)  
embeddings = []
model.eval
for split in data_splits:
    X = split.to(torch.float32).to(device)
    z,x = model(X)
    embeddings.append(z.cpu().detach())
      
embeddings = torch.cat(embeddings, dim=0)
torch.save(model,os.path.join(args.wdir, args.gene_model_name))
np.savetxt(os.path.join(args.wdir, args.out_gene_name),embeddings.numpy(),fmt='%.2f',delimiter=',')
