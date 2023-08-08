# -*- coding: utf-8 -*-
"""
Adapted from
https://github.com/raoofnaushad/Land-Cover-Classification-using-Sentinel-2-Dataset/blob/master/Land_Cover_Classification_using_Sentinel_2_Satellite_Imagery_and_Deep_Learning.ipynb
"""

"""
Libraries
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from tqdm.notebook import tqdm
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn import model_selection

import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler   
from sklearn.metrics import mean_squared_error, r2_score
from torchvision.transforms import transforms
from torchvision.utils import make_grid
import torch.nn.functional as F
import math

#for repeatable results
random_seed = 1
np.random.seed(random_seed) #for repeatable
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

"""
Config
"""
ROOT_PATH = 'ImagePath'
BASE_PATH = os.path.join(ROOT_PATH, 'Images_Suburban')

"""
Pre-Processing data folders to train and test
"""
path_cv = r"path_label.csv"
DATA_DF = pd.read_csv(path_cv)

n=1000 #the first 1000 samples of the route Jardim Oceânico.
TRAIN_DF= DATA_DF.iloc[:n]

n=374#the last 374 samples of the route Jardim Oceânico.
TEST_DF= DATA_DF.tail(n)

y_train = TRAIN_DF.iloc[:, [8]]  #PL
y_test = TEST_DF.iloc[:, [8]] #PL

min_data=np.min(y_train)
max_data=np.max(y_train)

scaler= MinMaxScaler()
y_train = np.array(y_train)
y_test =  np.array(y_test)

#normalize output
y_train = y_train.reshape(-1, 1)
y_train = scaler.fit_transform(y_train)
y_test = y_test.reshape(-1, 1)
y_test = scaler.transform(y_test)

# convert output variable to float
y_train, y_test = y_train.astype(float), y_test.astype(float),

df_y_train = pd.DataFrame(y_train)
df_y_test = pd.DataFrame(y_test)

df_test = pd.concat([df_y_train,df_y_test])
df_test = df_test.to_numpy()

DATA_DF['label']=df_test

n=1000 #the first 1000 samples of the route Jardim Oceânico.
TRAIN_DF= DATA_DF.iloc[:n]

n=374#the last 374 samples of the route Jardim Oceânico.
TEST_DF= DATA_DF.tail(n)

"""
Procesing data folders to train and test
"""
NUM_CLASSES=1
VALID_SIZE=0.2

TRAIN_DF = TRAIN_DF.sample(frac = 1, random_state=0) 
TRAIN_DF = TRAIN_DF[:-int(len(TRAIN_DF)*VALID_SIZE)]
VALID_DF = TRAIN_DF[-int(len(TRAIN_DF)*VALID_SIZE) :]

TRAIN_DF.reset_index(inplace = True)  
TRAIN_DF.head()

VALID_DF.reset_index(inplace = True) 
VALID_DF.head()

TEST_DF.reset_index(inplace = True) 
TEST_DF.head()

TRAIN_DF.size, VALID_DF.size, TEST_DF.size

"""
Creating Dataset and Dataloaders
"""
class EuroSAT(Dataset):
    def __init__(self, train_df, train_dir, transform=None):
        self.train_dir = train_dir
        self.train_df = train_df
        self.transform = transform
        
    def __len__(self):
        return len(self.train_df)
    
    def __getitem__(self, idx):
        row = self.train_df.loc[idx]
        img_id, label = row['image_id'], row['label']
        img = Image.open(os.path.join(self.train_dir, '', img_id))
        if self.transform:
            img = self.transform(img)
        return img, label


"""
Transformations and Datasets
"""
## Dataset and transformations
data_transform = transforms.Compose([
                                transforms.Resize(size=(224, 224)),                
                                transforms.ToTensor(),
                                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
train_ds = EuroSAT(TRAIN_DF, BASE_PATH, data_transform)
valid_ds = EuroSAT(VALID_DF, BASE_PATH, data_transform)
test_ds = EuroSAT(TEST_DF, BASE_PATH, data_transform)
print(len(train_ds), len(valid_ds), len(test_ds))
    
## Data loaders and showing batch of data
batch_size=15 
train_dl= DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_dl= DataLoader(valid_ds, batch_size, shuffle=False, num_workers=0, pin_memory=True)
batch_size=374
test_dl = DataLoader(test_ds, batch_size, shuffle=False, num_workers=0, pin_memory=True)

"""
Model
"""
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim = 1)
    return torch.tensor(torch.sum(preds==labels).item() / len(preds))

def RMSELoss(outputs, labels):
    return torch.sqrt(torch.mean((outputs-labels)**2))

def RMSELoss_desnormalized(outputs, labels,ab):
    
    min_data=50.953507
    max_data=78.564554

    outputs= min_data +(max_data-min_data)*outputs
    labels= min_data +(max_data-min_data)*labels

    MAE = torch.mean(np.abs(labels.detach().cpu() - outputs.detach().cpu()))
    MAPE= torch.mean(np.abs((labels.detach().cpu() -outputs.detach().cpu())/labels.detach().cpu()))*100
    R2 = r2_score(labels.detach().cpu(),outputs.detach().cpu())
    R2=torch.from_numpy(np.asarray(R2))
    
    #SD 
    n = len(outputs.detach().cpu())
    sum_model=0
    
    abs_dif = np.abs(labels.detach().cpu() -outputs.detach().cpu())
    mean_model = torch.mean(abs_dif)
    
    for x in abs_dif:
        t = (x - mean_model) ** 2
        sum_model += t 
    
    SD = math.sqrt(sum_model/(n))
    SD=torch.from_numpy(np.asarray(SD))
       
    y_target = labels.detach().cpu().numpy()
    y_pred = outputs.detach().cpu().numpy()
    
    return torch.sqrt(torch.mean((outputs-labels)**2)), R2,MAE,MAPE,SD

class MulticlassClassifierBase(nn.Module):
    
    def training_step(self, batch):
        img, label = batch
        out = self(img)
        label=label.unsqueeze(-1) 
        out=out.double()
        loss = criterion(out, label)
        accu = RMSELoss(out, label)
        ab=0
        rmse_desn,r2_desn,mae,mape,sd = RMSELoss_desnormalized(out, label,ab)
        return accu ,loss, rmse_desn, r2_desn,mae,mape,sd
    
    def validation_step(self, batch):
        img, label = batch
        out = self(img)
        label=label.unsqueeze(-1)
        loss = criterion(out, label)
        accu = RMSELoss(out, label)
        ab=1
        rmse_desn,r2_desn,mae,mape,sd = RMSELoss_desnormalized(out, label,ab)
        return {"val_loss": loss.detach(), "val_acc": accu, "val_rmse_desn":rmse_desn,"val_r2_desn":r2_desn,"val_mae_desn":mae,"val_mape_desn":mape,"val_sd_desn":sd}
    
    def validation_epoch_ends(self, outputs):
        batch_loss = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [x['val_acc'] for x in outputs]
        batch_rmse_desn = [x['val_rmse_desn'] for x in outputs]
        batch_r2_desn = [x['val_r2_desn'] for x in outputs] 
        batch_mae_desn = [x['val_mae_desn'] for x in outputs] 
        batch_mape_desn = [x['val_mape_desn'] for x in outputs] 
        batch_sd_desn = [x['val_sd_desn'] for x in outputs] 
        epoch_acc = torch.stack(batch_acc).mean()
        epoch_rmse_desn=torch.stack(batch_rmse_desn).mean()
        epoch_r2_desn=torch.stack(batch_r2_desn).mean()  
        epoch_mae_desn=torch.stack(batch_mae_desn).mean()  
        epoch_mape_desn=torch.stack(batch_mape_desn).mean() 
        epoch_sd_desn=torch.stack(batch_sd_desn).mean() 
        return {"val_loss":epoch_loss.item(), "val_acc":epoch_acc.item(),"val_rmse_desn":epoch_rmse_desn.item(),"val_r2_desn":epoch_r2_desn.item(),"val_mae_desn":epoch_mae_desn.item(),"val_mape_desn":epoch_mape_desn.item(),"val_sd_desn":epoch_sd_desn.item()}

        
    def epoch_end(self, epoch, result):
        print("Epoch [{}],train_accu: {:.4f},train_rmse_desn: {:.4f},train_r2_desn: {:.4f},train_mae_desn: {:.4f},train_mape_desn: {:.4f},train_sd_desn: {:.4f},learning_rate: {:.4f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f},val_rmse_desn:{:.4f},val_r2_desn:{:.4f},val_mae_desn:{:.4f},val_mape_desn:{:.4f},val_sd_desn:{:.4f}".format(
            epoch,result['train_accu'], result['train_rmse_desn'],result['train_r2_desn'],result['train_mae_desn'],result['train_mape_desn'],result['train_sd_desn'],result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc'], result['val_rmse_desn'],result['val_r2_desn'],result['val_mae_desn'],result['val_mape_desn'],result['val_sd_desn']))


NUM_CLASSES=45 

class LULC_Model(MulticlassClassifierBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnet18(pretrained=True)
        n_inputs = self.network.fc.in_features
        self.network.fc = nn.Sequential(
                              nn.Linear(n_inputs, 4096),
                              nn.ReLU(),
                              nn.Dropout(0.5),
                              nn.Linear(4096, 4096),
                              nn.ReLU(),
                              nn.Dropout(0.5),
                              nn.Linear(4096, 4096),
                              nn.ReLU(),
                              nn.Dropout(0.5),
                              nn.Linear(4096, NUM_CLASSES),
                              nn.LogSoftmax(dim=1)
                                )
    def forward(self, xb):
        return self.network(xb)
    
    def freeze(self):
        for param in self.network.parameters():
            param.require_grad=False
        for param in self.network.fc.parameters():
            param.require_grad=True
    def unfreeze(self):
        for param in self.network.parameters():
            param.require_grad=True

model = LULC_Model()

#Loading the trained model
model.load_state_dict(torch.load("Resnet18_NPUW.pt"))

model_test=list(model.children())[-1] #to extract the model ResNet from LULC model

class LULC_Model(MulticlassClassifierBase):
    def __init__(self):
        super().__init__()
        self.network = model_test
      
        self.network.fc = nn.Sequential(
                              nn.Linear(512, 256),
                              nn.ReLU(),   
                              nn.Dropout(0.2),
                              nn.Linear(256, 1),
                                )
    def forward(self, xb):
        return self.network(xb)
    
    def freeze(self):
        for param in self.network.parameters():
            param.require_grad=False
        for param in self.network.fc.parameters():
            param.require_grad=True
    def unfreeze(self):
        for param in self.network.parameters():
            param.require_grad=True


#Call the final model with the last layer prepared to be a regressor
model = LULC_Model()

def try_batch(dl):
    for images, labels in dl:  
        print(images.shape)
        out = model(images)
        print(out.dtype)
        out=out.double()
        print(out[0])
        break
try_batch(train_dl)


"""
Training and Evaluating
"""

@torch.no_grad()
def evaluate(model, valid_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in valid_loader]
    return model.validation_epoch_ends(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def fit(epochs, max_lr,  model, train_loader, valid_loader, test_loader, weight_decay=0,
                grad_clip=None,opt_func=torch.optim.Adam, max_epochs_stop=3):
  
    history = []
    valid_loss_min = np.Inf
    valid_acc_max = 0
    model_file_name = 'lulc.pth'
    model_file_name2 = 'lulc_max_acc.pth'
    epochs_no_improve =  0
    optimizer = opt_func(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.00001)
                         
    for epoch in range(epochs):
        
        model.train()
        train_loss = []
        train_accu = []
        train_rmse_desn = []
        train_r2_desn = []  
        train_mae_desn = [] 
        train_mape_desn = [] 
        train_sd_desn = [] 
        lrs = []
        
        for batch in tqdm(train_loader):
            
            accu, loss, rmse_desn,r2_desn,mae_desn,mape_desn,sd_desn = model.training_step(batch)

            train_loss.append(loss)
            
            train_accu.append(accu)
            
            train_rmse_desn.append(rmse_desn)
            
            train_r2_desn.append(r2_desn)
            train_mae_desn.append(mae_desn)
            train_mape_desn.append(mape_desn)
            train_sd_desn.append(sd_desn)
       
            loss.backward()

            ## Gradient Clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
            
            
        result = evaluate(model, valid_loader)
        scheduler.step(result['val_loss'])
        
        ########### Early Stopping ##############                                         
        valid_loss = result['val_loss']
        valid_acc = result['val_acc']
        if valid_acc > valid_acc_max:
            torch.save(model.state_dict(), model_file_name2)
            valid_acc_max = valid_acc
        if valid_loss<valid_loss_min:
            torch.save(model.state_dict(), model_file_name)
            valid_loss_min = valid_loss                                  
            epochs_no_improve = 0          
        else:
            epochs_no_improve += 1
            if epochs_no_improve > max_epochs_stop:
                result["train_loss"] = torch.stack(train_loss).mean().item()
                result["train_accu"] = torch.stack(train_accu).mean().item()
                result["train_rmse_desn"] = torch.stack(train_rmse_desn).mean().item()
                result["train_r2_desn"] = torch.stack(train_r2_desn).mean().item() 
                result["train_mae_desn"] = torch.stack(train_mae_desn).mean().item()
                result["train_mape_desn"] = torch.stack(train_mape_desn).mean().item()
                result["train_sd_desn"] = torch.stack(train_sd_desn).mean().item()
                result["lrs"] = lrs
                model.epoch_end(epoch, result)
                history.append(result)
                print("Early Stopping............................")
                return history                                
                                                 
        result["train_loss"] = torch.stack(train_loss).mean().item()
        result["train_accu"] = torch.stack(train_accu).mean().item()
        result["train_rmse_desn"] = torch.stack(train_rmse_desn).mean().item()
        result["train_r2_desn"] = torch.stack(train_r2_desn).mean().item() 
        result["train_mae_desn"] = torch.stack(train_mae_desn).mean().item()
        result["train_mape_desn"] = torch.stack(train_mape_desn).mean().item()
        result["train_sd_desn"] = torch.stack(train_sd_desn).mean().item()
        result["lrs"] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    print("VAL LOSS MIN {}".format(valid_loss_min))
    print("VAL ACC MAX {}".format(valid_acc_max))
    return history


"""
Initializing Device also Loading Data and Model to device
"""
def get_device():
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
            
    def __len__(self):
        return len(self.dl)


device = get_device()
train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)
test_dl = DeviceDataLoader(test_dl, device)
model = to_device(model, device)
try_batch(train_dl)

torch.cuda.empty_cache()

"""
Training
"""
max_epochs_stop = 10
max_lr = 1e-3
grad_clip = 0.05
weight_decay = 1e-3
criterion = nn.MSELoss()
epochs = 19
opt_func = torch.optim.Adam

## Freezing except last layer
model.freeze()

## Training
history = fit(epochs, max_lr, model, train_dl, valid_dl, test_dl, weight_decay, grad_clip, opt_func, max_epochs_stop)

#%%
### CNN testing with unseen data ###
result = evaluate(model, test_dl)
print(result)
