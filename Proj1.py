# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 16:53:42 2020

@author: Mikatsuki
"""
from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils import data
import matplotlib.pyplot as plt

np.random.seed(seed=int(time.time()))
# task1
in_dim = 4
X_a = np.random.normal(loc=[1,1,1,1], scale=[0.5,0.5,0.5,0.5], size=(10000, in_dim))
X_b = np.random.normal(loc=[0,0,0,0], scale=[2,2,2,2], size=(10000, in_dim))

y_a = np.ones((len(X_a),1))
y_b = np.zeros((len(X_b),1))

X = np.concatenate((X_a, X_b), axis=0)
label = np.concatenate((y_a, y_b), axis=0)

idx = np.arange(len(X))
np.random.shuffle(idx)
X = X [idx]
label = label [idx]

    
class func_model(nn.Module):
    
    def __init__(self, in_dim):
        super(func_model, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU(inplace = True)
        return
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        out = self.fc4(x)
        return torch.sigmoid(out)

model = func_model(in_dim)
out = model(torch.from_numpy(X).float())
print(out.shape)
# task1 end
        
# task2
def custom_loss(outputs, target,x):
    loss = nn.BCELoss()
    output = loss(outputs, target)+torch.mean(np.abs(x-torch.tensor([1,1,1,1])))
    return output
print(custom_loss(out, torch.from_numpy(label).float(), torch.from_numpy(X)))
# task2 end

# task3
class ndataset(data.Dataset):
    def __init__(self, X, label):
        self.X = torch.from_numpy(X).float()
        self.label = torch.from_numpy(label).float()
        return
    
    def __getitem__(self, index):
        x = self.X[index]
        y = self.label[index]
        return x,y
    
    def __len__(self):
        return len(self.X)

def get_dataloader(X, label, batch_size, test_size=0.2):
    # shuffle data
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X [idx]
    label = label [idx]
    
    # divide data for train and validation
    nb_train = int((1-test_size)*len(X))
    trainset = ndataset(X[:nb_train], label[:nb_train])
    testset = ndataset(X[:nb_train], label[:nb_train])
    
    dataloaders = {
        'train':data.DataLoader(trainset, batch_size=batch_size, shuffle=True),
        'val':data.DataLoader(testset, batch_size=batch_size, shuffle=True)
        }
    return dataloaders

def train_model(model, dataloaders, optimizer, num_epochs, device):
    since = time.time()
    
    history = {'loss':[], 'val_loss':[], 'acc':[], 'val_acc':[]}
    
    model = model.to(device)
    step_show = 10
    for epoch in range(num_epochs):
        if epoch%step_show==0:
            print('Epoch {0}/{1}'.format(epoch, num_epochs-1))
    
        model.train()
        running_loss, running_correct = 0.0, 0.0
        for i, (inputs, target) in enumerate(dataloaders['train']):
        
            # apply to device on inputs and target
            inputs = inputs.to(device).float()
            target = target.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = custom_loss(outputs, target, inputs)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()*inputs.size(0)
            running_correct += torch.sum((outputs+0.5).int().t()==target.data.int().t())
            
        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_corrects = running_correct.double() / len(dataloaders['train'].dataset)
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_corrects)
        
        model.eval()
        val_running_loss, val_running_correccts = 0.0, 0.0
        for i, (inputs, target) in enumerate(dataloaders['val']):
            inputs = inputs.to(device).float()
            target = target.to(device).float()
            outputs = model(inputs)
            loss = custom_loss(outputs, target, inputs)
            
            val_running_loss += loss.item()*inputs.size(0)
            val_running_correccts += torch.sum((outputs+0.5).int().t()==target.data.int().t())
        
        val_epoch_loss = val_running_loss/len(dataloaders['val'].dataset)
        val_epoch_corrects = val_running_correccts.double()/len(dataloaders['val'].dataset)
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_corrects)
        
        if epoch%step_show==0:
            print('Epoch Loss:{0:.6f}, Acc:{1:.6f}, Val Acc:{2:.6f}'
                  .format(epoch_loss, epoch_corrects, val_epoch_corrects))
            print('-'*10)
    time_elapsed = time.time() - since
    print('Training ccomplete in {:.0f}min {:.0f}.s'.format(time_elapsed//60, time_elapsed%60))
    
    return model, history

batch_size = 256
num_epochs = 50
test_size = 0.2

device = torch.device("cpu")

model = func_model(in_dim)
dataloaders = get_dataloader(X, label, batch_size, test_size)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.001,
                             betas=(0.9, 0.999),
                             eps=1e-8,
                             weight_decay=1e-5)

model, history = train_model(model=model,
                             dataloaders=dataloaders,
                             optimizer=optimizer,
                             num_epochs=num_epochs,
                             device=device)
# save paarameter tensor
torch.save(model.state_dict(), 'fcn_model.pt')

# task3 end

# task4
test_model = func_model(in_dim)
test_model.load_state_dict(torch.load('fcn_model.pt'))

np.random.seed(seed=int(time.time()))
a_test = np.random.normal(loc=[1,1,1,1], scale=[0.5,0.5,0.5,0.5], size=(2000, in_dim))
b_test = np.random.normal(loc=[0,0,0,0], scale=[2,2,2,2], size=(2000, in_dim))
test_model.eval()
a_out = test_model(torch.from_numpy(a_test).float()).detach().numpy()
b_out = test_model(torch.from_numpy(b_test).float()).detach().numpy()

bins = 50
prg = (0,1)

plt.figure(figsize=(5,5))
plt.hist(a_out, bins=bins, range=prg, histtype='step', label='a', density=True)
plt.hist(b_out, bins=bins, range=prg, histtype='step', label='b', density=True)
plt.show()
# task4 end

# task5
def separation_pro(output_data):
    n=np.zeros(bins, int)
    step=1/bins
    for a in output_data:
        for i in range(bins):
            if a>=i*step and a<(i+1)*step:
                n[i] += 1
    return n

def separation(a,b):
    a_n=separation_pro(a)
    b_n=separation_pro(b)
    sep=0
    for i in range(bins):
        u = (a_n[i]-b_n[i])
        d = a_n[i]+b_n[i]
        sep += int(u**2)/(2*int(d))
    return sep/bins

print("the separation of test sample is: "+repr(separation(a_out, b_out)))