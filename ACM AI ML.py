#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORT LIBRARIES
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#DOWNLOADING THE DATA SET
x, y = torch.load('C:\\files\\MNIST\\MNIST\\processed\\training.pt')


# In[3]:


y.shape #TOTAL ELEMENTS


# In[4]:


#SAMPLE IMAGE
plt.imshow(x[2].numpy())
plt.title(f'Number is {y[2].numpy()}')
plt.colorbar()
plt.show()


# In[5]:


#THE ONE HOT ENCODER
y_original = torch.tensor([2, 4, 3, 0, 1])
y_new = F.one_hot(y_original)


# In[6]:


y_original


# In[7]:


y_new


# In[8]:


#TRYING ON DATASET
y


# In[9]:


y_new


# In[10]:


y_new = F.one_hot(y, num_classes=10)
y_new.shape


# In[11]:


x.shape


# In[12]:


x.view(-1,28**2).shape


# In[13]:


y_new = F.one_hot(y, num_classes=10)
y_new.shape


# In[14]:


#SOME ISSUES WITH IMAGE
x.shape


# In[15]:


#The images are currently 28x28, but we want to turn the images (the xs) into a vector (which will be length ). We can do this using the .view property of a tensor.
#x.view(-1,28**2).shape


# In[16]:


#PYTORCH DATASET OBJECT

class CTDataset(Dataset):
    def __init__(self, filepath):
        self.x, self.y = torch.load(filepath)
        self.x = self.x / 255.
        self.y = F.one_hot(self.y, num_classes=10).to(float)
    def __len__(self): 
        return self.x.shape[0]
    def __getitem__(self, ix): 
        return self.x[ix], self.y[ix]


# In[17]:


#DOWNLOADING TEST AND TRAINING DATA SET.
train_ds = CTDataset('C:\\files\\MNIST\\MNIST\\processed\\training.pt')
test_ds = CTDataset('C:\\files\\MNIST\\MNIST\\processed\\test.pt')


# In[18]:


#Datasets of a __len__ and __getitem__ method, so they can be used with python functionality


len(train_ds)


# In[19]:


#SLICING
xs, ys = train_ds[0:4]


# In[20]:


ys.shape


# In[21]:


#DataLoader Object
#put the Dataset objects inside a DataLoader class
train_dl = DataLoader(train_ds, batch_size=5)


# In[22]:


for x, y in train_dl:
    print(x.shape)
    print(y.shape)
    break


# In[23]:


#batch_size here is 5, and there are 60000 images,LENGTH WILL BE 12000
len(train_dl)


# In[24]:


#CROSS ENTROPY LOSS
L = nn.CrossEntropyLoss()


# In[25]:


#THE NETWORK
class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(28**2,100)
        self.Matrix2 = nn.Linear(100,50)
        self.Matrix3 = nn.Linear(50,10)
        self.R = nn.ReLU()
    def forward(self,x):
        x = x.view(-1,28**2)
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.Matrix3(x)
        return x.squeeze()


     


# In[26]:


f = MyNeuralNet()


# In[27]:


#Network predictions (before optimization)
xs.shape


# In[28]:


f(xs)


# In[29]:


ys


# In[30]:


#loss between such predictions.
L(f(xs), ys)


# In[31]:


#TRAINING

def train_model(dl, f, n_epochs=20):
    # Optimization
    opt = SGD(f.parameters(), lr=0.01)
    L = nn.CrossEntropyLoss()

    # Train model
    losses = []
    epochs = []
    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        N = len(dl)
        for i, (x, y) in enumerate(dl):
            # Update the weights of the network
            opt.zero_grad() 
            loss_value = L(f(x), y) 
            loss_value.backward() 
            opt.step() 
            # Store training data
            epochs.append(epoch+i/N)
            losses.append(loss_value.item())
    return np.array(epochs), np.array(losses)



# In[32]:


#CALLING THE FUNCTION TO TRAIN THE MODEL
epoch_data, loss_data = train_model(train_dl, f)


# In[34]:


#Since there are 20 total epochs, SPLIT ARRAY INTO 20 EQUAL FUNCTIONS
epoch_data_avgd = epoch_data.reshape(20,-1).mean(axis=1)
loss_data_avgd = loss_data.reshape(20,-1).mean(axis=1)


# In[35]:


#SAMPLE IMAGE Y
y_sample = train_ds[0][1]
y_sample


# In[36]:


#PREDICTION OF Y
x_sample = train_ds[0][0]
yhat_sample = f(x_sample)
yhat_sample


# In[37]:


#taking the index of the maximum value, TO GET IMAGE
torch.argmax(yhat_sample)


# In[38]:


plt.imshow(x_sample)


# In[39]:


xs, ys = train_ds[0:2000]
yhats = f(xs).argmax(axis=1)


# In[40]:


#TRAINING DATASET
fig, ax = plt.subplots(10,4,figsize=(10,15))
for i in range(40):
    plt.subplot(10,4,i+1)
    plt.imshow(xs[i])
    plt.title(f'Predicted Digit: {yhats[i]}')
fig.tight_layout()
plt.show()


# In[41]:


#TESTING DATASET
xs, ys = test_ds[:2000]
yhats = f(xs).argmax(axis=1)


# In[42]:


#TOTAL 40 PREDICTIONS
fig, ax = plt.subplots(10,4,figsize=(10,15))
for i in range(40):
    plt.subplot(10,4,i+1)
    plt.imshow(xs[i])
    plt.title(f'Predicted Digit: {yhats[i]}')
fig.tight_layout()
plt.show()


# In[43]:


#..................#


# In[ ]:


#For above classification I first unknowingly tried using CUDA as one of the optimizer to help to run, but it failed
#as it wasnt installed in my computer.
#Above classification can alos be made using keras and tensorflow libraries but keras is slower and is usefull for small
#datasets.
#Even tried using sigmoid as the activation function.


# In[ ]:


#OVER FITTING

#Over fitting takes place when our model becomes very good in classfing
#the training dataset, but fails to classify our testing dataset or even validation dataset.
#Here in any model if validation matrix are considerbaly worse than the training matrix , than its indicated that
#our model is overfittied.
#Concepts of overfitting boils down to the fact that it has learned the featues of training set extremly well that
#it cant classify any slightly variance in any our training set or in any other dataset.
#Overfitting can be reduced by giving lots of data to our model so that it could be familiar to many variances
#before going to testing data,  ...even data agumentation is one of the way to neglect overfitting.


# In[ ]:


#Model converging.

#Its basically a moment which is below overfitting and above underfitting; means during model convergence, change
#in learning rate becomes lower, therefore our optimizer need not have to alter the weights of different epoch,
#therefore any additional training will not improve the model in further interations.
#Its global minima of our cost function wrt the gradient decent plot. Therefore by applying simple calculus we can predict 
#when our model convergies.

