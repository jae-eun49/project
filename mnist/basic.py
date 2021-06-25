#!/usr/bin/env python
# coding: utf-8

# # Import & Configuration

# In[1]:


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import warnings
# print option
torch.set_printoptions(precision=4, linewidth=50000, sci_mode=None)
# Control Warning Message
warnings.filterwarnings(action='ignore')


# # Preparing Data

# In[2]:


datasetDir = '~/dataset'
trainData = datasets.MNIST(root=datasetDir, train=True,  download=True, transform=ToTensor())
testData  = datasets.MNIST(root=datasetDir, train=False, download=True, transform=ToTensor())
trainData


# In[3]:


print('[Print] Type of trainData :', type(trainData))
print('[Print] Type of trainData.data :', type(trainData.data))
print('[Print] Type of trainData.targets :', type(trainData.targets))
print('[Print] Type of trainData.classes :',type(trainData.classes))


# In[4]:


print(trainData.targets)
print(trainData.classes)


# In[5]:


print(trainData.data[0])


# In[6]:


print(trainData.targets[0])


# In[7]:


# Difference between trainData and trainData.data
# Difference between trainData is nomalized value of trainData.data
torch.set_printoptions(precision=1, linewidth=50000, sci_mode=None)
print(trainData[0])
torch.set_printoptions(precision=4, linewidth=50000, sci_mode=None)


# In[8]:


# [0][0]: 3D-image data [0][1]: Label
print(f'Image: {trainData[0][0].size()}')
print(f'Label: {trainData[0][1]}')


# In[9]:


# trainData is nomalized using /255
print(f'trainData.data: {trainData.data[0][24][4]}')
print(f'trainData.data/255: {trainData.data[0][24][4]/255}')
print(f'trainData: {trainData[0][0][0][24][4]}')


# # Data Loader

# In[10]:


trainDataset, validDataset = torch.utils.data.random_split(trainData, [50000, 10000])
print(f'trainDataset: {len(trainDataset)}')
print(f'validDataset: {len(validDataset)}')

batchSize = 64

trainDataLoader = DataLoader(trainDataset, batch_size=batchSize)
ValidDataLoader = DataLoader(validDataset, batch_size=batchSize)
testDataLoader  = DataLoader(testData,  batch_size=batchSize)

for imgs, labs in trainDataLoader:
	print("Shape of image [N, C, H, W]: ", imgs.shape)
	print("Shape of label             : ", labs.shape, labs.dtype)
	break


# In[11]:


figure = plt.figure(figsize=(16, 16))
cols, rows = 8, 8
for i in range(cols * rows ):
	#sampleIndex = torch.randint(len(imgs), size=(1,)).item()
# 	img = imgs[sampleIndex]
# 	lab = labs[sampleIndex].item()

	img = imgs[i]
	lab = labs[i].item()
	figure.add_subplot(rows, cols, i+1)
	plt.title(lab)
	plt.axis("off")
	plt.imshow(img.squeeze(), cmap="gray")
plt.show()


# # Model Build-up

# In[12]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class NN(nn.Module):
	def __init__(self):
		super(NN, self).__init__()
		self.flatten = nn.Flatten()
		self.seq = nn.Sequential(
			nn.Linear(28*28, 30),
			nn.ReLU(),
			nn.Linear(30, 30),
			nn.ReLU(),
			nn.Linear(30, 10),
			nn.ReLU(),
			nn.Linear(10, 10),
			nn.ReLU()
		)

	def forward(self, x):
		x = self.flatten(x)
		logits = self.seq(x)
		return logits

model = NN().to(device)
print(model)


# In[13]:


for name, param in model.named_parameters(): 
	print(f'name:{name}') 
	print(type(param)) 
	print(f'param.shape:{param.shape}') 
	print(f'param.requries_grad:{param.requires_grad}') 
	print('=====')


# # Define Loss Function & Optimizer

# In[14]:


lossFunc = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)


# In[15]:


def train(trainDataLoader, model, lossFunc, optimizer):
	size = len(trainDataLoader.dataset)
	for loopInEpoch, (img, lab) in enumerate(trainDataLoader):
		img, lab = img.to(device), lab.to(device)
		pred = model(img)
		loss = lossFunc(pred, lab)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
#       select multiline -> 'ctrl + /' -> multiline comment
# 		if batchIndex % 10 == 0:
# 			loss, current = loss.item(), batchIndex * len(img)
# 			print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")		


# In[16]:


def test(testDataLoader, model):
	size = len(testDataLoader.dataset)
	model.eval()
	testLoss, correct = 0, 0
	with torch.no_grad():
		for img, lab in testDataLoader:
			img, lab = img.to(device), lab.to(device)
			pred = model(img)
			testLoss += lossFunc(pred, lab).item()
			correct += (pred.argmax(1) == lab).type(torch.float).sum().item()
	testLoss /= size
	correct /= size
	print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {testLoss:>8f}")


# In[17]:


model.state_dict().keys()


# In[18]:


print(model.state_dict()['seq.0.weight'][0][:8])
print(model.state_dict()['seq.2.weight'][0][:8])
print(model.state_dict()['seq.4.weight'][0][:8])
print(model.state_dict()['seq.6.weight'][0][:8])


# # Please check in the https://www.h-schmidt.net/FloatConverter/IEEE754.html

# In[19]:


model.state_dict()['seq.0.weight'][0][0]=0.3
print(model.state_dict()['seq.0.weight'][0][:8])
print(model.state_dict()['seq.2.weight'][0][:8])
print(model.state_dict()['seq.4.weight'][0][:8])
print(model.state_dict()['seq.6.weight'][0][:8])


# In[20]:


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1:3d}: ", end='')
    train(trainDataLoader, model, lossFunc, optimizer)
    test(testDataLoader, model)
print("Done!")


# # Check updated weight in the 1st layer

# In[21]:


torch.save(model.state_dict(), "model.pth")


# In[22]:


print(model.state_dict()['seq.0.weight'][0][:8])
print(model.state_dict()['seq.2.weight'][0][:8])
print(model.state_dict()['seq.4.weight'][0][:8])
print(model.state_dict()['seq.6.weight'][0][:8])


# In[23]:


model = NN().to(device)
model.load_state_dict(torch.load("model.pth"))


# In[24]:


epochs = 2
for t in range(epochs):
    print(f"Epoch {t+1:3d}: ", end='')
    test(testDataLoader, model)
print("Done!")


# In[25]:


print(model.state_dict()['seq.0.weight'][0][:8])
print(model.state_dict()['seq.2.weight'][0][:8])
print(model.state_dict()['seq.4.weight'][0][:8])
print(model.state_dict()['seq.6.weight'][0][:8])


# In[26]:


model.state_dict()['seq.6.weight']


# In[ ]:




