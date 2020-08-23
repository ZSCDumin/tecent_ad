import torch, os, gc
import pandas as pd
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import pickle as p
import torch.optim as optim
import joblib
from sklearn.model_selection import train_test_split

#定义模型
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
       
        # kernel
        self.word_embeddings = nn.Embedding(4450000, 64)
        #self.word_embeddings.weights = nn.Parameter(weights, requires_grad=False)
        
        self.conv1 = nn.Conv2d(1, 256, (2,64)) 
        self.conv2 = nn.Conv2d(1, 256, (3,64))
        self.conv3 = nn.Conv2d(1, 256, (4,64))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256*3, 2) #10age,2gender

    def conv_run(self, input, conv_layer):
        conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)# maxpool_out.size() = (batch_size, out_channels)

        return max_out
    
    def forward(self, x):
        # 池化层
        b_size = x.size(0) #获取batch大小
        input = self.word_embeddings(x)
        
        max_out1 = self.conv_run(input, self.conv1)
        max_out2 = self.conv_run(input, self.conv2)
        max_out3 = self.conv_run(input, self.conv3)

        all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        fc_in = self.dropout(all_out)
        #x = x.view(b_size, -1) #
        
        output = self.fc1(fc_in)

        return output

def weights_init(m):                                               
    classname = m.__class__.__name__                               
    if classname.find('Conv') != -1:                               
        #nn.init.normal_(m.weight.data, 0.0, 0.02)                 
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:                        
        nn.init.normal_(m.weight.data, 1.0, 0.02)                  
        nn.init.constant_(m.bias.data, 0)                          

def train(model, device, train_loader, optimizer, epoch):
    criterion = nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.squeeze())
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%100 == 0: 
            print('\rTrain Epoch: {} [ ({:.0f}%)], Loss: {:.6f}'.format(
                epoch, 100. * batch_idx / len(train_loader), loss.item()),end="")        
  
  #测试
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target.squeeze()).item() # 将损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(' Average loss: {:.4f}, Accuracy: ({:.3f}%)'.format(
        test_loss, 100. * correct / len(test_loader.dataset)))
    
    
if __name__ == '__main__':
    BATCH_SIZE = 512
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    EPOCHS = 50
	data = joblib.load('data_byClick.pkl')
	train_label = joblib.load('train_label_tensor.pkl')
	test_label = joblib.load('test_label_tensor.pkl')
	lst = []
	for i in data:
		if len(i) < 30:
			temp = np.append(np.array(i),np.zeros(30-len(i)))
			lst.append(temp)
		else :
			lst.append(i[0:30])
	data = torch.tensor(lst,dtype=torch.long)
	label = torch.cat((train_label,test_label),0)
	label = label//10
	train_data, test_data, train_label, test_label = train_test_split(
    	data, label, test_size=0.20, random_state=42)
	train_data =train_data.unsqueeze(1)
	test_data = test_data.unsqueeze(1)
	train_dataset = Data.TensorDataset(train_data,train_label)
	train_loader = Data.DataLoader(
		dataset=train_dataset,  # torch TensorDataset format
		batch_size=BATCH_SIZE,  
		shuffle=True,  # 是否打乱
		num_workers=4,  # 多线程
	)

	test_dataset = Data.TensorDataset(test_data,test_label)
	test_loader = Data.DataLoader(
		dataset=test_dataset,
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=4, 
	)
	net = Net()
	net.apply(weights_init)
	net.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(),lr=1e-4)
	for epoch in range(1, EPOCHS + 1):
		train(net, device, train_loader, optimizer, epoch)
		print('\n test set:',end="")
		test(net, device, test_loader)
		print('train set:',end="")
		test(net, device, train_loader)
		
	torch.save(net.state_dict(), r'***.pkl')

