import torch
import torchvision
import torch.utils.data as Data
import torch.nn.functional
from torch.autograd import Variable
import numpy as np
# import cv2

TRAIN_DIR = './data/train'
NUM_CLASS = 2
BATCH_SIZE = 100
LR = 0.0001
IMG_W = 208
IMG_H = 208
EPOCH = 50

'''
transform training image data to tensor
'''
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.CenterCrop( (IMG_W,IMG_H) ),
        torchvision.transforms.ToTensor(),  # (0-255)-->(0-1)
        # torchvision.transforms.ToPILImage(), # change to picture, not useful
    ]
)

'''
handle dataset
'''
dataset = torchvision.datasets.ImageFolder(
    root=TRAIN_DIR,  # the most important thing is that the root dir must have label sub root dir!!!!!
                     # i debug this place for a whole afternoon
    transform=transform,
)
# print dataset[0]
# print dataset.classes  #['0', '1'] --> the directory name. So it is the reason above.

'''
build data loader to batch size and shuffle
'''
data_loader = Data.DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

'''
build up the model
'''
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=2,
            ),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=2,
            )
        )
        self.fc1 = torch.nn.Linear(
            in_features=32*52*52,
            out_features=128,
        )
        self.fc2 = torch.nn.Linear(
            in_features=128,
            out_features=128,
        )
        self.out = torch.nn.Linear(
            in_features=128,
            out_features=NUM_CLASS,
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.out(x)
        # x = torch.nn.Softmax(x)
        return x

'''
instance cnn and
loss function and optimizer
'''
cnn = CNN()
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)

for epoch in range(EPOCH):
    for step, (x,y) in enumerate(data_loader):
        t_x = Variable(x)
        t_y = Variable(y)

        output = cnn(t_x)
        loss = loss_func(output,t_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.save(cnn.state_dict(), 'net_params.pkl')

        if step%50 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])