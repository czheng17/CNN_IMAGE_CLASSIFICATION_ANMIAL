import torch
import torchvision
import torch.utils.data as Data
import torch.nn.functional
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image

NUM_CLASS = 2
IMG_W = 208
IMG_H = 208
image_dir = '/home/czheng17/Documents/Projects/SpyderProjects/cat_and_dog/test_image/cat.5.jpeg'

class RESTORE_CNN(torch.nn.Module):
    def __init__(self):
        super(RESTORE_CNN, self).__init__()
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


restore_cnn = RESTORE_CNN()
restore_cnn.load_state_dict(torch.load('net_params.pkl'))




def get_one_image(test_image_dir):
    # image = cv2.imread(test_image_dir)
    image = Image.open(test_image_dir)
    plt.show(image)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.CenterCrop( (IMG_W,IMG_H) ),
            torchvision.transforms.ToTensor(),
        ]
    )
    tensor = transform(image)
    return tensor

image_tensor = get_one_image(image_dir)
# print image_tensor.size()
image_tensor = Variable(image_tensor.unsqueeze(0))
result = restore_cnn(image_tensor)

pred_y = torch.max(result, 1)[1].data.numpy().squeeze()
# print pred_y
if pred_y == 1:
    print('this is a dog.')
else:
    print('this is a cat.')