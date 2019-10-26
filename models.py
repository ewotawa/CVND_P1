## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

# maxpool layer
max_k = 2
max_s = 2

# convolutional layer 1
W_1 = 224 # length of one side of image; per resized shape from notebook 1
F_1 = 5 # length of one side of kernel
S_1 = 1 # default stride
Out_feat_1 = 32
Out_dim_1 = (W_1 - F_1) / (S_1) + 1
# print('Output size before max pool:\t(', Out_feat_1, ', ', Out_dim_1, ', ', Out_dim_1, ')')
# print('Output size after max pool:\t(', Out_feat_1, ', ', (Out_dim_1 / max_s), ', ', (Out_dim_1 / max_s), ')')

# convolutional layer 2
W_2 = Out_dim_1 / max_s
F_2 = 3 # length of one side of kernel
S_2 = 1 # default stride
Out_feat_2 = 128
Out_dim_2 = (W_2 - F_2) / (S_2) + 1
# print('Output size before max pool:\t(', Out_feat_2, ', ', Out_dim_2, ', ', Out_dim_2, ')')
# print('Output size after max pool:\t(', Out_feat_2, ', ', (Out_dim_2 / max_s), ', ', (Out_dim_2 / max_s), ')')

# convolutional layer 3
W_3 = Out_dim_2 / max_s
F_3 = 4 # length of one side of kernel
S_3 = 1 # default stride
Out_feat_3 = 512
Out_dim_3 = (W_3 - F_3) / (S_3) + 1
# print('Output size before max pool:\t(', Out_feat_2, ', ', Out_dim_2, ', ', Out_dim_2, ')')
# print('Output size after max pool:\t(', Out_feat_2, ', ', (Out_dim_2 / max_s), ', ', (Out_dim_2 / max_s), ')')

# convolutional layer 4
W_4 = Out_dim_3 / max_s
F_4 = 5 # length of one side of kernel
S_4 = 1 # default stride
Out_feat_4 = 2056
Out_dim_4 = (W_4 - F_4) / (S_4) + 1
# print('Output size before max pool:\t(', Out_feat_2, ', ', Out_dim_2, ', ', Out_dim_2, ')')
# print('Output size after max pool:\t(', Out_feat_2, ', ', (Out_dim_2 / max_s), ', ', (Out_dim_2 / max_s), ')')

# convolutional layer 4
W_5 = Out_dim_4 / max_s
F_5 = 3 # length of one side of kernel
S_5 = 1 # default stride
Out_feat_5 = 4112
Out_dim_5 = (W_5 - F_5) / (S_5) + 1
# print('Output size before max pool:\t(', Out_feat_2, ', ', Out_dim_2, ', ', Out_dim_2, ')')
# print('Output size after max pool:\t(', Out_feat_2, ', ', (Out_dim_2 / max_s), ', ', (Out_dim_2 / max_s), ')')

# fully connected layer 1
length = Out_dim_5 / max_s
# Round down to neared integer
if np.round(length, decimals=0) > length:
    length = np.round(length, 0) - 1
else:
    length = np.round(length, 0)
# print('Input shape for first fully connected layer:\t', Out_feat_2, ' * ', length, ' * ', length)
fc_1_in = Out_feat_5 * length * length
fc_1_out = 2000

# fully connected layer 2
fc_2_in = fc_1_out
fc_2_out = 3000

# fully connectd layer 3
fc_3_in = fc_2_out
fc_3_out = 68*2 # number of features required for prediction


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.conv1 = nn.Conv2d(1, Out_feat_1, F_1)
        self.conv2 = nn.Conv2d(Out_feat_1, Out_feat_2, F_2)
        self.conv3 = nn.Conv2d(Out_feat_2, Out_feat_3, F_3)
        self.conv4 = nn.Conv2d(Out_feat_3, Out_feat_4, F_4)
        self.conv5 = nn.Conv2d(Out_feat_4, Out_feat_5, F_5)
        
        self.pool = nn.MaxPool2d(max_k, max_s)

        self.fc1 = nn.Linear(int(fc_1_in), int(fc_1_out))
        self.fc2 = nn.Linear(int(fc_2_in), int(fc_2_out))
        self.fc3 = nn.Linear(int(fc_3_in), int(fc_3_out))
        
        self.fc1_drop = nn.Dropout(p=0.4)
        self.fc2_drop = nn.Dropout(p=0.2)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # two activated conv layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        
        # flatten
        x = x.view(x.size(0), -1)

        # two linear layers with dropout
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)
                
        # a modified x, having gone through all the layers of your model, should be returned
        return x
