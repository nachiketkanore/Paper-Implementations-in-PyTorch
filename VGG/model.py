# paper : https://arxiv.org/abs/1409.1556

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# You can easily change architecture as VGG16, VGG19,.. just by changing this list
VGG_architecture = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
# VGG[i] = number of channels in i-th layer if it is integer else performing maxpooling

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device : ', device)

class VGG(nn.Module):
	def __init__(self, in_channels, num_classes = 1000):
		super(VGG, self).__init__()
		self.in_channels = in_channels
		self.conv_layer = self.get_conv_layers(VGG_architecture)
		self.fc_layer = nn.Sequential(
				nn.Linear(512*7*7, 4096),
				nn.ReLU(),
				nn.Dropout(p = 0.5),
				nn.Linear(4096, 4096),
				nn.ReLU(),
				nn.Dropout(p = 0.5),
				nn.Linear(4096, num_classes)
			)

	def forward(self, x):
		x = self.conv_layer(x)
		x = x.reshape(x.shape[0], -1)
		x = self.fc_layer(x)
		return x

	def get_conv_layers(self, architecture):
		layers = []
		in_channels = self.in_channels

		for x in architecture:
			if type(x) == int:
				out_channels = x
				add_layers = [
					nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
					nn.BatchNorm2d(x),
					nn.ReLU()
				]
				layers += add_layers
				in_channels = x
			elif x == 'M':
				add_layers = [
					nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
				]
				layers += add_layers

		return nn.Sequential(*layers)

model = VGG(in_channels = 3, num_classes = 1000)
x = torch.randn(2, 3, 224, 224) # 2 images each having 3 channels and dimensions = 224 x 224
print(model(x).shape)
# expected shape of output from model is [2][1000]