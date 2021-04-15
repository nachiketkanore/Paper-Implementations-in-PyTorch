# paper : http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf

import torch
import torch.nn as nn

# LeNet architecture
# 1 x 32 x 32 input
# (5 x 5), s = 1, p = 0
# average pooling s = 2, p = 0
# (5 x 5), s = 1, p = 0
# average pooling s = 2, p = 0
# Conv 5 x 5 to 120 channels 
# Linear 120 
# Linear 10

class LeNet(nn.Module):

	def __init__(self):
		super(LeNet, self).__init__()

		self.relu = nn.ReLU()
		self.pool = nn.AvgPool2d(kernel_size = (2, 2), stride = (2, 2))
		self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = (5, 5), stride = (1, 1), padding = (0, 0))
		self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = (5, 5), stride = (1, 1), padding = (0, 0))
		self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = (5, 5), stride = (1, 1), padding = (0, 0))
		self.linear1 = nn.Linear(120, 84)
		self.linear2 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.pool(x)
		x = self.relu(self.conv2(x))
		x = self.pool(x)
		x = self.relu(self.conv3(x)) # [mini_batch_size][120][1][1] --> [mini_batch_size][120]
		# At this point we have images having 120 channels and having dimensions 1 x 1 literally
		# Now we flatten all cells
		x = x.reshape(x.shape[0], -1)
		x = self.relu(self.linear1(x))
		x = self.linear2(x)
		return x

x = torch.randn(64, 1, 32, 32)
model = LeNet()
print(model(x).shape)
# Expected shape = [64][10], 64 mini_batch_size and each has 10 classes distribution