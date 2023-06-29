import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class ResNet(nn.Module):
	def __init__(self, input_channel):
		super(ResNet, self).__init__()

		self.inplanes = input_channel 
		self.inplanes2 = 128
		self.relu = nn.ReLU(inplace=True)

		self.layer1 = nn.Conv2d(self.inplanes, self.inplanes2, kernel_size=1, stride=1, bias=False)
		self.norm1 = nn.BatchNorm2d(self.inplanes2, affine=True)
		nn.init.xavier_uniform_(self.layer1.weight)

		self.resblocks = nn.ModuleList()
		self.ks = 3
		for dilv in [1,1,2,1,1,4,1,1,8,1,1,16,1,1,32,1,1,64,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]:
			layer = nn.Conv2d(in_channels=self.inplanes2, out_channels=self.inplanes2, kernel_size=self.ks, dilation=dilv, padding=int(dilv*(self.ks-1)/2))
			nn.init.xavier_uniform_(layer.weight, gain=sqrt(2.0))
			self.resblocks.append(layer)
			self.resblocks.append(nn.BatchNorm2d(self.inplanes2, affine=True))
			layer = nn.Conv2d(in_channels=self.inplanes2, out_channels=self.inplanes2, kernel_size=self.ks, dilation=dilv, padding=int(dilv*(self.ks-1)/2))
			nn.init.xavier_uniform_(layer.weight, gain=sqrt(2.0))
			self.resblocks.append(layer)
			self.resblocks.append(nn.BatchNorm2d(self.inplanes2, affine=True))		

		
		self.layer2 = nn.Conv2d(self.inplanes2, 1, 1, bias=False)
		self.norm2 = nn.BatchNorm2d(1, affine=True)
		nn.init.xavier_uniform_(self.layer2.weight)

	def forward(self, x1):
		x = self.layer1(x1)
		x = F.relu(self.norm1(x))

		for i in range(int(len(self.resblocks)/4)):
			residual = x
			x = self.resblocks[i*4](x)
			x = F.relu(self.resblocks[i*4+1](x))
			x = self.resblocks[i*4+2](x)
			x = self.resblocks[i*4+3](x)
			x += residual
			x = F.relu(x)
		
		# last layer
		x = self.layer2(x)
		x = self.norm2(x)
		return x
