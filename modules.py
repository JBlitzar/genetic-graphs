from moduledag import Module, ImageModule
import torch.nn as nn


class SingleConv(ImageModule):
    def __init__(self, name):
        super().__init__(name, (1,1,1)) # Channels,x,y stay the same.

    def init_block(self):
        channels = self.realShape[0]
        self.block = nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1)

class ConvSandwich(ImageModule):
    def __init__(self, name):
        super().__init__(name, (1,1,1)) # Channels,x,y stay the same.

    def init_block(self):
        channels = self.realShape[0]
        self.block = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
        )

class ConvSandwichQuadruple(ImageModule):
    def __init__(self, name):
        super().__init__(name, (1,1,1)) # Channels,x,y stay the same.

    def init_block(self):
        channels = self.realShape[0]
        self.block = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
        )


class DeeperConv(ImageModule):
    def __init__(self, name):
        super().__init__(name, (2,0.5,0.5)) # deeper, smaller

    def init_block(self):
        channels = self.realShape[0]
        self.block = nn.Conv2d(channels,channels * 2,kernel_size=4,stride=2,padding=1)

class Upsample(ImageModule):
    def __init__(self, name):
        super().__init__(name, (1,2,2)) # x,y *= 2

    
        self.block = nn.Upsample(scale_factor=2,mode="nearest")



class MaxPool(ImageModule):
    def __init__(self, name):
        super().__init__(name, (1,0.5,0.5)) # x,y /= 2

    
        self.block = nn.MaxPool2d(2)


class ReLU(ImageModule):
    def __init__(self, name):
        super().__init__(name, (1,1,1)) # Channels,x,y stay the same.

    
        self.block = nn.ReLU()

class Tanh(ImageModule):
    def __init__(self, name):
        super().__init__(name, (1,1,1)) # Channels,x,y stay the same.

    
        self.block = nn.Tanh()

class Sigmoid(ImageModule):
    def __init__(self, name):
        super().__init__(name, (1,1,1)) # Channels,x,y stay the same.

    
        self.block = nn.Sigmoid()

class SiLU(ImageModule):
    def __init__(self, name):
        super().__init__(name, (1,1,1)) # Channels,x,y stay the same.

    
        self.block = nn.SiLU()