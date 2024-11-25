from moduledag import Module, ImageModule, CombinationModule, FunctionalToClass
import torch.nn as nn
import torch


class SimpleConv(ImageModule):
    shape_transform = (1, 1, 1)

    def __init__(self, name):
        super().__init__(name, self.shape_transform)

    def init_block(self):
        channels = int(self.realShape[1])
        self.block = nn.Conv2d(
            int(channels), int(channels), kernel_size=3, stride=1, padding=1
        )


class Conv211(ImageModule):
    shape_transform = (2, 1, 1)

    def __init__(self, name):
        super().__init__(name, self.shape_transform)

    def init_block(self):
        channels = int(self.realShape[1])
        self.block = nn.Conv2d(
            channels, channels * 2, kernel_size=3, stride=1, padding=1
        )


class ConvBack(ImageModule):
    shape_transform = (0.5, 1, 1)

    def __init__(self, name):
        super().__init__(name, self.shape_transform)

    def init_block(self):
        channels = int(self.realShape[1])
        self.block = nn.Conv2d(
            channels, channels // 2, kernel_size=3, stride=1, padding=1
        )


class ConvBackUpsample(ImageModule):
    shape_transform = (0.5, 2, 2)

    def __init__(self, name):
        super().__init__(name, self.shape_transform)

    def init_block(self):
        channels = int(self.realShape[1])
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2),
        )


class ConvDown(ImageModule):
    shape_transform = (1, 0.5, 0.5)

    def __init__(self, name):
        super().__init__(name, self.shape_transform)

    def init_block(self):
        channels = int(self.realShape[1])
        self.block = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)


class ConvTranspose(ImageModule):
    shape_transform = (1, 2, 2)

    def __init__(self, name):
        super().__init__(name, self.shape_transform)

    def init_block(self):
        channels = int(self.realShape[1])
        self.block = nn.ConvTranspose2d(
            channels, channels, kernel_size=4, stride=2, padding=1, output_padding=0
        )


class ConvSandwich(ImageModule):
    shape_transform = (1, 1, 1)

    def __init__(self, name):
        super().__init__(name, self.shape_transform)

    def init_block(self):
        channels = int(self.realShape[1])
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )


class ConvSandwichQuadruple(ImageModule):
    shape_transform = (1, 1, 1)

    def __init__(self, name):
        super().__init__(name, self.shape_transform)

    def init_block(self):
        channels = int(self.realShape[1])
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )


class DeeperConv(ImageModule):
    shape_transform = (2, 0.5, 0.5)

    def __init__(self, name):
        super().__init__(name, self.shape_transform)

    def init_block(self):
        channels = int(self.realShape[1])
        self.block = nn.Conv2d(
            channels, channels * 2, kernel_size=4, stride=2, padding=1
        )


class Upsample(ImageModule):
    shape_transform = (1, 2, 2)

    def __init__(self, name):
        super().__init__(name, self.shape_transform)
        self.block = nn.Upsample(scale_factor=2, mode="nearest")


class MaxPool(ImageModule):
    shape_transform = (1, 0.5, 0.5)

    def __init__(self, name):
        super().__init__(name, self.shape_transform)
        self.block = nn.MaxPool2d(2)


class ReLU(ImageModule):
    shape_transform = (1, 1, 1)

    def __init__(self, name):
        super().__init__(name, self.shape_transform)
        self.block = nn.ReLU()


class LeakyReLU(ImageModule):
    shape_transform = (1, 1, 1)

    def __init__(self, name):
        super().__init__(name, self.shape_transform)
        self.block = nn.LeakyReLU()


class Tanh(ImageModule):
    shape_transform = (1, 1, 1)

    def __init__(self, name):
        super().__init__(name, self.shape_transform)
        self.block = nn.Tanh()


class Sigmoid(ImageModule):
    shape_transform = (1, 1, 1)

    def __init__(self, name):
        super().__init__(name, self.shape_transform)
        self.block = nn.Sigmoid()


class SiLU(ImageModule):
    shape_transform = (1, 1, 1)

    def __init__(self, name):
        super().__init__(name, self.shape_transform)
        self.block = nn.SiLU()


class ELU(ImageModule):
    shape_transform = (1, 1, 1)

    def __init__(self, name):
        super().__init__(name, self.shape_transform)
        self.block = nn.ELU()


class GELU(ImageModule):
    shape_transform = (1, 1, 1)

    def __init__(self, name):
        super().__init__(name, self.shape_transform)
        self.block = nn.GELU()


class SELU(ImageModule):
    shape_transform = (1, 1, 1)

    def __init__(self, name):
        super().__init__(name, self.shape_transform)
        self.block = nn.SELU()


class Dropout(ImageModule):
    shape_transform = (1, 1, 1)

    def __init__(self, name):
        super().__init__(name, self.shape_transform)
        self.block = nn.Dropout()


# class BatchNorm(ImageModule):
#     shape_transform = (1, 1, 1)

#     def __init__(self, name):
#         super().__init__(name, self.shape_transform)

#     def init_block(self):
#         self.block = nn.BatchNorm2d(self.realShape[0])


class SummationModule(CombinationModule):
    shape_transform = "comb"

    def __init__(self, name):
        super().__init__(name)

    def init_block(self):
        self.block = FunctionalToClass(lambda tensors: torch.stack(tensors).sum(dim=0))
