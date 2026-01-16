import copy
import torch
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121

from utils.operations import MixedOp

ranges = {
    "densenet": ((0, 3), (4, 6), (6, 8), (8, 10), (10, 12)),
}

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.ranges = ranges["densenet"]
        self.features = densenet121(weights=DenseNet121_Weights.DEFAULT).features
        self.conv2d = nn.Conv2d(1024, 1024, kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        output = {}
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d" % (idx + 1)] = x
        output["x5"] = self.conv2d(x)
        return output


class SuperNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = DenseNet()
        self.deconv1 = MixedOp(1024, 512)
        self.deconv2 = MixedOp(512, 256)
        self.deconv3 = MixedOp(256, 128)
        self.deconv4 = MixedOp(128, 64)
        self.deconv5 = MixedOp(64, 32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)

        x5 = output["x5"]  # size=(N, 512, x.H/32, x.W/32)
        x4 = output["x4"]  # size=(N, 512, x.H/16, x.W/16)
        x3 = output["x3"]  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output["x2"]  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output["x1"]  # size=(N, 64, x.H/2,  x.W/2)

        score = self.deconv1(x5)  # size=(N, 512, x.H/16, x.W/16)
        score = score + x4  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.deconv2(score)  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.deconv3(score)  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2  # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.deconv4(score)  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1  # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.deconv5(score)  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)

    def clip_alphas(self):
        self.deconv1.clip_alphas()
        self.deconv2.clip_alphas()
        self.deconv3.clip_alphas()
        self.deconv4.clip_alphas()
        self.deconv5.clip_alphas()

    def get_alphas(self):
        # return by list
        alpha_list = [
            self.deconv1.alphas.cpu().detach().numpy().tolist(),
            self.deconv2.alphas.cpu().detach().numpy().tolist(),
            self.deconv3.alphas.cpu().detach().numpy().tolist(),
            self.deconv4.alphas.cpu().detach().numpy().tolist(),
            self.deconv5.alphas.cpu().detach().numpy().tolist(),
        ]
        return alpha_list

    def get_expected_flops(self, input_size=128):
        """
        Calculate expected FLOPs of decoder based on alpha weights (differentiable).
        Only calculates decoder FLOPs (encoder is fixed).
        """
        # Output sizes for each deconv layer (input_size -> /32 -> upsample)
        H = input_size // 32  # After encoder: 128 -> 4
        sizes = [
            (H * 2, H * 2),      # deconv1: 4 -> 8
            (H * 4, H * 4),      # deconv2: 8 -> 16
            (H * 8, H * 8),      # deconv3: 16 -> 32
            (H * 16, H * 16),    # deconv4: 32 -> 64
            (H * 32, H * 32),    # deconv5: 64 -> 128
        ]

        total_flops = 0
        total_flops += self.deconv1.get_expected_flops(sizes[0][0], sizes[0][1])
        total_flops += self.deconv2.get_expected_flops(sizes[1][0], sizes[1][1])
        total_flops += self.deconv3.get_expected_flops(sizes[2][0], sizes[2][1])
        total_flops += self.deconv4.get_expected_flops(sizes[3][0], sizes[3][1])
        total_flops += self.deconv5.get_expected_flops(sizes[4][0], sizes[4][1])

        return total_flops / 1e9  # Return in GFLOPs


class OptimizedNetwork(nn.Module):
    def __init__(self, super_net):
        super(OptimizedNetwork, self).__init__()

        if isinstance(super_net, torch.nn.DataParallel):
            module = super_net.module
        else:
            module = super_net

        # Use deepcopy to create independent copies that can be properly moved to device
        self.pretrained_net = copy.deepcopy(module.pretrained_net)

        self.deconv1 = copy.deepcopy(module.deconv1.get_max_op())
        self.deconv2 = copy.deepcopy(module.deconv2.get_max_op())
        self.deconv3 = copy.deepcopy(module.deconv3.get_max_op())
        self.deconv4 = copy.deepcopy(module.deconv4.get_max_op())
        self.deconv5 = copy.deepcopy(module.deconv5.get_max_op())

        self.classifier = copy.deepcopy(module.classifier)

    def forward(self, x):
        output = self.pretrained_net(x)

        x5 = output["x5"]  # size=(N, 512, x.H/32, x.W/32)
        x4 = output["x4"]  # size=(N, 512, x.H/16, x.W/16)
        x3 = output["x3"]  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output["x2"]  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output["x1"]  # size=(N, 64, x.H/2,  x.W/2)

        score = self.deconv1(x5)  # size=(N, 512, x.H/16, x.W/16)
        score = score + x4  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.deconv2(score)  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.deconv3(score)  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2  # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.deconv4(score)  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1  # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.deconv5(score)  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)