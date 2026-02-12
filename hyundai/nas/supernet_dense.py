import copy
import torch
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121

from utils.operations import MixedOp, MixedOpWithWidth
from nas.search_space import STANDARD_OP_NAMES, ALL_OP_NAMES, WIDTH_MULTS

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
    """
    SuperNet with configurable search space.

    Args:
        n_class: number of output classes
        search_space: 'basic', 'extended', or 'industry'
    """
    def __init__(self, n_class, search_space='basic'):
        super().__init__()
        self.n_class = n_class
        self.search_space = search_space
        self.pretrained_net = DenseNet()

        if search_space == 'industry':
            # Industry search space: 7 ops x 3 widths
            self.op_names = list(ALL_OP_NAMES)
            self.width_mults = list(WIDTH_MULTS)
            self.deconv1 = MixedOpWithWidth(1024, 512, op_names=self.op_names, width_mults=self.width_mults)
            self.deconv2 = MixedOpWithWidth(512, 256, op_names=self.op_names, width_mults=self.width_mults)
            self.deconv3 = MixedOpWithWidth(256, 128, op_names=self.op_names, width_mults=self.width_mults)
            self.deconv4 = MixedOpWithWidth(128, 64, op_names=self.op_names, width_mults=self.width_mults)
            self.deconv5 = MixedOpWithWidth(64, 32, op_names=self.op_names, width_mults=self.width_mults)
        elif search_space == 'extended':
            # Extended search space: 5 ops x 3 widths
            self.op_names = list(STANDARD_OP_NAMES)
            self.width_mults = list(WIDTH_MULTS)
            self.deconv1 = MixedOpWithWidth(1024, 512, op_names=self.op_names, width_mults=self.width_mults)
            self.deconv2 = MixedOpWithWidth(512, 256, op_names=self.op_names, width_mults=self.width_mults)
            self.deconv3 = MixedOpWithWidth(256, 128, op_names=self.op_names, width_mults=self.width_mults)
            self.deconv4 = MixedOpWithWidth(128, 64, op_names=self.op_names, width_mults=self.width_mults)
            self.deconv5 = MixedOpWithWidth(64, 32, op_names=self.op_names, width_mults=self.width_mults)
        else:
            # Basic search space (5 operations only)
            self.op_names = list(STANDARD_OP_NAMES)
            self.width_mults = [1.0]
            self.deconv1 = MixedOp(1024, 512, op_names=self.op_names)
            self.deconv2 = MixedOp(512, 256, op_names=self.op_names)
            self.deconv3 = MixedOp(256, 128, op_names=self.op_names)
            self.deconv4 = MixedOp(128, 64, op_names=self.op_names)
            self.deconv5 = MixedOp(64, 32, op_names=self.op_names)

        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x, temperature=1.0, hard=False):
        output = self.pretrained_net(x)

        x5 = output["x5"]  # size=(N, 512, x.H/32, x.W/32)
        x4 = output["x4"]  # size=(N, 512, x.H/16, x.W/16)
        x3 = output["x3"]  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output["x2"]  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output["x1"]  # size=(N, 64, x.H/2,  x.W/2)

        score = self.deconv1(x5, temperature=temperature, hard=hard)  # size=(N, 512, x.H/16, x.W/16)
        score = score + x4  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.deconv2(score, temperature=temperature, hard=hard)  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.deconv3(score, temperature=temperature, hard=hard)  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2  # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.deconv4(score, temperature=temperature, hard=hard)  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1  # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.deconv5(score, temperature=temperature, hard=hard)  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)

    def clip_alphas(self):
        self.deconv1.clip_alphas()
        self.deconv2.clip_alphas()
        self.deconv3.clip_alphas()
        self.deconv4.clip_alphas()
        self.deconv5.clip_alphas()

    def get_alphas(self):
        """Return alpha values for logging"""
        if self.search_space in ('extended', 'industry'):
            alpha_list = []
            for deconv in [self.deconv1, self.deconv2, self.deconv3, self.deconv4, self.deconv5]:
                alpha_list.append({
                    'op': deconv.alphas_op.cpu().detach().numpy().tolist(),
                    'width': deconv.alphas_width.cpu().detach().numpy().tolist()
                })
            return alpha_list
        else:
            alpha_list = [
                self.deconv1.alphas.cpu().detach().numpy().tolist(),
                self.deconv2.alphas.cpu().detach().numpy().tolist(),
                self.deconv3.alphas.cpu().detach().numpy().tolist(),
                self.deconv4.alphas.cpu().detach().numpy().tolist(),
                self.deconv5.alphas.cpu().detach().numpy().tolist(),
            ]
            return alpha_list

    def get_alpha_params(self):
        """Return all alpha parameters for optimizer"""
        if self.search_space in ('extended', 'industry'):
            params = []
            for deconv in [self.deconv1, self.deconv2, self.deconv3, self.deconv4, self.deconv5]:
                params.append(deconv.alphas_op)
                params.append(deconv.alphas_width)
            return params
        else:
            return [
                self.deconv1.alphas,
                self.deconv2.alphas,
                self.deconv3.alphas,
                self.deconv4.alphas,
                self.deconv5.alphas,
            ]

    def get_arch_description(self):
        """Return human-readable architecture description"""
        desc = []
        for i, deconv in enumerate([self.deconv1, self.deconv2, self.deconv3, self.deconv4, self.deconv5], 1):
            desc.append(f"deconv{i}: {deconv.get_op_name()}")
        return desc

    def get_sampled_flops(self, input_size=128, temperature=1.0):
        """
        Calculate FLOPs using Gumbel-Softmax sampling (differentiable).
        Returns FLOPs closer to the actual selected operation.
        """
        H = input_size // 32
        sizes = [
            (H * 2, H * 2),      # deconv1: 4 -> 8
            (H * 4, H * 4),      # deconv2: 8 -> 16
            (H * 8, H * 8),      # deconv3: 16 -> 32
            (H * 16, H * 16),    # deconv4: 32 -> 64
            (H * 32, H * 32),    # deconv5: 64 -> 128
        ]

        total_flops = 0
        total_flops += self.deconv1.get_sampled_flops(sizes[0][0], sizes[0][1], temperature)
        total_flops += self.deconv2.get_sampled_flops(sizes[1][0], sizes[1][1], temperature)
        total_flops += self.deconv3.get_sampled_flops(sizes[2][0], sizes[2][1], temperature)
        total_flops += self.deconv4.get_sampled_flops(sizes[3][0], sizes[3][1], temperature)
        total_flops += self.deconv5.get_sampled_flops(sizes[4][0], sizes[4][1], temperature)

        return total_flops / 1e9  # Return in GFLOPs

    def get_argmax_flops(self, input_size=128):
        """
        Calculate FLOPs of argmax selected operations (non-differentiable, for logging).
        """
        H = input_size // 32
        sizes = [
            (H * 2, H * 2),
            (H * 4, H * 4),
            (H * 8, H * 8),
            (H * 16, H * 16),
            (H * 32, H * 32),
        ]

        total_flops = 0
        total_flops += self.deconv1.get_argmax_flops(sizes[0][0], sizes[0][1])
        total_flops += self.deconv2.get_argmax_flops(sizes[1][0], sizes[1][1])
        total_flops += self.deconv3.get_argmax_flops(sizes[2][0], sizes[2][1])
        total_flops += self.deconv4.get_argmax_flops(sizes[3][0], sizes[3][1])
        total_flops += self.deconv5.get_argmax_flops(sizes[4][0], sizes[4][1])

        return total_flops / 1e9  # Return in GFLOPs

    def get_arch_indices(self):
        """
        Get architecture indices for all layers.

        Returns:
            op_indices: Tensor of shape [5] with operation indices
            width_indices: Tensor of shape [5] with width indices
        """
        deconvs = [self.deconv1, self.deconv2, self.deconv3, self.deconv4, self.deconv5]

        if self.search_space in ('extended', 'industry'):
            op_indices = []
            width_indices = []
            for deconv in deconvs:
                op_idx, width_idx = deconv.get_arch_indices()
                op_indices.append(op_idx)
                width_indices.append(width_idx)
            return torch.tensor(op_indices), torch.tensor(width_indices)
        else:
            # Basic search space: only op indices, width is always 1.0
            op_indices = [deconv.get_max_alpha_idx() for deconv in deconvs]
            width_indices = [len(WIDTH_MULTS) - 1] * len(deconvs)  # Index for width 1.0
            return torch.tensor(op_indices), torch.tensor(width_indices)

    def get_sampled_latency(self, latency_lut, temperature=1.0):
        """
        Calculate total latency using LUT and Gumbel-Softmax (differentiable).

        Args:
            latency_lut: LatencyLUT object with measured latencies
            temperature: Gumbel-Softmax temperature

        Returns:
            Total latency in milliseconds (differentiable)
        """
        if self.search_space not in ('extended', 'industry'):
            raise NotImplementedError("Latency sampling only supported for width-aware search spaces")

        deconvs = [self.deconv1, self.deconv2, self.deconv3, self.deconv4, self.deconv5]

        total_latency = deconvs[0].alphas_op.new_tensor(0.0)
        for layer_idx, deconv in enumerate(deconvs):
            total_latency = total_latency + deconv.get_sampled_latency(
                latency_lut, layer_idx, temperature
            )

        return total_latency

    def get_argmax_latency(self, latency_lut):
        """
        Get total latency of argmax selected architecture (non-differentiable).

        Args:
            latency_lut: LatencyLUT object

        Returns:
            Total latency in milliseconds
        """
        if self.search_space not in ('extended', 'industry'):
            raise NotImplementedError("Latency only supported for width-aware search spaces")

        deconvs = [self.deconv1, self.deconv2, self.deconv3, self.deconv4, self.deconv5]

        total_latency = 0.0
        for layer_idx, deconv in enumerate(deconvs):
            total_latency += deconv.get_argmax_latency(latency_lut, layer_idx)

        return total_latency

    def set_active_widths(self, active_width_indices):
        """Set active widths for all decoder layers (progressive shrinking).

        Args:
            active_width_indices: List of indices into WIDTH_MULTS that should be active.
                                 e.g., [2] for width=1.0 only
        """
        if self.search_space not in ('extended', 'industry'):
            return
        for deconv in [self.deconv1, self.deconv2, self.deconv3, self.deconv4, self.deconv5]:
            deconv.set_active_widths(active_width_indices)

    def get_active_widths(self):
        """Return current active width indices."""
        if self.search_space not in ('extended', 'industry'):
            return [0]
        return list(self.deconv1._active_width_indices)

    @torch.no_grad()
    def forward_teacher(self, x):
        """Forward pass with the largest subnet (width=1.0 only).
        Used as the teacher for knowledge distillation during progressive shrinking.
        """
        original_indices = self.get_active_widths()
        max_width_idx = len(self.width_mults) - 1
        self.set_active_widths([max_width_idx])
        output = self.forward(x, temperature=0.1, hard=True)
        self.set_active_widths(original_indices)
        return output

    def get_alpha_weights(self):
        """
        Get normalized alpha weights for all layers (for latency predictor).

        Returns:
            op_weights: Tensor [5, num_ops] - softmax weights for operations
            width_weights: Tensor [5, num_widths] - softmax weights for widths
        """
        deconvs = [self.deconv1, self.deconv2, self.deconv3, self.deconv4, self.deconv5]

        if self.search_space in ('extended', 'industry'):
            op_weights = torch.stack([
                torch.softmax(d.alphas_op, dim=0) for d in deconvs
            ])
            width_weights = torch.stack([
                torch.softmax(d.alphas_width, dim=0) for d in deconvs
            ])
            return op_weights, width_weights
        else:
            op_weights = torch.stack([
                torch.softmax(d.alphas, dim=0) for d in deconvs
            ])
            # For basic search space, width is always 1.0
            width_weights = op_weights.new_zeros((5, len(WIDTH_MULTS)))
            width_weights[:, -1] = 1.0  # Last index = width 1.0
            return op_weights, width_weights


class OptimizedNetwork(nn.Module):
    """Extract the final architecture from SuperNet based on max alpha selection"""
    def __init__(self, super_net):
        super(OptimizedNetwork, self).__init__()

        if isinstance(super_net, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            module = super_net.module
        else:
            module = super_net

        self.search_space = getattr(module, 'search_space', 'basic')
        self.pretrained_net = copy.deepcopy(module.pretrained_net)

        if self.search_space in ('extended', 'industry'):
            # For extended search space, we need to extract op + bn + optional projection
            self._setup_extended_deconv(module)
        else:
            # For basic search space
            self.deconv1 = copy.deepcopy(module.deconv1.get_max_op())
            self.bn1 = copy.deepcopy(module.deconv1.bn)
            self.deconv2 = copy.deepcopy(module.deconv2.get_max_op())
            self.bn2 = copy.deepcopy(module.deconv2.bn)
            self.deconv3 = copy.deepcopy(module.deconv3.get_max_op())
            self.bn3 = copy.deepcopy(module.deconv3.bn)
            self.deconv4 = copy.deepcopy(module.deconv4.get_max_op())
            self.bn4 = copy.deepcopy(module.deconv4.bn)
            self.deconv5 = copy.deepcopy(module.deconv5.get_max_op())
            self.bn5 = copy.deepcopy(module.deconv5.bn)

            self.proj1 = None
            self.proj2 = None
            self.proj3 = None
            self.proj4 = None
            self.proj5 = None

        self.classifier = copy.deepcopy(module.classifier)
        self.relu = nn.ReLU(inplace=True)

    def _setup_extended_deconv(self, module):
        """Setup deconv layers for extended search space"""
        for i, deconv in enumerate([module.deconv1, module.deconv2,
                                    module.deconv3, module.deconv4, module.deconv5], 1):
            op = copy.deepcopy(deconv.get_max_op())
            bn = copy.deepcopy(deconv.get_max_bn())
            proj = deconv.get_max_proj()
            if proj is not None:
                proj = copy.deepcopy(proj)

            setattr(self, f'deconv{i}', op)
            setattr(self, f'bn{i}', bn)
            setattr(self, f'proj{i}', proj)

    def forward(self, x):
        output = self.pretrained_net(x)

        x5 = output["x5"]
        x4 = output["x4"]
        x3 = output["x3"]
        x2 = output["x2"]
        x1 = output["x1"]

        # deconv1
        score = self.deconv1(x5)
        score = self.relu(score)
        score = self.bn1(score)
        if self.proj1 is not None:
            score = self.proj1(score)
        score = score + x4

        # deconv2
        score = self.deconv2(score)
        score = self.relu(score)
        score = self.bn2(score)
        if self.proj2 is not None:
            score = self.proj2(score)
        score = score + x3

        # deconv3
        score = self.deconv3(score)
        score = self.relu(score)
        score = self.bn3(score)
        if self.proj3 is not None:
            score = self.proj3(score)
        score = score + x2

        # deconv4
        score = self.deconv4(score)
        score = self.relu(score)
        score = self.bn4(score)
        if self.proj4 is not None:
            score = self.proj4(score)
        score = score + x1

        # deconv5
        score = self.deconv5(score)
        score = self.relu(score)
        score = self.bn5(score)
        if self.proj5 is not None:
            score = self.proj5(score)

        score = self.classifier(score)
        return score
