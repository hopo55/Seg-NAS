import copy
import torch
from torch import nn
from torchvision import models as tv_models
from torchvision.models import DenseNet121_Weights, densenet121

from utils.operations import MixedOp, MixedOpWithWidth, MixedSkip
from nas.search_space import STANDARD_OP_NAMES, ALL_OP_NAMES, WIDTH_MULTS

# ---------------------------------------------------------------------------
# Encoder configurations: channel counts at each spatial resolution
#   channels = [x1 (H/2), x2 (H/4), x3 (H/8), x4 (H/16), x5 (H/32)]
# ---------------------------------------------------------------------------
ENCODER_CONFIGS = {
    'densenet121':        {'channels': [64,  128, 256, 512,  1024]},
    'resnet50':           {'channels': [64,  256, 512, 1024, 2048]},
    'efficientnet_b0':    {'channels': [16,  24,  40,  112,  320]},
    'mobilenet_v3_large': {'channels': [16,  24,  40,  112,  960]},
}

ranges = {
    "densenet": ((0, 3), (4, 6), (6, 8), (8, 10), (10, 12)),
}


# ---------------------------------------------------------------------------
# Encoder wrappers — each returns dict with keys "x1".."x5"
# ---------------------------------------------------------------------------
class DenseNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
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


class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = x                       # H/2, 64 ch
        x = self.maxpool(x)
        x2 = self.layer1(x)          # H/4, 256 ch
        x3 = self.layer2(x2)         # H/8, 512 ch
        x4 = self.layer3(x3)         # H/16, 1024 ch
        x5 = self.layer4(x4)         # H/32, 2048 ch
        return {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}


class EfficientNetEncoder(nn.Module):
    """EfficientNet-B0 encoder. Uses features[0..7], skips features[8] (1x1 expansion)."""
    def __init__(self):
        super().__init__()
        backbone = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT)
        feats = backbone.features
        # Group into 5 stages by spatial resolution:
        #   stage1 (H/2):  features[0..1] -> 16 ch
        #   stage2 (H/4):  features[2]    -> 24 ch
        #   stage3 (H/8):  features[3]    -> 40 ch
        #   stage4 (H/16): features[4..5] -> 112 ch
        #   stage5 (H/32): features[6..7] -> 320 ch
        self.stage1 = nn.Sequential(*feats[0:2])
        self.stage2 = feats[2]
        self.stage3 = feats[3]
        self.stage4 = nn.Sequential(*feats[4:6])
        self.stage5 = nn.Sequential(*feats[6:8])

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        return {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}


class MobileNetV3Encoder(nn.Module):
    """MobileNetV3-Large encoder."""
    def __init__(self):
        super().__init__()
        backbone = tv_models.mobilenet_v3_large(weights=tv_models.MobileNet_V3_Large_Weights.DEFAULT)
        feats = backbone.features
        # Group into 5 stages by spatial resolution:
        #   stage1 (H/2):  features[0..1]   -> 16 ch
        #   stage2 (H/4):  features[2..3]   -> 24 ch
        #   stage3 (H/8):  features[4..6]   -> 40 ch
        #   stage4 (H/16): features[7..12]  -> 112 ch
        #   stage5 (H/32): features[13..16] -> 960 ch
        self.stage1 = nn.Sequential(*feats[0:2])
        self.stage2 = nn.Sequential(*feats[2:4])
        self.stage3 = nn.Sequential(*feats[4:7])
        self.stage4 = nn.Sequential(*feats[7:13])
        self.stage5 = nn.Sequential(*feats[13:17])

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        return {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}


def get_encoder(name: str):
    """Factory function returning (encoder_module, channels_list).

    channels_list = [x1_ch, x2_ch, x3_ch, x4_ch, x5_ch]
    """
    if name not in ENCODER_CONFIGS:
        raise ValueError(f"Unknown encoder '{name}'. Choose from: {list(ENCODER_CONFIGS.keys())}")
    channels = ENCODER_CONFIGS[name]['channels']
    if name == 'densenet121':
        return DenseNetEncoder(), channels
    elif name == 'resnet50':
        return ResNetEncoder(), channels
    elif name == 'efficientnet_b0':
        return EfficientNetEncoder(), channels
    elif name == 'mobilenet_v3_large':
        return MobileNetV3Encoder(), channels
    else:
        raise ValueError(f"No encoder class for '{name}'")


# Keep legacy alias for backward compat
DenseNet = DenseNetEncoder


class SuperNet(nn.Module):
    """
    SuperNet with configurable search space and encoder.

    Args:
        n_class: number of output classes
        search_space: 'basic', 'extended', or 'industry'
        encoder_name: encoder backbone name (see ENCODER_CONFIGS)
    """
    def __init__(self, n_class, search_space='basic', encoder_name='densenet121'):
        super().__init__()
        self.n_class = n_class
        self.search_space = search_space
        self.encoder_name = encoder_name

        self.pretrained_net, enc_channels = get_encoder(encoder_name)
        # enc_channels = [x1, x2, x3, x4, x5]
        self.encoder_channels = enc_channels

        # Derive decoder channel chain from encoder
        # deconv1: x5 -> x4 resolution
        # deconv2: x4 -> x3 resolution
        # deconv3: x3 -> x2 resolution
        # deconv4: x2 -> x1 resolution
        # deconv5: x1 -> full resolution
        c5, c4, c3, c2, c1 = enc_channels[4], enc_channels[3], enc_channels[2], enc_channels[1], enc_channels[0]
        c_final = max(c1 // 2, 16)  # final output channels before classifier
        self.decoder_channels = [
            (c5, c4),       # deconv1
            (c4, c3),       # deconv2
            (c3, c2),       # deconv3
            (c2, c1),       # deconv4
            (c1, c_final),  # deconv5
        ]

        if search_space == 'industry':
            self.op_names = list(ALL_OP_NAMES)
            self.width_mults = list(WIDTH_MULTS)
            for i, (cin, cout) in enumerate(self.decoder_channels, 1):
                setattr(self, f'deconv{i}',
                        MixedOpWithWidth(cin, cout, op_names=self.op_names, width_mults=self.width_mults))
        elif search_space == 'extended':
            self.op_names = list(STANDARD_OP_NAMES)
            self.width_mults = list(WIDTH_MULTS)
            for i, (cin, cout) in enumerate(self.decoder_channels, 1):
                setattr(self, f'deconv{i}',
                        MixedOpWithWidth(cin, cout, op_names=self.op_names, width_mults=self.width_mults))
        else:
            self.op_names = list(STANDARD_OP_NAMES)
            self.width_mults = [1.0]
            for i, (cin, cout) in enumerate(self.decoder_channels, 1):
                setattr(self, f'deconv{i}',
                        MixedOp(cin, cout, op_names=self.op_names))

        # Searchable skip-connection fusion for deconv1–4
        # (deconv5 outputs to full resolution with no encoder skip)
        skip_channels = [c4, c3, c2, c1]  # encoder channels at skip points
        for i, sc in enumerate(skip_channels, 1):
            setattr(self, f'skip{i}', MixedSkip(sc, sc))

        self.classifier = nn.Conv2d(c_final, n_class, kernel_size=1)

    def _deconvs(self):
        return [self.deconv1, self.deconv2, self.deconv3, self.deconv4, self.deconv5]

    def _skips(self):
        return [self.skip1, self.skip2, self.skip3, self.skip4]

    def forward(self, x, temperature=1.0, hard=False):
        output = self.pretrained_net(x)

        x5 = output["x5"]
        x4 = output["x4"]
        x3 = output["x3"]
        x2 = output["x2"]
        x1 = output["x1"]

        score = self.deconv1(x5, temperature=temperature, hard=hard)
        score = self.skip1(score, x4, temperature=temperature, hard=hard)
        score = self.deconv2(score, temperature=temperature, hard=hard)
        score = self.skip2(score, x3, temperature=temperature, hard=hard)
        score = self.deconv3(score, temperature=temperature, hard=hard)
        score = self.skip3(score, x2, temperature=temperature, hard=hard)
        score = self.deconv4(score, temperature=temperature, hard=hard)
        score = self.skip4(score, x1, temperature=temperature, hard=hard)
        score = self.deconv5(score, temperature=temperature, hard=hard)
        score = self.classifier(score)

        return score

    def clip_alphas(self):
        for d in self._deconvs():
            d.clip_alphas()
        for s in self._skips():
            s.clip_alphas()

    def get_alphas(self):
        """Return alpha values for logging"""
        if self.search_space in ('extended', 'industry'):
            result = [{'op': d.alphas_op.cpu().detach().numpy().tolist(),
                        'width': d.alphas_width.cpu().detach().numpy().tolist()}
                       for d in self._deconvs()]
        else:
            result = [d.alphas.cpu().detach().numpy().tolist() for d in self._deconvs()]
        result.append({'skip': [s.alphas_skip.cpu().detach().numpy().tolist()
                                for s in self._skips()]})
        return result

    def get_alpha_params(self):
        """Return all alpha parameters for optimizer"""
        if self.search_space in ('extended', 'industry'):
            params = []
            for d in self._deconvs():
                params.append(d.alphas_op)
                params.append(d.alphas_width)
        else:
            params = [d.alphas for d in self._deconvs()]
        for s in self._skips():
            params.append(s.alphas_skip)
        return params

    def get_arch_description(self):
        """Return human-readable architecture description"""
        desc = []
        skips = self._skips()
        for i, d in enumerate(self._deconvs(), 1):
            line = f"deconv{i}: {d.get_op_name()}"
            if i <= len(skips):
                line += f" | skip{i}: {skips[i-1].get_skip_name()}"
            desc.append(line)
        return desc

    def get_sampled_flops(self, input_size=128, temperature=1.0):
        """Calculate FLOPs using Gumbel-Softmax sampling (differentiable)."""
        H = input_size // 32
        sizes = [(H * 2, H * 2), (H * 4, H * 4), (H * 8, H * 8),
                 (H * 16, H * 16), (H * 32, H * 32)]
        total_flops = 0
        for d, (h, w) in zip(self._deconvs(), sizes):
            total_flops += d.get_sampled_flops(h, w, temperature)
        # Skip fusion FLOPs (skip1–4 correspond to sizes[0]–sizes[3])
        for s, (h, w) in zip(self._skips(), sizes[:4]):
            total_flops += s.get_sampled_skip_flops(h, w, temperature)
        return total_flops / 1e9

    def get_argmax_flops(self, input_size=128):
        """Calculate FLOPs of argmax selected operations (non-differentiable, for logging)."""
        H = input_size // 32
        sizes = [(H * 2, H * 2), (H * 4, H * 4), (H * 8, H * 8),
                 (H * 16, H * 16), (H * 32, H * 32)]
        total_flops = 0
        for d, (h, w) in zip(self._deconvs(), sizes):
            total_flops += d.get_argmax_flops(h, w)
        for s, (h, w) in zip(self._skips(), sizes[:4]):
            total_flops += s.get_argmax_skip_flops(h, w)
        return total_flops / 1e9

    def get_arch_indices(self):
        """
        Get architecture indices for all layers.

        Returns:
            op_indices: Tensor of shape [5] with operation indices
            width_indices: Tensor of shape [5] with width indices
        """
        deconvs = self._deconvs()
        if self.search_space in ('extended', 'industry'):
            op_indices = []
            width_indices = []
            for d in deconvs:
                op_idx, width_idx = d.get_arch_indices()
                op_indices.append(op_idx)
                width_indices.append(width_idx)
            return torch.tensor(op_indices), torch.tensor(width_indices)
        else:
            op_indices = [d.get_max_alpha_idx() for d in deconvs]
            width_indices = [len(WIDTH_MULTS) - 1] * len(deconvs)
            return torch.tensor(op_indices), torch.tensor(width_indices)

    def get_sampled_latency(self, latency_lut, temperature=1.0):
        """Calculate total latency using LUT and Gumbel-Softmax (differentiable)."""
        if self.search_space not in ('extended', 'industry'):
            raise NotImplementedError("Latency sampling only supported for width-aware search spaces")
        deconvs = self._deconvs()
        total_latency = deconvs[0].alphas_op.new_tensor(0.0)
        for layer_idx, d in enumerate(deconvs):
            total_latency = total_latency + d.get_sampled_latency(latency_lut, layer_idx, temperature)
        return total_latency

    def get_argmax_latency(self, latency_lut):
        """Get total latency of argmax selected architecture (non-differentiable)."""
        if self.search_space not in ('extended', 'industry'):
            raise NotImplementedError("Latency only supported for width-aware search spaces")
        total_latency = 0.0
        for layer_idx, d in enumerate(self._deconvs()):
            total_latency += d.get_argmax_latency(latency_lut, layer_idx)
        return total_latency

    def set_active_widths(self, active_width_indices):
        """Set active widths for all decoder layers (progressive shrinking)."""
        if self.search_space not in ('extended', 'industry'):
            return
        for d in self._deconvs():
            d.set_active_widths(active_width_indices)

    def get_active_widths(self):
        """Return current active width indices."""
        if self.search_space not in ('extended', 'industry'):
            return [0]
        return list(self.deconv1._active_width_indices)

    @torch.no_grad()
    def forward_teacher(self, x):
        """Forward pass with the largest subnet (width=1.0 only)."""
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
            op_weights: Tensor [5, num_ops]
            width_weights: Tensor [5, num_widths]
        """
        deconvs = self._deconvs()
        if self.search_space in ('extended', 'industry'):
            op_weights = torch.stack([torch.softmax(d.alphas_op, dim=0) for d in deconvs])
            width_weights = torch.stack([torch.softmax(d.alphas_width, dim=0) for d in deconvs])
            return op_weights, width_weights
        else:
            op_weights = torch.stack([torch.softmax(d.alphas, dim=0) for d in deconvs])
            width_weights = op_weights.new_zeros((5, len(WIDTH_MULTS)))
            width_weights[:, -1] = 1.0
            return op_weights, width_weights


class OptimizedNetwork(nn.Module):
    """Extract the final architecture from SuperNet based on max alpha selection"""

    SKIP_NAMES = ['add', 'concat', 'attn_gate']

    def __init__(self, super_net):
        super(OptimizedNetwork, self).__init__()

        if isinstance(super_net, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            module = super_net.module
        else:
            module = super_net

        self.search_space = getattr(module, 'search_space', 'basic')
        self.pretrained_net = copy.deepcopy(module.pretrained_net)

        if self.search_space in ('extended', 'industry'):
            self._setup_extended_deconv(module)
        else:
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

        # Extract selected skip fusion modes (deconv1–4)
        self.skip_modes = []
        for i in range(1, 5):
            skip_module = getattr(module, f'skip{i}')
            mode_idx = skip_module.get_max_skip_idx()
            mode_name = self.SKIP_NAMES[mode_idx]
            self.skip_modes.append(mode_name)
            if mode_name == 'concat':
                setattr(self, f'skip_concat_proj{i}',
                        copy.deepcopy(skip_module.concat_proj))
                setattr(self, f'skip_concat_bn{i}',
                        copy.deepcopy(skip_module.concat_bn))
            elif mode_name == 'attn_gate':
                setattr(self, f'skip_attn_conv{i}',
                        copy.deepcopy(skip_module.attn_conv))
                setattr(self, f'skip_attn_bn{i}',
                        copy.deepcopy(skip_module.attn_bn))

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

    def _apply_skip(self, idx, decoder_feat, encoder_feat):
        """Apply the selected skip fusion for layer idx (1-based)."""
        mode = self.skip_modes[idx - 1]
        if mode == 'add':
            return decoder_feat + encoder_feat
        elif mode == 'concat':
            proj = getattr(self, f'skip_concat_proj{idx}')
            bn = getattr(self, f'skip_concat_bn{idx}')
            return bn(proj(torch.cat([decoder_feat, encoder_feat], dim=1)))
        else:  # attn_gate
            conv = getattr(self, f'skip_attn_conv{idx}')
            bn = getattr(self, f'skip_attn_bn{idx}')
            gate = torch.sigmoid(bn(conv(encoder_feat)))
            return decoder_feat * gate

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
        score = self._apply_skip(1, score, x4)

        # deconv2
        score = self.deconv2(score)
        score = self.relu(score)
        score = self.bn2(score)
        if self.proj2 is not None:
            score = self.proj2(score)
        score = self._apply_skip(2, score, x3)

        # deconv3
        score = self.deconv3(score)
        score = self.relu(score)
        score = self.bn3(score)
        if self.proj3 is not None:
            score = self.proj3(score)
        score = self._apply_skip(3, score, x2)

        # deconv4
        score = self.deconv4(score)
        score = self.relu(score)
        score = self.bn4(score)
        if self.proj4 is not None:
            score = self.proj4(score)
        score = self._apply_skip(4, score, x1)

        # deconv5
        score = self.deconv5(score)
        score = self.relu(score)
        score = self.bn5(score)
        if self.proj5 is not None:
            score = self.proj5(score)

        score = self.classifier(score)
        return score
