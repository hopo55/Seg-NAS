"""
Latency Look-Up Table Builder for multiple hardware platforms.

This module measures the actual inference latency of each operation
on target hardware and builds a LUT for efficient latency estimation.
"""

import torch
import torch.nn as nn
import time
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.operations import (
    DepthwiseSeparableConvTranspose2d,
    _same_padding
)


# Hardware specifications for encoding
HARDWARE_SPECS = {
    # --- GPU (CUDA) ---
    'A6000': {
        'cuda_cores': 10752,
        'memory_bandwidth': 768,  # GB/s
        'tensor_cores': 336,
        'memory_gb': 48,
    },
    'RTX3090': {
        'cuda_cores': 10496,
        'memory_bandwidth': 936,
        'tensor_cores': 328,
        'memory_gb': 24,
    },
    'RTX4090': {
        'cuda_cores': 16384,
        'memory_bandwidth': 1008,
        'tensor_cores': 512,
        'memory_gb': 24,
    },
    # --- Edge (CUDA) ---
    'JetsonOrin': {
        'cuda_cores': 2048,
        'memory_bandwidth': 205,
        'tensor_cores': 64,
        'memory_gb': 8,
    },
    # --- Edge (CPU-only) ---
    'RaspberryPi5': {
        'cuda_cores': 0,
        'memory_bandwidth': 32,   # LPDDR4X-4267
        'tensor_cores': 0,
        'memory_gb': 8,
    },
    'Odroid': {
        'cuda_cores': 0,
        'memory_bandwidth': 25,   # LPDDR4/4X
        'tensor_cores': 0,
        'memory_gb': 4,
    },
}

# Operation names mapping
OP_NAMES = ['Conv3x3', 'Conv5x5', 'Conv7x7', 'DWSep3x3', 'DWSep5x5',
            'EdgeConv', 'DilatedDWSep', 'Conv3x3_SE', 'DWSep3x3_SE']
WIDTH_MULTS = [0.25, 0.5, 0.75, 1.0, 1.25]


class LatencyLUTBuilder:
    """
    Build latency look-up table by measuring actual inference time.

    Usage:
        builder = LatencyLUTBuilder(input_size=128)
        lut = builder.build_lut('RTX4090', save_path='lut_rtx4090.json')
    """

    def __init__(self, input_size: int = 128, warmup: int = 200, repeat: int = 300,
                 encoder_name: str = 'densenet121',
                 input_h: Optional[int] = None, input_w: Optional[int] = None):
        """
        Args:
            input_size: Input image size (square, used when input_h/input_w not set)
            warmup: Number of warmup iterations
            repeat: Number of measurement iterations
            encoder_name: Encoder backbone name (determines decoder channel sizes)
            input_h: Input height (overrides input_size if set with input_w)
            input_w: Input width (overrides input_size if set with input_h)
        """
        self.warmup = warmup
        self.repeat = repeat
        self.encoder_name = encoder_name

        # Resolve input dimensions (non-square support)
        if input_h is not None and input_w is not None:
            self.input_h = input_h
            self.input_w = input_w
        else:
            self.input_h = input_size
            self.input_w = input_size
        self.input_size = f"{self.input_h}x{self.input_w}"

        # Derive layer configurations from encoder channels
        from nas.supernet_dense import ENCODER_CONFIGS
        enc = ENCODER_CONFIGS[encoder_name]['channels']  # [x1, x2, x3, x4, x5]
        c5, c4, c3, c2, c1 = enc[4], enc[3], enc[2], enc[1], enc[0]
        c_final = max(c1 // 2, 16)

        H = self.input_h // 32
        W = self.input_w // 32
        self.layer_configs = [
            {'C_in': c5, 'C_out': c4, 'H_out': H * 2, 'W_out': W * 2},         # deconv1
            {'C_in': c4, 'C_out': c3, 'H_out': H * 4, 'W_out': W * 4},         # deconv2
            {'C_in': c3, 'C_out': c2, 'H_out': H * 8, 'W_out': W * 8},         # deconv3
            {'C_in': c2, 'C_out': c1, 'H_out': H * 16, 'W_out': W * 16},       # deconv4
            {'C_in': c1, 'C_out': c_final, 'H_out': H * 32, 'W_out': W * 32},  # deconv5
        ]

    def _create_operation(self, op_name: str, C_in: int, C_out: int,
                          width_mult: float = 1.0) -> nn.Module:
        """Create a single operation module."""
        C_mid = max(1, int(C_out * width_mult))

        if op_name == 'Conv3x3':
            return nn.ConvTranspose2d(C_in, C_mid, 3, stride=2, padding=1, output_padding=1)
        elif op_name == 'Conv5x5':
            return nn.ConvTranspose2d(C_in, C_mid, 5, stride=2, padding=2, output_padding=1)
        elif op_name == 'Conv7x7':
            return nn.ConvTranspose2d(C_in, C_mid, 7, stride=2, padding=3, output_padding=1)
        elif op_name == 'DWSep3x3':
            return DepthwiseSeparableConvTranspose2d(C_in, C_mid, 3, stride=2, padding=1, output_padding=1)
        elif op_name == 'DWSep5x5':
            return DepthwiseSeparableConvTranspose2d(C_in, C_mid, 5, stride=2, padding=2, output_padding=1)
        elif op_name == 'EdgeConv':
            from nas.search_space import EdgeAwareConvTranspose
            return EdgeAwareConvTranspose(C_in, C_mid, stride=2)
        elif op_name == 'DilatedDWSep':
            from nas.search_space import DilatedDWSepConvTranspose
            return DilatedDWSepConvTranspose(C_in, C_mid, kernel_size=3, stride=2, dilation=2)
        elif op_name == 'Conv3x3_SE':
            from nas.search_space import ConvTransposeSE
            return ConvTransposeSE(C_in, C_mid, kernel_size=3, stride=2)
        elif op_name == 'DWSep3x3_SE':
            from nas.search_space import DWSepConvTransposeSE
            return DWSepConvTransposeSE(C_in, C_mid, kernel_size=3, stride=2)
        else:
            raise ValueError(f"Unknown operation: {op_name}")

    def measure_op_latency(self, op: nn.Module, C_in: int, H_in: int, W_in: int,
                           device: str = 'cuda') -> Tuple[float, float]:
        """
        Measure single operation latency with IQR-based outlier removal.

        Args:
            op: Operation module
            C_in: Input channels
            H_in: Input height
            W_in: Input width
            device: Device to measure on

        Returns:
            (mean_latency_ms, std_latency_ms)
        """
        op = op.to(device).eval()

        # Clear GPU cache before measurement
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        x = torch.randn(1, C_in, H_in, W_in, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup):
                _ = op(x)

        if device == 'cuda':
            torch.cuda.synchronize()

        # Measure
        times = []
        with torch.no_grad():
            for _ in range(self.repeat):
                if device == 'cuda':
                    torch.cuda.synchronize()

                start = time.perf_counter()
                _ = op(x)

                if device == 'cuda':
                    torch.cuda.synchronize()

                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

        times = np.array(times)

        # IQR-based outlier removal
        q1 = np.percentile(times, 25)
        q3 = np.percentile(times, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        filtered = times[(times >= lower) & (times <= upper)]

        if len(filtered) < len(times) * 0.5:
            # Too many outliers removed — fall back to median-based estimate
            print(f"  Warning: {len(times) - len(filtered)}/{len(times)} outliers detected, "
                  f"using median-based estimate")
            median = float(np.median(times))
            mad = float(np.median(np.abs(times - median)))
            return median, mad

        return float(np.mean(filtered)), float(np.std(filtered))

    def build_lut(self, hardware_name: str, save_path: Optional[str] = None,
                  op_names: List[str] = None, width_mults: List[float] = None,
                  device: str = 'cuda') -> Dict:
        """
        Build complete LUT for one hardware.

        Args:
            hardware_name: Name of hardware (for metadata)
            save_path: Path to save JSON file
            op_names: List of operation names to measure
            width_mults: List of width multipliers
            device: Device to measure on

        Returns:
            LUT dictionary
        """
        if op_names is None:
            op_names = OP_NAMES
        if width_mults is None:
            width_mults = WIDTH_MULTS

        lut = {
            'hardware': hardware_name,
            'input_size': self.input_size,
            'input_h': self.input_h,
            'input_w': self.input_w,
            'warmup': self.warmup,
            'repeat': self.repeat,
            'layers': {}
        }

        print(f"\nBuilding LUT for {hardware_name}")
        print(f"Operations: {op_names}")
        print(f"Width multipliers: {width_mults}")
        print("=" * 60)

        for layer_idx, cfg in enumerate(tqdm(self.layer_configs, desc="Layers")):
            layer_key = f"layer_{layer_idx}"
            lut['layers'][layer_key] = {
                'C_in': cfg['C_in'],
                'C_out': cfg['C_out'],
                'H_out': cfg['H_out'],
                'W_out': cfg['W_out'],
                'ops': {}
            }

            # Input size for this layer (before upsampling)
            H_in = cfg['H_out'] // 2
            W_in = cfg['W_out'] // 2

            for op_name in op_names:
                for wm in width_mults:
                    try:
                        # Create operation
                        op = self._create_operation(
                            op_name, cfg['C_in'], cfg['C_out'], wm
                        )

                        # Measure latency
                        mean_lat, std_lat = self.measure_op_latency(
                            op, cfg['C_in'], H_in, W_in, device
                        )

                        # Store in LUT
                        key = f"{op_name}_w{int(wm*100)}"
                        lut['layers'][layer_key]['ops'][key] = {
                            'mean_ms': round(mean_lat, 4),
                            'std_ms': round(std_lat, 4)
                        }

                    except Exception as e:
                        print(f"Warning: Failed to measure {op_name}_w{int(wm*100)} "
                              f"at layer {layer_idx}: {e}")
                        continue

        # Calculate total latency range
        min_total, max_total = self._calculate_latency_range(lut)
        lut['latency_range'] = {
            'min_ms': round(min_total, 4),
            'max_ms': round(max_total, 4)
        }

        print(f"\nLatency range: {min_total:.2f} - {max_total:.2f} ms")

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(lut, f, indent=2)
            print(f"LUT saved to: {save_path}")

        return lut

    def _calculate_latency_range(self, lut: Dict) -> Tuple[float, float]:
        """Calculate min and max total latency from LUT."""
        min_total = 0
        max_total = 0

        for layer_key in lut['layers']:
            ops = lut['layers'][layer_key]['ops']
            if ops:
                latencies = [v['mean_ms'] for v in ops.values()]
                min_total += min(latencies)
                max_total += max(latencies)

        return min_total, max_total

    def build_all_hardware_luts(self, hardware_list: List[str] = None,
                                 save_dir: str = './hyundai/latency/luts',
                                 hardware_name: Optional[str] = None,
                                 **kwargs) -> Dict[str, Dict]:
        """
        Build LUTs for all specified hardware.

        Note: This requires running on each hardware separately.
        For now, it builds LUT for the current hardware.

        Args:
            hardware_list: List of known hardware names
            save_dir: Directory to save LUT JSON files
            hardware_name: Manually specify hardware name (for CPU-only devices
                like RaspberryPi5, Odroid). Overrides auto-detection.
        """
        if hardware_list is None:
            hardware_list = list(HARDWARE_SPECS.keys())

        luts = {}
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Manual hardware name override (for CPU-only edge devices)
        if hardware_name is not None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Hardware name override: {hardware_name} (device: {device})")
            lut = self.build_lut(
                hardware_name,
                save_path=save_dir / f"lut_{hardware_name.lower()}.json",
                device=device,
                **kwargs
            )
            luts[hardware_name] = lut
            return luts

        # Auto-detect current hardware
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Detected GPU: {gpu_name}")

            # Match to known hardware (ignore spaces for comparison)
            current_hw = None
            gpu_name_nospace = gpu_name.replace(' ', '').lower()
            for hw_name in hardware_list:
                if hw_name.replace(' ', '').lower() in gpu_name_nospace:
                    current_hw = hw_name
                    break

            if current_hw:
                print(f"Building LUT for: {current_hw}")
                lut = self.build_lut(
                    current_hw,
                    save_path=save_dir / f"lut_{current_hw.lower()}.json",
                    **kwargs
                )
                luts[current_hw] = lut
            else:
                print(f"Warning: GPU '{gpu_name}' not in known hardware list.")
                print(f"Building LUT with generic name...")
                lut = self.build_lut(
                    gpu_name.replace(' ', '_'),
                    save_path=save_dir / f"lut_{gpu_name.replace(' ', '_').lower()}.json",
                    **kwargs
                )
                luts[gpu_name] = lut
        else:
            print("No CUDA available. Building CPU LUT...")
            lut = self.build_lut('CPU', save_path=save_dir / 'lut_cpu.json',
                                 device='cpu', **kwargs)
            luts['CPU'] = lut

        return luts


def load_lut(lut_path: str) -> Dict:
    """Load LUT from JSON file."""
    with open(lut_path, 'r') as f:
        return json.load(f)


def merge_luts(lut_paths: List[str], save_path: Optional[str] = None) -> Dict:
    """
    Merge multiple hardware LUTs into a single file.

    Args:
        lut_paths: List of LUT JSON file paths
        save_path: Path to save merged LUT

    Returns:
        Merged LUT dictionary
    """
    merged = {'hardware_luts': {}}

    for path in lut_paths:
        lut = load_lut(path)
        hw_name = lut['hardware']
        merged['hardware_luts'][hw_name] = lut

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(merged, f, indent=2)

    return merged


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder_name', type=str, default='densenet121',
                    choices=['densenet121', 'resnet50', 'efficientnet_b0', 'mobilenet_v3_large'])
    ap.add_argument('--input_size', type=int, default=128)
    ap.add_argument('--input_h', type=int, default=None,
                    help='Input height (overrides input_size if set with --input_w)')
    ap.add_argument('--input_w', type=int, default=None,
                    help='Input width (overrides input_size if set with --input_h)')
    ap.add_argument('--hardware_name', type=str, default=None,
                    help='Override hardware name (for CPU-only devices)')
    cli_args = ap.parse_args()

    builder = LatencyLUTBuilder(input_size=cli_args.input_size, warmup=200, repeat=300,
                                encoder_name=cli_args.encoder_name,
                                input_h=cli_args.input_h, input_w=cli_args.input_w)
    luts = builder.build_all_hardware_luts(hardware_name=cli_args.hardware_name)
    print("\nLUT building complete!")
