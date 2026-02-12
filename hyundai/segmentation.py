import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
import json

import wandb
from utils.utils import set_device, check_tensor_in_list, get_model_complexity, AverageMeter, get_iou_score
from utils.wandb_filter import is_minimal_metrics_enabled
from nas.supernet_dense import SuperNet, OptimizedNetwork
from nas.train_supernet import train_architecture, train_architecture_with_latency
from nas.train_samplenet import train_samplenet
from nas.pareto_search import ParetoSearcher, print_pareto_summary
from nas.search_space import STANDARD_OP_NAMES, ALL_OP_NAMES, WIDTH_MULTS, calculate_search_space_size


def _wandb_active(args) -> bool:
    return getattr(wandb, 'run', None) is not None and (not hasattr(args, 'rank') or args.rank == 0)


def _describe_search_space(search_space: str):
    if search_space == 'industry':
        op_names = ALL_OP_NAMES
        width_mults = WIDTH_MULTS
    elif search_space == 'extended':
        op_names = STANDARD_OP_NAMES
        width_mults = WIDTH_MULTS
    else:
        op_names = STANDARD_OP_NAMES
        width_mults = [1.0]
    return op_names, width_mults


@torch.no_grad()
def _evaluate_extracted_subnet(supernet_model, arch, val_loader, device):
    """Evaluate an extracted subnet (argmax-fixed architecture) on validation set."""
    if isinstance(supernet_model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        module = supernet_model.module
    else:
        module = supernet_model

    deconvs = [module.deconv1, module.deconv2, module.deconv3, module.deconv4, module.deconv5]
    saved = []
    for deconv in deconvs:
        if hasattr(deconv, 'alphas_op'):
            saved.append((deconv.alphas_op.detach().clone(), deconv.alphas_width.detach().clone()))
        else:
            saved.append((deconv.alphas.detach().clone(), None))

    for i, deconv in enumerate(deconvs):
        if hasattr(deconv, 'alphas_op'):
            deconv.alphas_op.data.fill_(-10)
            deconv.alphas_op.data[arch.op_indices[i]] = 10
            deconv.alphas_width.data.fill_(-10)
            deconv.alphas_width.data[arch.width_indices[i]] = 10
        else:
            deconv.alphas.data.fill_(-10)
            deconv.alphas.data[arch.op_indices[i]] = 10

    subnet = OptimizedNetwork(supernet_model).to(device)
    subnet.eval()
    meter = AverageMeter()
    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = subnet(images)
        meter.update(get_iou_score(outputs, labels), images.size(0))

    # Restore supernet alpha state
    for deconv, state in zip(deconvs, saved):
        if hasattr(deconv, 'alphas_op'):
            deconv.alphas_op.data.copy_(state[0])
            deconv.alphas_width.data.copy_(state[1])
        else:
            deconv.alphas.data.copy_(state[0])

    return meter.avg


def search_architecture(args, dataset):
    # Set device (use local_rank for DDP if available)
    local_rank = getattr(args, 'local_rank', None)
    device = set_device(args.gpu_idx, local_rank=local_rank)

    # Create SuperNet with specified search space
    search_space = getattr(args, 'search_space', 'basic')
    model = SuperNet(n_class=2, search_space=search_space)
    op_names, width_mults = _describe_search_space(search_space)
    print(f"Search space: {search_space}")
    print(f"  - {len(op_names)} operations: {', '.join(op_names)}")
    if search_space in ('extended', 'industry'):
        print(f"  - {len(width_mults)} width multipliers: {list(width_mults)}")
        print(f"  - Total: {calculate_search_space_size(op_names, list(width_mults)):,} architectures")
    else:
        print(f"  - Total: {calculate_search_space_size(op_names, [1.0]):,} architectures")

    model = model.to(device)

    # Distributed training setup
    if hasattr(args, 'distributed') and args.distributed:
        # Wrap with DDP
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False
        )
        if args.rank == 0:
            print(f"Using DDP with {args.world_size} GPUs")
    # Fallback to DataParallel (legacy support)
    elif torch.cuda.is_available() and len(args.gpu_idx) >= 2:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_idx)
        print(f"Using DataParallel with GPUs: {args.gpu_idx}")
    else:
        if device.type == "cuda":
            print(f"Using single GPU: cuda:{args.gpu_idx[0]}")
        else:
            print("Using CPU")

    loss = nn.CrossEntropyLoss()

    # Get alpha parameters (handles both basic and extended search spaces)
    alphas_params = [
        param for name, param in model.named_parameters() if "alpha" in name.lower()
    ]
    weight_params = [
        param
        for param in model.parameters()
        if not check_tensor_in_list(param, alphas_params)
    ]

    optimizer_alpha = torch.optim.Adam(alphas_params, lr=args.alpha_lr)
    optimizer_weight = torch.optim.Adam(weight_params, lr=args.weight_lr)

    data_name = args.data if isinstance(args.data, str) else "_".join(args.data)
    timestamp = str(datetime.now().date()) + "_" + datetime.now().strftime("%H_%M_%S")
    args.save_dir = f"./hyundai/checkpoints/{args.mode}_{data_name}_seed{args.seed}_lambda{args.flops_lambda}_{search_space}/{timestamp}/"

    train_architecture(
        args,
        model,
        dataset,
        loss,
        optimizer_alpha,
        optimizer_weight,
    )

    return model


def search_architecture_linas(args, dataset):
    """
    LINAS: Latency-aware architecture search.

    This function performs NAS with latency-aware multi-objective optimization
    instead of FLOPs-based optimization.
    """
    # Set device (use local_rank for DDP if available)
    local_rank = getattr(args, 'local_rank', None)
    device = set_device(args.gpu_idx, local_rank=local_rank)

    # Create SuperNet (width-aware search space required for latency)
    search_space = getattr(args, 'search_space', 'industry')
    if search_space == 'basic':
        raise ValueError("LINAS requires width-aware search space ('extended' or 'industry').")
    model = SuperNet(n_class=2, search_space=search_space)
    op_names, width_mults = _describe_search_space(search_space)
    print(f"LINAS Search Space: {search_space}")
    print(f"  - {len(op_names)} operations: {', '.join(op_names)}")
    print(f"  - {len(width_mults)} width multipliers: {list(width_mults)}")
    print(f"  - Total: {calculate_search_space_size(op_names, list(width_mults)):,} architectures")

    model = model.to(device)

    # Distributed training setup
    if hasattr(args, 'distributed') and args.distributed:
        # Wrap with DDP
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False
        )
        if args.rank == 0:
            print(f"Using DDP with {args.world_size} GPUs")
    # Fallback to DataParallel (legacy support)
    elif torch.cuda.is_available() and len(args.gpu_idx) >= 2:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_idx)
        print(f"Using DataParallel with GPUs: {args.gpu_idx}")
    else:
        if device.type == "cuda":
            print(f"Using single GPU: cuda:{args.gpu_idx[0]}")
        else:
            print("Using CPU")

    loss = nn.CrossEntropyLoss()

    # Get alpha and weight parameters
    alphas_params = [
        param for name, param in model.named_parameters() if "alpha" in name.lower()
    ]
    weight_params = [
        param
        for param in model.parameters()
        if not check_tensor_in_list(param, alphas_params)
    ]

    optimizer_alpha = torch.optim.Adam(alphas_params, lr=args.alpha_lr)
    optimizer_weight = torch.optim.Adam(weight_params, lr=args.weight_lr)

    # Setup save directory
    data_name = args.data if isinstance(args.data, str) else "_".join(args.data)
    timestamp = str(datetime.now().date()) + "_" + datetime.now().strftime("%H_%M_%S")
    target_lat = getattr(args, 'target_latency', None) or 'min'
    args.save_dir = f"./hyundai/checkpoints/linas_{data_name}_seed{args.seed}_lat{target_lat}_lambda{args.latency_lambda}/{timestamp}/"

    # Load latency predictor or LUT
    latency_predictor = None
    latency_lut = None
    hardware_targets = None

    # Option 1: Multi-hardware predictor
    predictor_path = getattr(args, 'predictor_path', None)
    if predictor_path and Path(predictor_path).exists():
        print(f"Loading latency predictor from: {predictor_path}")
        from latency import CrossHardwareLatencyPredictor
        latency_predictor = CrossHardwareLatencyPredictor(
            num_ops=len(op_names),
            num_widths=len(width_mults)
        )
        latency_predictor.load_state_dict(torch.load(predictor_path, map_location=device))
        latency_predictor = latency_predictor.to(device)
        latency_predictor.eval()

        # Get hardware targets
        hardware_targets = getattr(args, 'hardware_targets_dict', None)
        if hardware_targets:
            print(f"Hardware targets: {hardware_targets}")

    # Option 2: Single-hardware LUT (explicit path)
    lut_path = getattr(args, 'lut_path', None)
    if lut_path and Path(lut_path).exists() and latency_predictor is None:
        print(f"Loading latency LUT from: {lut_path}")
        from latency import LatencyLUT
        latency_lut = LatencyLUT(lut_path)
        print(f"LUT hardware: {latency_lut.hardware_name}")

    # Option 3: Single-hardware LUT (directory + primary hardware)
    if latency_lut is None and latency_predictor is None:
        lut_dir = getattr(args, 'lut_dir', None)
        if lut_dir:
            lut_dir_path = Path(lut_dir)
            if lut_dir_path.exists():
                primary_hw = getattr(args, 'primary_hardware', None)
                candidate = None
                if primary_hw:
                    candidate = lut_dir_path / f"lut_{primary_hw.lower()}.json"
                    if not candidate.exists():
                        print(f"Warning: LUT not found for primary_hardware={primary_hw}: {candidate}")
                        candidate = None

                if candidate is None:
                    lut_files = sorted(lut_dir_path.glob("lut_*.json"))
                    if lut_files:
                        candidate = lut_files[0]
                        print(f"Using first available LUT in {lut_dir_path}: {candidate.name}")

                if candidate is not None and candidate.exists():
                    print(f"Loading latency LUT from: {candidate}")
                    from latency import LatencyLUT
                    latency_lut = LatencyLUT(str(candidate))
                    print(f"LUT hardware: {latency_lut.hardware_name}")

    # Print optimization mode
    if latency_predictor is not None:
        print("\nOptimization: Multi-hardware latency (predictor-based)")
    elif latency_lut is not None:
        target = getattr(args, 'target_latency', None)
        if target:
            print(f"\nOptimization: Single-hardware latency (target: {target}ms)")
        else:
            print("\nOptimization: Single-hardware latency (minimize)")
    else:
        print("\nWarning: No latency info provided, falling back to FLOPs")

    # Train with latency-aware optimization
    train_architecture_with_latency(
        args,
        model,
        dataset,
        loss,
        optimizer_alpha,
        optimizer_weight,
        latency_predictor=latency_predictor,
        latency_lut=latency_lut,
        hardware_targets=hardware_targets,
    )

    return model


def train_searched_model(args, opt_model, dataset):
    # Set device (use local_rank for DDP if available)
    local_rank = getattr(args, 'local_rank', None)
    device = set_device(args.gpu_idx, local_rank=local_rank)

    # Extract architecture description before creating OptimizedNetwork
    if isinstance(opt_model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        supernet = opt_model.module
    else:
        supernet = opt_model

    # Use validation-selected supernet weights if checkpoint exists.
    best_ckpt = Path(args.save_dir) / 'best_architecture.pt'
    if hasattr(args, 'distributed') and args.distributed:
        torch.distributed.barrier()
    if best_ckpt.exists():
        state_dict = torch.load(best_ckpt, map_location=device)
        supernet.load_state_dict(state_dict)

    # Log final selected architecture
    arch_desc = supernet.get_arch_description()
    print("\n" + "=" * 60)
    print("Final Selected Architecture:")
    print("=" * 60)
    for layer_desc in arch_desc:
        print(f"  {layer_desc}")
    print("=" * 60 + "\n")

    # Log to wandb as a summary (appears in Overview tab)
    arch_text = "\n".join(arch_desc)
    if _wandb_active(args):
        if not is_minimal_metrics_enabled():
            wandb.run.summary['Selected Architecture'] = arch_text

            # Also log as individual metrics for easier filtering
            for i, layer_desc in enumerate(arch_desc, 1):
                wandb.run.summary[f'Architecture/Layer{i}'] = layer_desc

        # Log alpha values for detailed analysis
        alphas = supernet.get_alphas()
        if supernet.search_space in ('extended', 'industry'):
            for i, alpha_dict in enumerate(alphas, 1):
                wandb.log({
                    f'Final_Alphas/deconv{i}_op': alpha_dict['op'],
                    f'Final_Alphas/deconv{i}_width': alpha_dict['width']
                })
        else:
            for i, alpha_list in enumerate(alphas, 1):
                wandb.log({f'Final_Alphas/deconv{i}': alpha_list})

    opt_model = OptimizedNetwork(opt_model)
    opt_model = opt_model.to(device)

    # Distributed training setup
    if hasattr(args, 'distributed') and args.distributed:
        # Wrap with DDP
        opt_model = torch.nn.parallel.DistributedDataParallel(
            opt_model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False  # OptimizedNetwork uses all parameters
        )
        if args.rank == 0:
            print(f"Using DDP with {args.world_size} GPUs")
    # Fallback to DataParallel (legacy support)
    elif torch.cuda.is_available() and len(args.gpu_idx) >= 2:
        opt_model = torch.nn.DataParallel(opt_model, device_ids=args.gpu_idx)
        print(f"Using DataParallel with GPUs: {args.gpu_idx}")
    else:
        if device.type == "cuda":
            print(f"Using single GPU: cuda:{args.gpu_idx[0]}")
        else:
            print("Using CPU")

    # Measure FLOPs and Parameters for Pareto analysis
    gflops, params_m = get_model_complexity(opt_model, input_size=(1, 3, args.resize, args.resize), device=device)
    print(f"OptimizedNetwork - FLOPs: {gflops:.4f} GFLOPs, Parameters: {params_m:.4f} M")
    if _wandb_active(args):
        wandb.log({
            'Model/FLOPs (GFLOPs)': gflops,
            'Model/Parameters (M)': params_m
        })
        if not is_minimal_metrics_enabled():
            wandb.run.summary['Model/Final FLOPs (GFLOPs)'] = gflops
            wandb.run.summary['Model/Final Parameters (M)'] = params_m

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(opt_model.parameters(), lr=args.opt_lr)

    train_samplenet(args,
        opt_model,
        dataset,
        loss,
        optimizer
    )


def discover_pareto_architectures(args, model, dataset):
    """
    RF-DETR style: Discover accuracy-latency Pareto curve from trained supernet.

    This function:
    1. Takes a trained supernet
    2. Samples thousands of architectures
    3. Evaluates each with weight-sharing (no re-training)
    4. Computes latency using LUT for each hardware
    5. Extracts Pareto-optimal architectures
    6. Selects best architecture for each target latency

    Args:
        args: Arguments containing lut_dir, target_latency, etc.
        model: Trained supernet
        dataset: Dataset for evaluation
    """
    # Set device (use local_rank for DDP if available)
    local_rank = getattr(args, 'local_rank', None)
    device = set_device(args.gpu_idx, local_rank=local_rank)

    # Load LUTs for all hardware
    lut_dir = Path(getattr(args, 'lut_dir', './hyundai/latency/luts'))
    hardware_list = getattr(args, 'hardware_list', ['A6000', 'RTX3090', 'RTX4090', 'JetsonOrin'])

    from latency import LatencyLUT
    luts = {}
    for hw in hardware_list:
        lut_path = lut_dir / f'lut_{hw.lower()}.json'
        if lut_path.exists():
            luts[hw] = LatencyLUT(str(lut_path))
            print(f"Loaded LUT for {hw}")
        else:
            print(f"Warning: LUT not found for {hw}: {lut_path}")

    if not luts:
        raise RuntimeError("No LUT files found. Run measure_latency.sh first.")

    # Create validation loader
    if isinstance(dataset, dict):
        val_dataset = dataset.get('valid') or dataset.get('val')
        if val_dataset is None:
            raise KeyError("Expected dataset dict to have 'valid' or 'val' key.")
    elif isinstance(dataset, (list, tuple)):
        if len(dataset) < 2:
            raise ValueError("Expected dataset tuple/list with at least (train, val, ...).")
        val_dataset = dataset[1]
    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset).__name__}")

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create Pareto searcher
    searcher = ParetoSearcher(model, luts)

    # Discover Pareto curve
    num_samples = getattr(args, 'pareto_samples', 1000)
    eval_subset = getattr(args, 'pareto_eval_subset', 100)

    print("\n" + "=" * 70)
    print("PARETO-BASED ARCHITECTURE DISCOVERY (RF-DETR Style)")
    print("=" * 70)
    print(f"  Sampling: {num_samples} architectures")
    print(f"  Evaluating: {eval_subset} architectures (weight-sharing)")
    print(f"  Hardware: {list(luts.keys())}")
    print("=" * 70 + "\n")

    pareto_front = searcher.discover_pareto_curve(
        val_loader, device,
        num_samples=num_samples,
        eval_subset=eval_subset,
        strategy='mixed'
    )

    # Print summary
    print_pareto_summary(pareto_front)

    # Save results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    searcher.save_results(str(save_dir / 'pareto_results.json'))

    # Select best architectures for each hardware's target latency
    hardware_targets = getattr(args, 'hardware_targets_dict', {
        'A6000': 50,
        'RTX3090': 60,
        'RTX4090': 40,
        'JetsonOrin': 95
    })

    print("\n" + "=" * 70)
    print("SELECTED ARCHITECTURES FOR TARGET LATENCIES")
    print("=" * 70)

    refine_topk = max(1, int(getattr(args, 'pareto_refine_topk', 1)))
    if refine_topk > 1:
        print(f"Refinement: evaluating top-{refine_topk} Pareto candidates as extracted subnets")

    selected_archs = {}
    for hw, target_lat in hardware_targets.items():
        if hw in luts:
            pareto_candidates = list(searcher.pareto_front.get(hw, []))
            valid_candidates = [
                cand for cand in pareto_candidates
                if cand.latencies.get(hw, float('inf')) <= target_lat
            ]
            if not valid_candidates:
                valid_candidates = pareto_candidates[:1]

            valid_candidates = sorted(valid_candidates, key=lambda a: a.accuracy, reverse=True)
            shortlist = valid_candidates[:refine_topk]

            arch = None
            best_refined = -float('inf')
            for cand in shortlist:
                refined_val_iou = _evaluate_extracted_subnet(model, cand, val_loader, device)
                if refined_val_iou > best_refined:
                    best_refined = refined_val_iou
                    arch = cand

            if arch:
                selected_archs[hw] = arch
                actual_lat = arch.latencies.get(hw, 0)
                print(f"\n{hw} (target: {target_lat}ms):")
                print(f"  Accuracy: {arch.accuracy:.4f}")
                print(f"  Latency:  {actual_lat:.2f}ms")
                print(f"  Refined Val mIoU: {best_refined:.4f}")
                print(f"  Ops:      {arch.op_indices}")
                print(f"  Widths:   {arch.width_indices}")

                # Log to wandb (only rank 0)
                if _wandb_active(args):
                    wandb.log({
                        f'Pareto/{hw}/accuracy': arch.accuracy,
                        f'Pareto/{hw}/refined_val_mIoU': best_refined,
                        f'Pareto/{hw}/latency_ms': actual_lat,
                        f'Pareto/{hw}/target_ms': target_lat,
                        f'Pareto/{hw}/meets_target': actual_lat <= target_lat
                    })

    print("\n" + "=" * 70)

    # Save selected architectures
    selected_dict = {
        hw: arch.to_dict() for hw, arch in selected_archs.items()
    }
    with open(save_dir / 'selected_architectures.json', 'w') as f:
        json.dump(selected_dict, f, indent=2)

    return selected_archs, pareto_front


def search_and_discover_pareto(args, dataset):
    """
    Two-phase approach:
    1. Train supernet with latency-aware NAS
    2. Discover Pareto-optimal architectures

    This is the main entry point for RF-DETR style NAS.
    """
    print("\n" + "=" * 70)
    print("PHASE 1: SUPERNET TRAINING")
    print("=" * 70 + "\n")

    # Phase 1: Train supernet
    model = search_architecture_linas(args, dataset)

    print("\n" + "=" * 70)
    print("PHASE 2: PARETO DISCOVERY")
    print("=" * 70 + "\n")

    # Phase 2: Discover Pareto curve
    selected_archs, pareto_front = discover_pareto_architectures(args, model, dataset)

    return model, selected_archs, pareto_front
