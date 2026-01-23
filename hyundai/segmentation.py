import torch
import torch.nn as nn
from datetime import datetime

import wandb
from utils.utils import set_device, check_tensor_in_list, get_model_complexity
from nas.supernet_dense import SuperNet, OptimizedNetwork
from nas.train_supernet import train_architecture
from nas.train_samplenet import train_samplenet

def search_architecture(args, dataset):
    device = set_device(args.gpu_idx)

    # Create SuperNet with specified search space
    search_space = getattr(args, 'search_space', 'basic')
    model = SuperNet(n_class=2, search_space=search_space)
    print(f"Search space: {search_space}")
    if search_space == 'extended':
        print("  - 5 operations (Conv3x3, Conv5x5, Conv7x7, DWSep3x3, DWSep5x5)")
        print("  - 3 width multipliers (0.5x, 0.75x, 1.0x)")
        print("  - Total: 5^5 x 3^5 = 759,375 architectures")
    else:
        print("  - 5 operations (Conv3x3, Conv5x5, Conv7x7, DWSep3x3, DWSep5x5)")
        print("  - Total: 5^5 = 3,125 architectures")

    model = model.to(device)
    use_dp = torch.cuda.is_available() and len(args.gpu_idx) >= 2
    if use_dp:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_idx)
        print(f"Using multiple GPUs: {args.gpu_idx}")
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

def train_searched_model(args, opt_model, dataset):
    device = set_device(args.gpu_idx)

    # Extract architecture description before creating OptimizedNetwork
    if isinstance(opt_model, torch.nn.DataParallel):
        supernet = opt_model.module
    else:
        supernet = opt_model

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
    wandb.run.summary['Selected Architecture'] = arch_text

    # Also log as individual metrics for easier filtering
    for i, layer_desc in enumerate(arch_desc, 1):
        wandb.run.summary[f'Architecture/Layer{i}'] = layer_desc

    # Log alpha values for detailed analysis
    alphas = supernet.get_alphas()
    if supernet.search_space == 'extended':
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
    use_dp = torch.cuda.is_available() and len(args.gpu_idx) >= 2
    if use_dp:
        opt_model = torch.nn.DataParallel(opt_model, device_ids=args.gpu_idx)
        print(f"Using multiple GPUs: {args.gpu_idx}")
    else:
        if device.type == "cuda":
            print(f"Using single GPU: cuda:{args.gpu_idx[0]}")
        else:
            print("Using CPU")

    # Measure FLOPs and Parameters for Pareto analysis
    gflops, params_m = get_model_complexity(opt_model, input_size=(1, 3, args.resize, args.resize), device=device)
    print(f"OptimizedNetwork - FLOPs: {gflops:.4f} GFLOPs, Parameters: {params_m:.4f} M")
    wandb.log({
        'Model/FLOPs (GFLOPs)': gflops,
        'Model/Parameters (M)': params_m
    })
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
