import torch
import torch.nn as nn
from datetime import datetime

from utils.utils import set_device, check_tensor_in_list
from nas.supernet_dense import SuperNet, OptimizedNetwork
from nas.train_supernet import train_architecture
from nas.train_samplenet import train_samplenet

def search_architecture(args, dataset):
    device = set_device(args.gpu_idx)

    model = SuperNet(n_class=2)
    if len(args.gpu_idx) >= 2:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_idx).to(device)
        print(f"Using multiple GPUs: {args.gpu_idx}")
    else:
        model.to(device)
        print(f"Using single GPU: cuda:{args.gpu_idx[0]}")

    loss = nn.CrossEntropyLoss()

    alphas_params = [
        param for name, param in model.named_parameters() if "alphas" in name
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
    args.save_dir = f"./hyundai/checkpoints/{args.mode}_{data_name}_seed{args.seed}/{timestamp}/"

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
    
    opt_model = OptimizedNetwork(opt_model)
    if len(args.gpu_idx) >= 2:
        opt_model = torch.nn.DataParallel(opt_model, device_ids=args.gpu_idx).to(device)
        print(f"Using multiple GPUs: {args.gpu_idx}")
    else:
        opt_model.to(device)
        print(f"Using single GPU: cuda:{args.gpu_idx[0]}")

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(opt_model.parameters(), lr=args.opt_lr)

    train_samplenet(args,
        opt_model,
        dataset,
        loss,
        optimizer
    )