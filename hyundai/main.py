import json
import warnings
import wandb

# Suppress PyTorch internal deprecation warnings
warnings.filterwarnings("ignore", message=".*_all_gather_base.*")
# PS/CALOFA: find_unused_parameters=True triggers false positives during warmup/early phases
warnings.filterwarnings("ignore", message=".*find_unused_parameters=True was specified.*")

from datetime import datetime
from utils.argument import get_args
from utils.utils import set_seed

from preprocessing import get_roi, get_dataset
from segmentation import (
    search_architecture,
    search_architecture_linas,
    search_architecture_linas_ps,
    train_searched_model,
    search_and_discover_pareto,
    discover_pareto_architectures
)
from test import test_model


def main():
    import setproctitle
    setproctitle.setproctitle('hyundai/hspark')
    args = get_args()

    # Initialize distributed training if launched with torchrun
    from utils.distributed import init_distributed_mode
    init_distributed_mode(args)

    set_seed(args.seed)

    data_name = args.data if isinstance(args.data, str) else "_".join(args.data)
    dataset_profile = getattr(args, 'dataset_profile', 'hyundai')
    timestamp = str(datetime.now().date()) + "_" + datetime.now().strftime("%H_%M_%S")
    search_space = getattr(args, 'search_space', 'basic')

    # Determine optimization mode for naming
    use_latency = getattr(args, 'use_latency', False)
    if use_latency:
        opt_mode = 'latency'
        target_val = getattr(args, 'target_latency', None) or 'min'
        lambda_val = args.latency_lambda
    else:
        opt_mode = 'flops'
        target_val = getattr(args, 'target_flops', None) or 'min'
        lambda_val = args.flops_lambda

    encoder_name = getattr(args, 'encoder_name', 'densenet121')
    args.log_dir = (
        f"./hyundai/logs/{args.mode}_{dataset_profile}_{data_name}_seed{args.seed}_"
        f"{opt_mode}{target_val}_lambda{lambda_val}_{search_space}_{encoder_name}/{timestamp}/"
    )

    # Include optimization mode in run name
    run_name = (
        f"hyundai_{dataset_profile}_seed{args.seed}_{opt_mode}_λ{lambda_val}_"
        f"{search_space}_{encoder_name}"
    )

    # Initialize wandb only on rank 0
    if not hasattr(args, 'rank') or args.rank == 0:
        wandb.init(config=args, project="Seg-NAS", entity="hopo55", name=run_name)
    else:
        import os
        os.environ['WANDB_MODE'] = 'disabled'

    # Parse hardware targets if provided
    if hasattr(args, 'hardware_targets') and args.hardware_targets:
        try:
            args.hardware_targets_dict = json.loads(args.hardware_targets)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse hardware_targets: {args.hardware_targets}")
            args.hardware_targets_dict = None
    else:
        args.hardware_targets_dict = None
    
    # Data Preprocessing
    # get_roi(args.data)
    dataset = get_dataset(args)

    if args.mode == 'pareto':
        # RF-DETR style: Train supernet + Discover Pareto curve
        print("\n" + "=" * 60)
        print("PARETO-BASED NAS (RF-DETR Style)")
        print("=" * 60)

        model, selected_archs, pareto_front = search_and_discover_pareto(args, dataset)

        # Train the best architecture for primary hardware
        primary_hw = getattr(args, 'primary_hardware', 'JetsonOrin')
        if primary_hw in selected_archs:
            best_arch = selected_archs[primary_hw]
            print(f"\nTraining selected architecture for {primary_hw}...")

            # Set the supernet to the selected architecture before extracting
            if hasattr(model, 'module'):
                module = model.module
            else:
                module = model

            for i, deconv in enumerate([module.deconv1, module.deconv2,
                                        module.deconv3, module.deconv4, module.deconv5]):
                if hasattr(deconv, 'alphas_op'):
                    deconv.alphas_op.data.fill_(-10)
                    deconv.alphas_op.data[best_arch.op_indices[i]] = 10
                    deconv.alphas_width.data.fill_(-10)
                    deconv.alphas_width.data[best_arch.width_indices[i]] = 10

            train_searched_model(args, model, dataset)

        if not hasattr(args, 'rank') or args.rank == 0:
            wandb.finish()

    elif args.mode in ['nas', 'ind', 'zero']:
        # Search Architecture
        use_latency = getattr(args, 'use_latency', False)

        if use_latency:
            print("\n" + "=" * 60)
            print("LINAS: Latency-aware Industrial NAS")
            print("=" * 60)
            use_ps = getattr(args, 'use_progressive_shrinking', False)
            if use_ps:
                print("  Progressive Shrinking: ENABLED")
                searched_model = search_architecture_linas_ps(args, dataset)
            else:
                searched_model = search_architecture_linas(args, dataset)
        else:
            print("\n" + "=" * 60)
            print("FLOPs-based NAS")
            print("=" * 60)
            searched_model = search_architecture(args, dataset)

        # Train and Test of the Optimized Architecture
        train_searched_model(args, searched_model, dataset)

        if not hasattr(args, 'rank') or args.rank == 0:
            wandb.finish()

    elif args.mode == 'hot':
        # Model Testing
        test_model(args, dataset)
        if not hasattr(args, 'rank') or args.rank == 0:
            wandb.finish()

if __name__ == "__main__":
    main()

    
