import json
import wandb

from datetime import datetime
from utils.argument import get_args
from utils.utils import set_seed

from preprocessing import get_roi, get_dataset
from segmentation import search_architecture, search_architecture_linas, train_searched_model
from comparison import run_comparison
from test import test_model


def main():
    import setproctitle
    setproctitle.setproctitle('hyundai/hspark')
    args = get_args()
    set_seed(args.seed)

    data_name = args.data if isinstance(args.data, str) else "_".join(args.data)
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

    args.log_dir = f"./hyundai/logs/{args.mode}_{data_name}_seed{args.seed}_{opt_mode}{target_val}_lambda{lambda_val}_{search_space}/{timestamp}/"

    # Include optimization mode in run name
    run_name = f"hyundai_seed{args.seed}_{opt_mode}_λ{lambda_val}_{search_space}"
    wandb.init(config=args, project="Seg-NAS", entity="hopo55", name=run_name)

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

    if args.mode in ['nas', 'ind', 'zero']:
        # Search Architecture
        use_latency = getattr(args, 'use_latency', False)

        if use_latency:
            print("\n" + "=" * 60)
            print("LINAS: Latency-aware Industrial NAS")
            print("=" * 60)
            searched_model = search_architecture_linas(args, dataset)
        else:
            print("\n" + "=" * 60)
            print("FLOPs-based NAS")
            print("=" * 60)
            searched_model = search_architecture(args, dataset)

        # Train and Test of the Optimized Architecture
        train_searched_model(args, searched_model, dataset)

        # Run comparison with baseline models if enabled
        if args.comparison:
            print("\n" + "=" * 60)
            print("Running Baseline Comparison")
            print("=" * 60)
            run_comparison(args, dataset)

        wandb.finish()
    elif args.mode == 'hot':
        # Model Testing
        test_model(args, dataset)
        wandb.finish()

if __name__ == "__main__":
    main()

    