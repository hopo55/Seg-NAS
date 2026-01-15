import wandb

from datetime import datetime
from utils.argument import get_args
from utils.utils import set_seed

from preprocessing import get_roi, get_dataset
from segmentation import search_architecture, train_searched_model
from comparison import run_comparison
from test import test_model


def main():
    import setproctitle
    setproctitle.setproctitle('hyundai/hspark')
    args = get_args()
    set_seed(args.seed)

    data_name = args.data if isinstance(args.data, str) else "_".join(args.data)
    timestamp = str(datetime.now().date()) + "_" + datetime.now().strftime("%H_%M_%S")
    args.log_dir = f"./hyundai/logs/{args.mode}_{data_name}_seed{args.seed}/{timestamp}/"

    wandb.init(config=args, project="Seg-NAS", entity="hopo55", name=f"hyundai_seed{args.seed}")
    
    # Data Preprocessing
    # get_roi(args.data)
    dataset = get_dataset(args)
    
    if args.mode in ['nas', 'ind', 'zero']:
        # Search Architecture
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

    