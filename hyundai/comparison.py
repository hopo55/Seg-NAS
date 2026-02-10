import warnings
from datetime import datetime

import wandb

from baselines.comparison import run_comparison
from preprocessing import get_dataset
from utils.argument import get_args
from utils.utils import set_seed

# Suppress PyTorch internal deprecation warnings
warnings.filterwarnings("ignore", message=".*_all_gather_base.*")


def main():
    import setproctitle
    setproctitle.setproctitle('hyundai/comparison')

    args = get_args(include_comparison_args=True)
    set_seed(args.seed)

    data_name = args.data if isinstance(args.data, str) else "_".join(args.data)
    timestamp = str(datetime.now().date()) + "_" + datetime.now().strftime("%H_%M_%S")
    args.log_dir = f"./hyundai/logs/comparison_{data_name}_seed{args.seed}/{timestamp}/"

    run_name = f"hyundai_comparison_seed{args.seed}"
    wandb.init(config=args, project="Seg-NAS", entity="hopo55", name=run_name)

    dataset = get_dataset(args)
    run_comparison(args, dataset)

    wandb.finish()


if __name__ == "__main__":
    main()
