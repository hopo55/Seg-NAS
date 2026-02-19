import warnings
from datetime import datetime

import torch
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
    # set_seed disables cuDNN to work around AMP+non-square resolution issues in NAS
    # training, but comparison uses plain fp32 with no AMP — re-enable so standard
    # cuDNN kernels are used instead of the fallback ATen kernels that cause SIGFPE
    # when processing large (480×640) tensors with DataParallel.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False  # deterministic=True triggers cuDNN internal graph capture

    data_name = args.data if isinstance(args.data, str) else "_".join(args.data)
    timestamp = str(datetime.now().date()) + "_" + datetime.now().strftime("%H_%M_%S")
    args.log_dir = f"./hyundai/logs/comparison_{data_name}_seed{args.seed}/{timestamp}/"

    run_name = f"hyundai_comparison_seed{args.seed}"
    wandb_config = vars(args).copy()
    wandb_config["mode"] = "comparison"
    wandb_config["encoder_name"] = ",".join(args.baseline_models)
    wandb.init(config=wandb_config, project="Seg-NAS", entity="hopo55", name=run_name)

    dataset = get_dataset(args)
    run_comparison(args, dataset)

    wandb.finish()


if __name__ == "__main__":
    main()
