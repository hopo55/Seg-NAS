import argparse

DEFAULT_BASELINE_MODELS = ['autopatch', 'realtimeseg', 'unet', 'deeplabv3plus']

def add_comparison_args(parser):
    """Register CLI arguments for baseline comparison experiments."""
    parser.add_argument('--baseline_models', type=str, nargs='+',
                        default=DEFAULT_BASELINE_MODELS.copy(),
                        choices=DEFAULT_BASELINE_MODELS,
                        help='Baseline models to compare (default: all)')


def get_args(include_comparison_args=False):
    parser = argparse.ArgumentParser()

    # Environment Argument
    parser.add_argument('--seed', type=int, default=42, help="Seed value to ensure reproducibility.")
    # Preprocessing Argument
    parser.add_argument('--mode', type=str, default='nas', choices=['nas', 'ind', 'zero', 'hot', 'pareto'],
                        help='(nas): train and test the optimized model\n'
                            '(ind): train and test individual models\n'
                            '(zero): training for zero-shot testing\n'
                            '(hot): train and test a model on the hot-stamping dataset\n'
                            '(pareto): RF-DETR style Pareto-based NAS')
    parser.add_argument('--data', type=str, default=['all'], nargs='+', 
                        choices=['all', 'ce', 'df', 'gn7norm', 'gn7pano'],
                        help='select one or more datasets')
    parser.add_argument('--data_dir', type=str, default='./dataset/image', 
                        help='select one or more datasets')
    parser.add_argument('--label_dir_name', type=str, default='target',
                        help='Label directory name paired with data_dir (e.g., target, target_ori).')
    parser.add_argument('--resize', type=int, default=128,
                        help='Resize dimension as a single integer (e.g., 128).')
    parser.add_argument('--ratios', type=float, default=0.2, 
                        help='Ratios for splitting train and test sets (default: 0.2)')
    # Training Argument
    parser.add_argument('--gpu_idx', type=int, default=[0], nargs='+',
                        help='use single GPU(e.g., --gpu_idx 0), or use multiple GPUs(e.g., --gpu_idx 0 1). Ignored when using torchrun for DDP.')
    parser.add_argument('--alpha_lr', type=float, default=0.01, 
                        help='Learning rate for alpha used for architecture search (default: 0.01)')
    parser.add_argument('--train_size', type=int, default=64, 
                        help='Batch size for the training and validation (default: 64)')
    parser.add_argument('--test_size', type=int, default=32, 
                        help='Batch size for the test splits (default: 32)')
    parser.add_argument('--weight_lr', type=float, default=0.001, 
                        help='Learning rate for network weight parameters (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=2e-4, 
                        help='Weight decay for network weight parameters (default: 2e-4)')
    parser.add_argument('--warmup_epochs', type=int, default=5, 
                        help='Number of warmup epochs before main training starts (default: 5)')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--clip_grad', type=float, default=5.0, 
                        help='Maximum gradient clipping value (default: 5.0)')
    parser.add_argument('--opt_lr', type=float, default=5e-4,
                        help='Learning rate for optimzied network (default: 5e-4)')
    parser.add_argument('--flops_lambda', type=float, default=1.0,
                        help='FLOPs penalty weight for multi-objective NAS (default: 1.0)')
    parser.add_argument('--target_flops', type=float, default=None,
                        help='Target FLOPs (GFLOPs) for architecture search. If set, loss = |expected - target|. '
                             'If None, uses simple penalty: loss = expected_flops.')
    parser.add_argument('--flops_norm_base', type=float, default=None,
                        help='Normalize expected FLOPs by this base (GFLOPs). If None, use initial expected FLOPs.')
    parser.add_argument('--search_space', type=str, default='basic',
                        choices=['basic', 'extended', 'industry'],
                        help='Search space type: basic (5 ops, 3125 archs), '
                             'extended (5 ops x 3 widths, 759375 archs), '
                             'or industry (7 ops x 3 widths, 4084101 archs)')
    parser.add_argument('--train_val_split', type=float, default=0.8,
                        help='Train ratio within train+val pool after test split '
                             '(e.g., 0.8 => 80%% train, 20%% val)')

    # LINAS: Latency-aware NAS arguments
    parser.add_argument('--use_latency', action='store_true',
                        help='Use latency-aware optimization instead of FLOPs')
    parser.add_argument('--latency_lambda', type=float, default=1.0,
                        help='Latency penalty weight for multi-objective NAS (default: 1.0)')
    parser.add_argument('--target_latency', type=float, default=None,
                        help='Target latency (ms) for architecture search. If set, loss = |predicted - target|.')
    parser.add_argument('--lut_path', type=str, default=None,
                        help='Path to latency LUT JSON file for current hardware')
    parser.add_argument('--predictor_path', type=str, default=None,
                        help='Path to trained latency predictor checkpoint')
    parser.add_argument('--hardware_targets', type=str, default=None,
                        help='JSON string of hardware targets, e.g., \'{"A6000": 50, "JetsonOrin": 100}\'')
    parser.add_argument('--primary_hardware', type=str, default='A6000',
                        choices=['A6000', 'RTX3090', 'RTX4090', 'JetsonOrin',
                                 'RaspberryPi5', 'Odroid'],
                        help='Primary hardware for single-hardware latency optimization')
    parser.add_argument('--lut_dir', type=str, default='./hyundai/latency/luts',
                        help='Directory containing LUT files for all hardware')
    parser.add_argument('--hardware_list', type=str, nargs='+',
                        default=['A6000', 'RTX3090', 'RTX4090', 'JetsonOrin',
                                 'RaspberryPi5', 'Odroid'],
                        help='List of hardware to consider for Pareto search')

    # Pareto search arguments (RF-DETR style)
    parser.add_argument('--pareto_samples', type=int, default=1000,
                        help='Number of architectures to sample for Pareto discovery (default: 1000)')
    parser.add_argument('--pareto_eval_subset', type=int, default=100,
                        help='Number of architectures to actually evaluate with weight-sharing (default: 100)')
    parser.add_argument('--pareto_refine_topk', type=int, default=5,
                        help='Top-k Pareto candidates to re-evaluate as extracted subnet for final selection (default: 5)')

    # Comparison arguments (only for comparison entrypoint)
    if include_comparison_args:
        add_comparison_args(parser)
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save trained model checkpoints (default: ./checkpoints)')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help="Directory to save TensorBoard logs")

    # Distributed training arguments
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='Distributed backend (default: nccl)')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='URL used to set up distributed training (default: env://)')

    # parser.add_argument('--resume', action='store_true',
    #                     help='Resume training from the last checkpoint if available')

    # args = parser.parse_args()
    args, _ = parser.parse_known_args()

    if args.mode == 'hot':
        hotstamping_args(parser)

    args = parser.parse_args()

    return args

# Hot-Stamping Arguments
def hotstamping_args(parser):
    parser.add_argument('--model_dir', type=str, required=False,
                        help='Path to the best_model.pt file')
