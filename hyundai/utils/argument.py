import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # Environment Argument
    parser.add_argument('--seed', type=int, default=42, help="Seed value to ensure reproducibility.")
    # Preprocessing Argument
    parser.add_argument('--mode', type=str, default='nas', choices=['nas', 'ind', 'zero', 'hot'],
                        help='(nas): train and test the optimized model\n'
                            '(ind): train and test individual models\n'
                            '(zero): training for zero-shot testing\n'
                            '(hot): train and test a model on the hot-stamping dataset')
    parser.add_argument('--data', type=str, default=['all'], nargs='+', 
                        choices=['all', 'ce', 'df', 'gn7norm', 'gn7pano'],
                        help='select one or more datasets')
    parser.add_argument('--data_dir', type=str, default='./dataset/image', 
                        help='select one or more datasets')
    parser.add_argument('--resize', type=int, default=128,
                        help='Resize dimension as a single integer (e.g., 128).')
    parser.add_argument('--ratios', type=float, default=0.2, 
                        help='Ratios for splitting train and test sets (default: 0.2)')
    # Training Argument
    parser.add_argument('--gpu_idx', type=int, default=[0, 1], nargs='+', 
                        help='use single GPU(e.g., --gpu_idx 0), or use multiple GPUs(e.g., --gpu_idx 0 1)')
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
    parser.add_argument('--flops_lambda', type=float, default=0.0,
                        help='FLOPs penalty weight for multi-objective NAS (default: 0.0, no penalty)')
    parser.add_argument('--comparison', action='store_true',
                        help='Run comparison with baseline models (AutoPatch, RealtimeSeg style)')
    parser.add_argument('--baseline_models', type=str, nargs='+',
                        default=['autopatch', 'realtimeseg', 'unet', 'deeplabv3plus'],
                        choices=['autopatch', 'realtimeseg', 'unet', 'deeplabv3plus'],
                        help='Baseline models to compare (default: all)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save trained model checkpoints (default: ./checkpoints)')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help="Directory to save TensorBoard logs")
    
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

