import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # Environment Argument
    parser.add_argument('--seed', type=int, default=42, help="Seed value to ensure reproducibility.")
    # Preprocessing Argument
    parser.add_argument('--mode', type=str, default='nas', choices=['nas', 'ind', 'zero', 'hot', 'e2e'],
                        help='(nas): train and test the optimized model\n'                         
                            '(ind): train and test individual models\n'
                            '(zero): training for zero-shot testing\n'
                            '(hot): train and test a model on the hot-stamping dataset\n'
                            '(e2e): end-to-end training from preprocessing to learning')
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
    elif args.mode == 'e2e':
        e2e_args(parser)

    args = parser.parse_args()

    return args

# Hot-Stamping Arguments
def hotstamping_args(parser):
    parser.add_argument('--model_dir', type=str, required=False, 
                        help='Path to the best_model.pt file')
    

# End-to-End Arguments
def e2e_args(parser):
    parser.add_argument('--model_dir', type=str, required=False, 
                        help='Path to the best_model.pt file')
    parser.add_argument('--output_dir', type=str, required=False, 
                        help='Path to the best_model.pt file')
    parser.add_argument('--viz_infer', type=lambda x: (str(x).lower() == 'true'), default=False, required=False,
                        help='Visualize inferred results during testing to evaluate model predictions')
    # Skeleton Arguments
    parser.add_argument('--min_filtered_object_size', type=int, default=50, 
                        help='Minimum filtered object size (default: 20)')
    parser.add_argument('--min_threshold', type=float, default=0.5,
                        help='Minimum threshold value (default: 0.5)')
    parser.add_argument('--over_threshold', type=int, default=8,
                        help='The maximum allowable length range for processing.')
    parser.add_argument('--skeleton_window_size', type=int, default=20, 
                        help='Window size for skeleton calculation (default: 20)')
    parser.add_argument('--z_threshold', type=float, default=2.25, 
                        help='Z-score threshold for outlier detection (default: 2.25)')
    parser.add_argument('--step_size', type=int, default=5, 
                        help='Step size for processing (default: 5)')
    parser.add_argument('--pix_to_mm', type=float, default=0.14, 
                        help='Pixel to millimeter conversion factor (default: 0.14)')
    parser.add_argument('--processes', type=int, default=4,
                        help='Number of processes to use for multiprocessing (default: 4)')
    parser.add_argument('--min_NG_count', type=int, default=3,
                        help='The minimum number of consecutive occurrences of the min_threshold.')
    parser.add_argument('--viz_mode', type=str, default='contour', choices=['masking', 'contour'],
                        help='choose "masking" or "contour" for visualization.')

    
