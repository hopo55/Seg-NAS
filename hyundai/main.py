import wandb

import os
from datetime import datetime
from utils.argument import get_args
from utils.utils import set_seed
from preprocessing import get_roi, get_dataset
from segmentation import search_architecture, train_searched_model
from test import test_model, inference
from torch.utils.tensorboard import SummaryWriter


def main():
    import setproctitle
    setproctitle.setproctitle('hyundai/hspark')
    args = get_args()
    set_seed(args.seed)

    date_time = str(datetime.now().date()) + "/" + datetime.now().strftime("%H_%M_%S")
    args.log_dir = os.path.join(args.log_dir, args.mode)
    if args.mode == 'ind' or args.mode == 'zero':
        args.log_dir = os.path.join(args.log_dir, args.data[0])
    args.log_dir = os.path.join(args.log_dir, date_time)
    args.writer = SummaryWriter(log_dir=args.log_dir)
    args_text = "<br>".join([f"{arg}: {value}" for arg, value in vars(args).items()])
    args.writer.add_text('Arguments', args_text)
    
    args_text_file = "\n".join([f"{arg}: {value}" for arg, value in vars(args).items()])
    with open(args.log_dir + '/args.txt', 'w') as f:
        f.write(args_text_file)

    '''
    wandb.login()
    wandb.init(config=args, project="hyundai_segnas", entity="hyundai_ai")

    # Data Preprocessing
    get_roi(args.data)
    dataset = get_dataset(args)
    
    if args.mode in ['nas', 'ind', 'zero']:
        # Search Architecture
        searched_model = search_architecture(args, dataset)
        
        # Train and Test of the Optimized Architecture
        train_searched_model(args, searched_model, dataset)
        args.writer.close()
    elif args.mode == 'hot':
        # Model Testing
        test_model(args, dataset)
        args.writer.close()
    else:   # e2e
        # Model Testing
        args.writer = None
        inference(args, dataset)
    '''

if __name__ == "__main__":
    main()

    