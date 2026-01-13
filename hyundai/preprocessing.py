import os
import sys
import pandas as pd
from PIL import Image
from utils.dataloaders import load_folder_model, load_zero_shot, set_transforms, ImageDataset


def get_roi(names):
    csv_root_folder = "dataset/roi"

    if names == ['all']:
        names = ["CE", "DF", "GN7 일반", "GN7 파노라마"]
    else:
        name_mapping = {
        'ce': "CE",
        'df': "DF",
        'gn7norm': "GN7 일반",
        'gn7pano': "GN7 파노라마"
        }
        names = [name_mapping.get(name, name) for name in names]

    csv_file_dict = {
        "DF": "02_ROIprofile.csv",
        "CE": "03_ROIprofile.csv",
        "GN7 일반": "05_ROIprofile.csv",
        "GN7 파노라마": "06_ROIprofile.csv",
    }

    roi_dfs = {}

    for folder_name, _, filenames in os.walk(csv_root_folder):
        for filename in filenames:
            if filename.lower().endswith('.csv'):
                full_path = os.path.join(folder_name, filename)
                for name in names:
                    if csv_file_dict[name] in full_path:
                        roi_dfs[name] = pd.read_csv(full_path, header=None)

    image_indexes = {}

    for name, df in roi_dfs.items():
        row_count = []
        
        max_len = len(df.iloc[0])
        max_count = int((max_len)/8)

        for _, row in df.iterrows():
            count = 0

            for idx in range(max_len, 0, -8):
                section_start = max(0, idx-8)
                row_val = row[section_start:idx]

                if all(val == 0 for val in row_val):
                    count += 1
                else:
                    break

            row_count.append(max_count - count)

        image_indexes[name] = row_count

    root_folder = "dataset/original/"

    for folder_name, _, filenames in os.walk(root_folder):
        if filenames:
            folder = folder_name.split('\\')[-1]
            target_name = folder.split('/')[2:]
            image_folder = target_name[-1]
            
            target_name = '/'.join(target_name)

            target = next((name for name in names if name in folder), None)
            if target is None: continue
            
            target_roi = roi_dfs[target]
            
            for filename in filenames:
                if filename.lower().endswith('.jpg'):
                    target_file, target_num = filename.split('-')

                    target_limit_list = image_indexes[target]
                    target_idx = int(target_file[-1]) - 1
                    target_limit = (target_limit_list[target_idx])
                    current_idx = int(os.path.splitext(target_num)[0][-2:]) - 1

                    if target_limit > current_idx:
                        image_path = os.path.join(folder_name, filename)
                        image = Image.open(image_path)

                        col_start = current_idx * 8

                        roi = [(target_roi.iloc[target_idx, i], target_roi.iloc[target_idx, i + 1]) for i in range(col_start, col_start + 8, 2)]
                        
                        if not all(x == 0 for xy in roi for x in xy):
                            left = min(roi, key=lambda x: x[0])[0]
                            top = min(roi, key=lambda x: x[1])[1]
                            right = max(roi, key=lambda x: x[0])[0]
                            bottom = max(roi, key=lambda x: x[1])[1]

                            roi_image = image.crop((left, top, right, bottom))
                            
                            width, height = roi_image.size
                            if width > 30 or height > 30:   # check
                                # create train and target folder
                                train_path = os.path.join("dataset/crop", target_name, os.path.basename(image_path))
                                os.makedirs(os.path.dirname(train_path), exist_ok=True)
                                roi_image.save(train_path)

                                os.makedirs("dataset/data", exist_ok=True)
                                os.makedirs("dataset/label", exist_ok=True)

                                image_folder_path = os.path.join("dataset/data", image_folder, os.path.basename(image_path))
                                os.makedirs(os.path.dirname(image_folder_path), exist_ok=True)
                                roi_image.save(image_folder_path)


def get_dataset(args):
    if args.mode in ['nas', 'hot'] and args.data != ['all']:
        print(f"Error: The '{args.data}' option cannot be used in modes other than 'nas' or 'hot'.")
        sys.exit(1)
    elif args.mode == 'ind' and args.data == ['all']:
        print("Error: The 'all' option cannot be used in 'ind' mode.")
        sys.exit(1)
    elif args.mode == 'zero' and (len(args.data) != 1 or args.data[0] not in ['ce', 'df', 'gn7norm', 'gn7pano']):
        print("Error: In 'zero' mode must be one of 'ce', 'df', 'gn7norm', 'gn7pano'.")
        sys.exit(1)

    if args.data == ['all']:        
        '''Shuffle and split by car model'''
        dataset = load_folder_model(args.data_dir, args.ratios, args.data)

        '''Shuffle and split all image files'''
        # dataset = load_image(args.data_dir, args.ratios)
    else:
        if args.mode == 'zero':
            dataset = load_zero_shot(args.data_dir, args.data)
        else:   # ind
            dataset = load_folder_model(args.data_dir, args.ratios, args.data)

    transform = set_transforms(args.resize)

    if args.mode == 'hot':
        # set test dataset for hotstamping mode
        _, _, test_data, test_ind_data = dataset

        return test_data, test_ind_data
    else:
        # set dataset for nas, ind, zero mode
        train_data, val_data, test_data, test_ind_data = dataset

        train_dataset = ImageDataset(train_data, transform)
        val_dataset = ImageDataset(val_data, transform)
        test_dataset = ImageDataset(test_data, transform)

        return train_dataset, val_dataset, test_dataset, test_ind_data