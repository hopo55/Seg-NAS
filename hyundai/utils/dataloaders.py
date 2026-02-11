import os
import cv2
import random
import matplotlib
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

# matplotlib.use('Agg')


def _replace_dir_component(path, src_name, dst_name):
    """Replace one directory component in a path, preserving absolute/relative form."""
    normalized = os.path.normpath(path)
    drive, tail = os.path.splitdrive(normalized)
    is_abs = tail.startswith(os.sep)
    parts = [p for p in tail.split(os.sep) if p]
    for i, part in enumerate(parts):
        if part == src_name:
            parts[i] = dst_name
            rebuilt = os.path.join(*parts) if parts else ""
            if is_abs:
                rebuilt = os.sep + rebuilt
            return drive + rebuilt
    return path.replace(src_name, dst_name, 1)


def _resolve_label_path(label_path):
    """
    Prefer lossless masks when both exist (e.g., .png over .jpg) by trying
    sibling files with common image extensions.
    """
    root, _ = os.path.splitext(label_path)
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        candidate = root + ext
        if os.path.exists(candidate):
            return candidate
    return label_path

def set_transforms(resize=128):
    # set up transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((resize, resize)),
        ]
    )
    return transform

# Split data into train, validation, and test sets based on cycle(folder) / 기존 방식
def load_folder_cycle(data_dir, ratios):
    train_ratio, valid_ratio, test_ratio = ratios

    folder_list = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    random.shuffle(folder_list)

    train_end = int(len(folder_list) * train_ratio)
    valid_end = train_end + int(len(folder_list) * valid_ratio)

    train_folders = folder_list[:train_end]
    valid_folders = folder_list[train_end:valid_end]
    test_folders = folder_list[valid_end:]
    
    return train_folders, valid_folders, test_folders


# Split data into train, validation, and test sets based on car model.
def load_folder_model(data_dir, ratios, names, train_val_split=0.8):
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

    car_model = {name: [] for name in names}
    # train_ratio, valid_ratio, test_ratio = ratios
    train_folders, valid_folders, test_folders, test_ind_folders = [], [], [], []
    
    folder_list = [f.path for f in os.scandir(data_dir) if f.is_dir()]

    for name in names:
        for folder in folder_list:
            if name in folder:
                car_model[name].append(folder)

    train_val_split = float(train_val_split)
    if not (0.0 < train_val_split < 1.0):
        raise ValueError(f"train_val_split must be in (0, 1), got {train_val_split}")

    for folders in car_model.values():
        random.shuffle(folders)
        total = len(folders)
        train_end = int(total * (1 - ratios))

        train_and_valid = folders[:train_end]
        test_folders.extend(folders[train_end:])
        test_ind_folders.append(folders[train_end:])

        train_valid_total = len(train_and_valid)
        if train_valid_total <= 1:
            train_folders.extend(train_and_valid)
            continue

        train_split_end = int(train_valid_total * train_val_split)
        train_split_end = max(1, min(train_split_end, train_valid_total - 1))

        train_folders.extend(train_and_valid[:train_split_end])
        valid_folders.extend(train_and_valid[train_split_end:])

    return train_folders, valid_folders, test_folders, test_ind_folders


# Split data into train, validation, and test sets based on all image files.
def load_image(data_dir, ratios):
    jpg_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))

    random.shuffle(jpg_files)

    temp_files, test_files = train_test_split(jpg_files, test_size=ratios)
    train_files, valid_files = train_test_split(temp_files, test_size=0.5)

    names = ["CE", "DF", "GN7 일반", "GN7 파노라마"]
    car_model = {name: [] for name in names}
    test_ind_folders = []

    for file in test_files:
        for name in names:
            if name in file:
                car_model[name].append(file)
                break

    for folders in car_model.values():
        test_ind_folders.append(folders)

    return train_files, valid_files, test_files, test_ind_folders


# Split data into train, validation, and test sets based on car model.
def load_zero_shot(data_dir, name):
    name_mapping = {
    'ce': "CE",
    'df': "DF",
    'gn7norm': "GN7 일반",
    'gn7pano': "GN7 파노라마"
    }
    if name[0] not in name_mapping:
        raise ValueError(f"Invalid name: {name}")
    mapped_name = name_mapping[name[0]]
        
    folder_list = [f.path for f in os.scandir(data_dir) if f.is_dir()]

    # zero_folder, test_folders, test_ind_folders = [], [], []
    train_folders, valid_folders, zero_folders, test_ind_folders = [], [], [], []
    test_ind_names = {value: [] for key, value in name_mapping.items() if key != name[0]}

    for folder in folder_list:
        if mapped_name in folder:
            zero_folders.append(folder) # test
        else:
            train_folders.append(folder)    # train & validation
            for value in test_ind_names.keys():
                if value in folder:
                    test_ind_names[value].append(folder)
                    break

    random.shuffle(train_folders)
    train_folders, valid_folders = train_test_split(train_folders, test_size=0.5)

    for folders in test_ind_names.values():
        test_ind_folders.append(folders)

    return train_folders, valid_folders, zero_folders, test_ind_folders


class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, label_dir_name="target", source_dir_name="image"):
        self.data = []
        self.data_dir = data_dir
        self.transform = transform
        self.label_dir_name = label_dir_name
        self.source_dir_name = source_dir_name

        for folder in self.data_dir:
            # # if use all image files
            # self.data.append(folder)
            for file in os.listdir(folder):
                if file.endswith(".jpg"):
                    self.data.append(os.path.join(folder, file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # check if data and label are the same name after ../(dir)/
        label_dir = _replace_dir_component(self.data[idx], self.source_dir_name, self.label_dir_name)
        label_dir = _resolve_label_path(label_dir)
        
        image = cv2.imread(self.data[idx])
        label = cv2.imread(label_dir, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        label = label.squeeze(0)  # Remove the channel dimension
        label = (label >= 0.5).long()  # Threshold to create binary mask

        # Create a two-channel mask for background and target
        label_background = (label == 0).float()  # Background channel
        label_target = (label == 1).float()      # Target channel

        # Stack the channels to create a 2-channel label
        label = torch.stack([label_background, label_target], dim=0)

        return image, label

'''hotstamping'''
class HotDataset(Dataset):
    def __init__(self, data_dir, transform=None, name=None, label_dir_name="target", source_dir_name="image"):
        self.data = []
        self.data_dir = data_dir
        self.transform = transform
        self.label_dir_name = label_dir_name
        self.source_dir_name = source_dir_name

        # Define the ranges for each car model
        ranges = {
            "CE": [
                ("F003-0003", "F003-0008"),
                ("F003-0013", "F003-0025"),
                ("F005-0001", "F005-0049")
            ],
            "DF": [
                ("F001-0012", "F001-0023")
            ],
            "GN7 일반": [
                ("F001-0029", "F001-0041"),
                ("F002-0013", "F002-0031"),
                ("F003-0010", "F003-0018")
            ],
            "GN7 파노라마": [
                ("F001-0029", "F001-0041"),
                ("F002-0013", "F002-0031"),
                ("F003-0010", "F003-0019")
            ]
        }

        for folder in self.data_dir:
            for file in os.listdir(folder):
                if file.endswith(".jpg"):
                    car_model = self.get_car_model_from_folder(folder)
                    if car_model in ranges:
                        for start, end in ranges[car_model]:
                            if self.is_within_range(file, start, end):
                                self.data.append(os.path.join(folder, file))
                                break

        # Save the data list to a txt file
        file_name = f"{name}_hot_dataset_list.txt" if name else "hot_dataset_list.txt"
        file_dir = os.path.join('dataset/hotstamping')
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        file_name = os.path.join(file_dir, file_name)
        with open(file_name, "w") as f:
            for file_path in sorted(self.data):
                f.write(file_path + "\n")

    def __len__(self):
        return len(self.data)
    
    def get_car_model_from_folder(self, folder):
        # Extract car model from folder name
        for model in ["CE", "DF", "GN7 일반", "GN7 파노라마"]:
            if model in folder:
                return model
        return None

    def is_within_range(self, filename, start, end):
            # Check if the filename is within the given range
            file_prefix = filename.split(".")[0]
            return start <= file_prefix <= end

    def __getitem__(self, idx):
        # check if data and label are the same name after ../(dir)/
        label_dir = _replace_dir_component(self.data[idx], self.source_dir_name, self.label_dir_name)
        label_dir = _resolve_label_path(label_dir)
        
        image = cv2.imread(self.data[idx])
        label = cv2.imread(label_dir, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        label = label.squeeze(0)  # Remove the channel dimension
        label = (label >= 0.5).long()  # Threshold to create binary mask

        # Create a two-channel mask for background and target
        label_background = (label == 0).float()  # Background channel
        label_target = (label == 1).float()      # Target channel

        # Stack the channels to create a 2-channel label
        label = torch.stack([label_background, label_target], dim=0)

        return image, label
    

class TestDataset(Dataset):
    def __init__(self, args, data_dir, transform=None):
        self.args = args
        self.data = data_dir
        self.transform = transform
        self.data_dir = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # check if data and label are the same name after ../(dir)/
        source_dir_name = os.path.basename(os.path.normpath(getattr(self.args, "data_dir", "image")))
        label_dir_name = getattr(self.args, "label_dir_name", "target")
        label_dir = _replace_dir_component(self.data[idx], source_dir_name, label_dir_name)
        label_dir = _resolve_label_path(label_dir)
        self.data_dir.append(self.data[idx])
        
        image = cv2.imread(self.data[idx])
        label = cv2.imread(label_dir, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        label = label.squeeze(0)  # Remove the channel dimension
        label = (label >= 0.5).long()  # Threshold to create binary mask

        # Create a two-channel mask for background and target
        label_background = (label == 0).float()  # Background channel
        label_target = (label == 1).float()      # Target channel

        # Stack the channels to create a 2-channel label
        label = torch.stack([label_background, label_target], dim=0)

        return image, label    
    
    # def get_name(self, root, mode):
    #     file_path_list = self.data_dir
    #     file_dir_list = []
    #     file_name_list = []
        
    #     for file_path in file_path_list:
    #         file_path = file_path.split('/')[-2:]
        
    #         file_dir = os.path.join(root, mode, file_path[0])
    #         file_dir_list.append(file_dir)
    #         file_name = file_path[1]
    #         file_name_list.append(file_name)

    #         os.makedirs(file_dir, exist_ok=True)

    #     self.data_dir = []

    #     return file_dir_list, file_name_list
    
    def get_name(self, root, idx, mode):
        file_path = self.data[idx]
        file_path = file_path.split('/')[-2:]
        
        file_dir = os.path.join(root, mode, file_path[0])
        file_name = file_path[1]

        os.makedirs(file_dir, exist_ok=True)

        return file_dir, file_name
    
    def visualization(self, idx, outputs, mode):
        # TODO: Add batch-wise visualization code
        save_dir, file_name = self.get_name(self.args.output_dir, idx, 'viz')
        os.makedirs(save_dir, exist_ok=True)
        save_dir = os.path.join(save_dir, file_name)

        image = cv2.imread(self.data[idx])
        image = cv2.resize(image, (self.args.resize, self.args.resize))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # label_dir = self.data[idx].replace("image", "target")
        # label = cv2.imread(label_dir, cv2.IMREAD_GRAYSCALE)
        # label = cv2.resize(label, (self.args.resize, self.args.resize)) > 0.5

        prediction = F.softmax(outputs, dim=1).cpu().numpy() > 0.5
        prediction = prediction[0, 1, :, :]  # shape: (128, 128)
        
        if mode == "masking":
            pink_mask = np.zeros_like(image)
            pink_mask[prediction] = [255, 0, 0]
            image = cv2.addWeighted(image, 1.0, pink_mask, 0.4, 0)

            # mismatch_mask = np.zeros_like(image)
            # mismatch = (prediction != label)
            # mismatch_mask[mismatch] = [0, 0, 255]
            # overlay_image = cv2.addWeighted(overlay_image, 1.0, mismatch_mask, 0.4, 0)

        else:   # contour
            # Convert the prediction to uint8 for finding contours
            prediction_mask = prediction.astype(np.uint8) * 255

            # Find contours of the prediction (external contour only)
            contours, _ = cv2.findContours(prediction_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (255, 0, 0), 2)  # 빨간색 테두리 (0, 0, 255)

        cv2.imwrite(save_dir, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
