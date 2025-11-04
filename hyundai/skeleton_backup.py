import os
import cv2
import sknw
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from multiprocessing import Pool
from skimage.morphology import skeletonize

import torch
import torch.nn.functional as F

from utils.utils import calculate_average_direction, draw_averaged_normals, filter_outliers_by_zscore, relabel_filtered_lengths

def skeleton(args, binary_image, num_labels, labels_im, save_dir, file_name_list):
    # filtering out components smaller than the specified minimum size
    file_name = file_name_list
    binary_img = binary_image

    for label in range(1, num_labels):
        label_mask = (labels_im == label)
        if np.sum(label_mask) < args.min_filtered_object_size:
            binary_img[label_mask] = 0
            continue

        object_mask = (labels_im == label).astype(np.uint8) * 255
        skeleton = skeletonize(object_mask).astype(np.uint8)
        graph = sknw.build_sknw(skeleton)
        accumulated_values = np.zeros_like(object_mask)

        count = 0
        total_lengths = []
        label_counter = 0
        label_image_total = np.zeros_like(object_mask)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        for (start, end, path) in graph.edges(data='pts'):
            directions = calculate_average_direction(path, window_size=args.skeleton_window_size)
            skeleton_with_normals, label_image, line_lengths, label_counter = draw_averaged_normals(object_mask, path, directions, step=args.step_size, start_label=label_counter, pix_to_mm = args.pix_to_mm)

            total_lengths.extend(line_lengths)
            skeleton_with_normals[skeleton_with_normals == 255] = 0
            accumulated_values = np.maximum(accumulated_values, skeleton_with_normals)
            label_image_total = np.maximum(label_image_total, label_image)

            # axs[0].plot(path[:, 1], path[:, 0], color='red', linewidth=1)
            
            # for label, (y, x) in enumerate(path):
            #     if count % args.step_size == 0:
            #         axs[0].text(x, y, str(int(count/args.step_size)), color='blue', fontweight='bold', fontsize=12, ha='center')
            #     count += 1

        # 전체 길이에 대해 z-score 필터링 적용
        filtered_lengths, outlier_labels = filter_outliers_by_zscore(total_lengths, z_threshold=args.z_threshold)

        # 필터링된 라벨을 0부터 연속적으로 재정렬
        relabeled_lengths = relabel_filtered_lengths(filtered_lengths)
        accumulated_values[np.isin(label_image_total, outlier_labels)] = 0
        accumulated_values = np.where(accumulated_values == 128, accumulated_values, np.nan)
        combined_image = np.where(accumulated_values > 0, accumulated_values, object_mask)
        labels, lengths = zip(*relabeled_lengths)

        # axs[0].imshow(combined_image, cmap="gray")  ###### 스켈레톤 이미지 = combined_image
        # axs[0].axis('off')

        # axs[1].bar(labels, lengths)
        # axs[1].set_xticks(labels)
        # axs[1].set_xticklabels(labels, rotation=45, ha='right')
        # axs[1].set_xticks(axs[1].get_xticks()[::2])

        # axs[1].set_ylim(0, 10)
        # axs[1].set_xlabel('Label Number')
        # axs[1].set_ylabel('Length (mm)')
        # axs[1].set_title('Line Lengths [' + file_name + ']')

        # if any(length < args.min_threshold for length in lengths):
        #     output_file_skeleton = os.path.join(save_dir, f"{file_name}_NG_CAM.jpg")
        # else:
        #     output_file_skeleton = os.path.join(save_dir, f"{file_name}.jpg")

        # plt.savefig(output_file_skeleton, dpi=70, format='jpg', bbox_inches='tight', transparent=False)
        # plt.close(fig)

def skeleton_worker(components):
    args, image, n_labels, im_labels, save_dir, save_file = components
    skeleton(args, image, n_labels, im_labels, save_dir, save_file)

def calculate_width(args, outputs, save_dir):
    # TODO: Skeleton Image
    save_dir_list, save_file_list = save_dir

    outputs = F.softmax(outputs, dim=1)
    _, predicted = torch.max(outputs, dim=1)
    output_numpy = predicted.cpu().numpy()
    binary_image = (output_numpy * 255).astype(np.uint8)
    binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)[1]
    binary_image_normalized = binary_image // 255

    num_labels_list, im_labels_list = [], []
    for b_image in binary_image_normalized:
        num_labels, labels_im = cv2.connectedComponents(b_image)
        num_labels_list.append(num_labels)
        im_labels_list.append(labels_im)

    skeleton(args, binary_image, num_labels_list, im_labels_list, save_dir_list, save_file_list)

