import os
import time
import sknw
import torch
import numpy as np
from torch.utils.data import DataLoader

from utils.utils import AverageMeter, get_iou_score, set_device, capture, move_sections
from utils.dataloaders import set_transforms, HotDataset, TestDataset

from utils.skeleton import calculate_width


def inference_viz(args, model, test_loader):
    if isinstance(model, torch.nn.DataParallel):
        model.module.eval()  # DataParallel을 사용 중인 경우
    else:
        model.eval()

    with torch.no_grad():
        start_inference = time.time()

        for idx, (data, labels) in enumerate(test_loader):
            data, labels = data.cuda(), labels.cuda()

            torch.cuda.synchronize()
            outputs = model(data)
            torch.cuda.synchronize()

            # # Visualize model inference results
            # if args.viz_infer:
            #     test_loader.dataset.visualization(idx, outputs, args.viz_mode)   # option

            # get skeleton image and width
            # save_dir = test_loader.dataset.get_name(args.output_dir, idx, 'width')
            # calculate_width(args, outputs, save_dir)
            test_loader.dataset.calculate_width(args, idx, outputs)

        inference_time = (time.time() - start_inference)

    return inference_time


def test_hotstamping(model, test_hot_loader):
    if isinstance(model, torch.nn.DataParallel):
        model.module.eval()  # DataParallel을 사용 중인 경우
    else:
        model.eval()
    test_iou = AverageMeter()
    
    with torch.no_grad():
        for data, labels in test_hot_loader:
            data, labels = data.cuda(), labels.cuda()
            batch_size = data.size(0)
            outputs = model(data)
            test_iou.update(get_iou_score(outputs, labels), batch_size)

    return test_iou.avg

'''test hotstamping'''
def test_model(args, dataset):
    device = set_device(args.gpu_idx)

    checkpoint = torch.load(args.model_dir, map_location=device)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])

    if len(args.gpu_idx) >= 2:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_idx).to(device)
        print(f"Using multiple GPUs: {args.gpu_idx}")
    else:
        model = model.to(device)
        print(f"Using single GPU: cuda:{args.gpu_idx[0]}")

    test_data, test_ind_data = dataset
    transform = set_transforms(args.resize)

    test_dataset = HotDataset(test_data, transform)
    test_loader = DataLoader(test_dataset, batch_size=args.test_size, shuffle=False)
    test_iou = test_hotstamping(model, test_loader)

    args.writer.add_scalars('HotStamping Test/IOU', {f'Test_mIoU]': test_iou})
    print(f"TEST[ALL], Test IoU: {test_iou:.4f}")

    names = ["CE", "DF", "GN7 일반", "GN7 파노라마"]
    for test_ind in test_ind_data:
        matching_name = next((name for sublist in test_ind for name in names if name in sublist), None)

        test_ind_dataset = HotDataset(test_ind, transform, matching_name)

        test_ind_loader = DataLoader(test_ind_dataset, batch_size=args.test_size, shuffle=False)
        test_ind_iou = test_hotstamping(model, test_ind_loader)

        args.writer.add_scalars('HotStamping individual Test/IOU', {f'Test_mIoU[{matching_name}]': test_ind_iou})
        print(f"TEST[{matching_name}], Test IoU: {test_ind_iou:.4f}")


'''e2e mode'''
def inference(args, dataset):
    # TODO: Set up real-world scenario for e2e
    print("e2e test mode")
    device = set_device(args.gpu_idx)

    checkpoint = torch.load(args.model_dir, map_location=device)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])

    if len(args.gpu_idx) >= 2:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_idx).to(device)
        print(f"Using multiple GPUs: {args.gpu_idx}")
    else:
        model = model.to(device)
        print(f"Using single GPU: cuda:{args.gpu_idx[0]}")

    _, test_ind_data = dataset
    transform = set_transforms(args.resize)

    dummy_skeleton = np.zeros((10, 10), dtype=np.uint8)
    dummy_graph = sknw.build_sknw(dummy_skeleton)

    names = ["CE", "DF", "GN7 일반", "GN7 파노라마"]
    sections_name = {
        'CE': {'F001', 'F002', 'F003', 'F004', 'F005'},
        'DF': {'F001', 'F002', 'F003', 'F004'},
        'GN7 일반': {'F001', 'F002', 'F003'},
        'GN7 파노라마': {'F001', 'F002', 'F003', 'F004'},
    }
    for test_ind in test_ind_data:
        matching_name = next((name for sublist in test_ind for name in names if name in sublist), None)

        total_inference_time = []
        total_data_count = []
        total_inspection = []
        total_margin = []
        for test_data in test_ind:
            all_images = sorted([os.path.join(test_data, f) for f in os.listdir(test_data) if f.endswith('.jpg')])
            
            sections = {}
            for img_path in all_images:
                img_name = os.path.basename(img_path) 
                section = img_name.split('-')[0]  # Extract section prefix (e.g., F001, F002)
                if section not in sections:
                    sections[section] = []
                sections[section].append(img_path)

            total_section = []
            section_data = []
            inspection = []
            margin = []
            for section, section_images in sections.items():    # section = F001, F002, ...
                capture_time = capture(section_images)
                # capture_time = 0

                test_dataset = TestDataset(args, section_images, transform)
                test_loader = DataLoader(test_dataset, batch_size=args.test_size, shuffle=False)
                infer_time = inference_viz(args, model, test_loader)

                move_time = move_sections(matching_name, section)
                # move_time = 0

                # total_section_time = capture_time + infer_time + move_time
                total_section_time = capture_time + infer_time
                # total_section_time = infer_time
                total_section.append(total_section_time)
                section_data.append(len(section_images))

                inspection_time = capture_time + move_time
                inspection.append(inspection_time)

                if inspection_time > total_section_time:
                    margin.append(0)
                else:
                    margin.append(total_section_time - inspection_time)

            total_inference_time.append(total_section)
            total_data_count.append(section_data)
            total_inspection.append(inspection)
            total_margin.append(margin)

            print("=============================")
            print(matching_name)
            test_avg = [sum(values) / len(values) for values in zip(*total_inference_time)]
            print("test avg: ", test_avg)
            # test_sum_inference = sum(test_avg)
            # print("test sum: ", test_sum_inference)
            
            test_data_avg = [sum(values) / len(values) for values in zip(*total_data_count)]
            test_sum_data = sum(test_data_avg)
            print("test data avg: ", test_data_avg)
            print("test data sum: ", test_sum_data)

            inspection_avg = [sum(values) / len(values) for values in zip(*total_inspection)]
            inspection_sum = sum(inspection_avg)
            print("inspection avg: ", inspection_avg)
            print("inspection sum: ", inspection_sum)

            margin_avg = [sum(values) / len(values) for values in zip(*total_margin)]
            margin_sum = sum(margin_avg)
            print("margin avg: ", margin_avg)
            print("margin sum: ", margin_sum)

            print("total time: ", (inspection_sum + margin_sum))
            print("=============================")

            # Create the output directory if it doesn't exist
            output_dir = os.path.join(args.output_dir, "time", matching_name)
            os.makedirs(output_dir, exist_ok=True)
            output_file_path = os.path.join(output_dir, f"{matching_name}_results.txt")

            # Write the print contents to the file
            with open(output_file_path, "w") as f:
                f.write(f"{matching_name}\n")
                f.write(f"test avg: {test_avg}\n")
                f.write(f"test data avg: {test_data_avg}\n")
                f.write(f"test data sum: {test_sum_data}\n")
                f.write(f"inspection avg: {inspection_avg}\n")
                f.write(f"inspection sum: {inspection_sum}\n")
                f.write(f"margin avg: {margin_avg}\n")
                f.write(f"margin sum: {margin_sum}\n")


        inference_avg = [sum(values) / len(values) for values in zip(*total_inference_time)]
        # data_avg = [sum(values) / len(values) for values in zip(*total_data_count)]
        # sum_inference = sum(inference_avg)

        # for idx, avg_meter in enumerate(inference_avg):
        #     args.writer.add_scalars('E2E Inference Data Count', 
        #                             {f'#Data[{matching_name}][{idx+1}]': data_avg[idx]})
        #     args.writer.add_scalars('E2E Total Time', 
        #                             {f'Inference_Time[{matching_name}][{idx+1}]': avg_meter})
            
        # args.writer.add_scalars('E2E Total Time', 
        #                         {f'Inference_Time[{matching_name}][{idx+1}]': avg_meter.avg})
        # print(f"TEST[{matching_name}], Inference Time: {sum_inference:.4f} seconds")