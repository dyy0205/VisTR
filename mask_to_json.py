import numpy as np
import os, glob
import json
from PIL import Image
import pycocotools.mask as maskUtils


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


if __name__ == '__main__':
    root = '/versa/dataset/TIANCHI2021/PreRoundData'
    img_dir = os.path.join(root, 'JPEGImages')
    ann_dir = os.path.join(root, 'Annotations')
    videos = os.listdir(ann_dir)

    categories = [{'supercategory': 'person', 'id': 1, 'name': 'person'}]
    video_infos = []
    annotations = []
    ann_count = 1
    for i, video in enumerate(videos):
        print(i, video)
        mask_paths = sorted(glob.glob(os.path.join(ann_dir, video, '*.png')))

        masks = []
        file_names = []
        max_num = 0
        for mask_path in mask_paths:
            filename = '/'.join(mask_path.split('/')[-2:]).replace('.png', '.jpg')
            if not os.path.exists(os.path.join(img_dir, filename)):
                continue
            file_names.append(filename)
            mask = np.array(Image.open(mask_path))
            masks.append(mask)
            num_ins = np.unique(mask)[-1]
            if num_ins > max_num:
                max_num = num_ins
        if max_num == 0:
            continue

        h, w = masks[0].shape
        video_info = {
            'id': i + 1, 'height': h, 'width': w, 'length': len(masks), 'file_names': file_names
        }
        video_infos.append(video_info)

        for idx in range(1, max_num+1):
            segmentations = []
            areas = []
            bboxes = []
            for mask in masks:
                if idx in np.unique(mask):
                    ins_mask = (mask == idx).astype(np.uint8) * 255
                    binary_mask = maskUtils.encode(np.asfortranarray(ins_mask))
                    area = maskUtils.area(binary_mask)
                    bbox = maskUtils.toBbox(binary_mask)
                    segmentations.append(binary_mask)
                    areas.append(area)
                    bboxes.append(bbox)
                else:
                    segmentations.append(None)
                    areas.append(None)
                    bboxes.append(None)

            ann_info = {
                'id': ann_count, 'video_id': i + 1, 'height': h, 'width': w, 'category_id': 1,
                'iscrowd': 0, 'segmentations': segmentations, 'areas': areas, 'bboxes': bboxes
            }
            annotations.append(ann_info)
            ann_count += 1

    json_file = {'videos': video_infos, 'annotations': annotations, 'categories': categories}
    with open(os.path.join(root, 'instances_train.json'), 'w') as f:
        json.dump(json_file, f, cls=MyEncoder)


    # with open(os.path.join(root, 'instances_train.json'), 'r') as f:
    #     raw_data = json.load(f)
    # data = raw_data.copy()
    # annotations = data['annotations']
    # categories = data['categories']
    # ids = []
    # for i, vid in enumerate(videos):
    #     vid_name = vid['file_names'][0].split('/')[0]
    #     if vid_name in ['626133', '628040', '627423', '628027', '626138', '627822']:
    #         ids.append(vid['id'])
    #         videos[i]['length'] = videos[i]['length'] - 1
    #         videos[i]['file_names'] = videos[i]['file_names'][:-1]
    #
    # for i, ann in enumerate(annotations):
    #     if ann['video_id'] in ids:
    #         annotations[i]['segmentations'] = annotations[i]['segmentations'][:-1]
    #         annotations[i]['areas'] = annotations[i]['areas'][:-1]
    #         annotations[i]['bboxes'] = annotations[i]['bboxes'][:-1]
    #
    # json_file = {'videos': videos, 'annotations': annotations, 'categories': categories}
    # with open(os.path.join(root, 'instances_train_new.json'), 'w') as f:
    #     json.dump(json_file, f, cls=MyEncoder)