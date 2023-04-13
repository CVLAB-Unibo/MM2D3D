import json
import os.path as osp
import pickle

import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms as T

from lib.utils.augmentation_3d import augment_and_scale_3d

ImageFile.LOAD_TRUNCATED_IMAGES = True


class A2D2Base(Dataset):
    """A2D2 dataset"""

    class_names = [
        "Car 1",
        "Car 2",
        "Car 3",
        "Car 4",
        "Bicycle 1",
        "Bicycle 2",
        "Bicycle 3",
        "Bicycle 4",
        "Pedestrian 1",
        "Pedestrian 2",
        "Pedestrian 3",
        "Truck 1",
        "Truck 2",
        "Truck 3",
        "Small vehicles 1",
        "Small vehicles 2",
        "Small vehicles 3",
        "Traffic signal 1",
        "Traffic signal 2",
        "Traffic signal 3",
        "Traffic sign 1",
        "Traffic sign 2",
        "Traffic sign 3",
        "Utility vehicle 1",
        "Utility vehicle 2",
        "Sidebars",
        "Speed bumper",
        "Curbstone",
        "Solid line",
        "Irrelevant signs",
        "Road blocks",
        "Tractor",
        "Non-drivable street",
        "Zebra crossing",
        "Obstacles / trash",
        "Poles",
        "RD restricted area",
        "Animals",
        "Grid structure",
        "Signal corpus",
        "Drivable cobblestone",
        "Electronic traffic",
        "Slow drive area",
        "Nature object",
        "Parking area",
        "Sidewalk",
        "Ego car",
        "Painted driv. instr.",
        "Traffic guide obj.",
        "Dashed line",
        "RD normal street",
        "Sky",
        "Buildings",
        "Blurred area",
        "Rain dirt",
    ]

    # use those categories if merge_classes == True
    categories = {
        "car": ["Car 1", "Car 2", "Car 3", "Car 4", "Ego car"],
        "truck": ["Truck 1", "Truck 2", "Truck 3"],
        "bike": [
            "Bicycle 1",
            "Bicycle 2",
            "Bicycle 3",
            "Bicycle 4",
            "Small vehicles 1",
            "Small vehicles 2",
            "Small vehicles 3",
        ],  # small vehicles are "usually" motorcycles
        "person": ["Pedestrian 1", "Pedestrian 2", "Pedestrian 3"],
        "road": [
            "RD normal street",
            "Zebra crossing",
            "Solid line",
            "RD restricted area",
            "Slow drive area",
            "Drivable cobblestone",
            "Dashed line",
            "Painted driv. instr.",
        ],
        "parking": ["Parking area"],
        "sidewalk": ["Sidewalk", "Curbstone"],
        "building": ["Buildings"],
        "nature": ["Nature object"],
        "other-objects": [
            "Poles",
            "Traffic signal 1",
            "Traffic signal 2",
            "Traffic signal 3",
            "Traffic sign 1",
            "Traffic sign 2",
            "Traffic sign 3",
            "Sidebars",
            "Speed bumper",
            "Irrelevant signs",
            "Road blocks",
            "Obstacles / trash",
            "Animals",
            "Signal corpus",
            "Electronic traffic",
            "Traffic guide obj.",
            "Grid structure",
        ],
        # 'ignore': ['Sky', 'Utility vehicle 1', 'Utility vehicle 2', 'Tractor', 'Non-drivable street',
        #            'Blurred area', 'Rain dirt'],
    }

    def __init__(
        self,
        split,
        preprocess_dir,
        merge_classes=False,
        short_run=False,
        reduce_factor=1,
    ):

        self.split = split
        self.preprocess_dir = preprocess_dir

        print("Initialize A2D2 dataloader")

        with open(osp.join(self.preprocess_dir, "cams_lidars.json"), "r") as f:
            self.config = json.load(f)

        print("Load", split)
        self.data = []
        for curr_split in split:
            with open(
                osp.join(self.preprocess_dir, "preprocess", curr_split + ".pkl"), "rb"
            ) as f:
                self.data.extend(pickle.load(f))

        if "train" in split[0] and short_run:
            keep_list = []
            for idx, elem in enumerate(self.data):
                keep_list.append((idx, elem["camera_path"]))

            self.data_short = []
            keep_list = sorted(keep_list, key=lambda x: x[1])
            for idx, elem in enumerate(keep_list):
                if idx % reduce_factor == 0:
                    self.data_short.append(self.data[elem[0]])

            self.data = self.data_short

        with open(osp.join(self.preprocess_dir, "class_list.json"), "r") as f:
            class_list = json.load(f)
            self.rgb_to_class = {}
            self.rgb_to_cls_idx = {}
            count = 0
            for k, v in class_list.items():
                # hex to rgb
                rgb_value = tuple(int(k.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
                self.rgb_to_class[rgb_value] = v
                self.rgb_to_cls_idx[rgb_value] = count
                count += 1

        assert self.class_names == list(self.rgb_to_class.values())
        if merge_classes:
            self.label_mapping = -100 * np.ones(len(self.rgb_to_class) + 1, dtype=int)
            for cat_idx, cat_list in enumerate(self.categories.values()):
                for class_name in cat_list:
                    self.label_mapping[self.class_names.index(class_name)] = cat_idx
            self.class_names = list(self.categories.keys())
        else:
            self.label_mapping = None

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


class A2D2SCN(A2D2Base):
    def __init__(
        self,
        split,
        preprocess_dir,
        merge_classes=True,
        merge_classes_style="A2D2",
        scale=20,
        full_scale=4096,
        resize=(480, 302),
        image_normalizer=None,
        noisy_rot=0.0,  # 3D augmentation
        flip_x=0.0,  # 3D augmentation
        rot=0.0,  # 3D augmentation
        transl=False,  # 3D augmentation
        rand_crop=tuple(),  # 2D augmentation
        crop_size=tuple(),  # 2D augmentation
        bottom_crop=False,
        fliplr=0.0,  # 2D augmentation
        color_jitter=None,  # 2D augmentation
        short_run=False,
        reduce_factor=1,
        camera_coords=False,
        use_rgb=False,
    ):
        super().__init__(
            split,
            preprocess_dir,
            merge_classes=merge_classes,
            short_run=short_run,
            reduce_factor=reduce_factor,
        )

        # point cloud parameters
        self.scale = scale
        self.full_scale = full_scale
        # 3D augmentation
        self.noisy_rot = noisy_rot
        self.flip_x = flip_x
        self.rot = rot
        self.transl = transl

        # image parameters
        self.resize = resize
        self.image_normalizer = image_normalizer

        # data augmentation
        if rand_crop:
            self.crop_prob = rand_crop[0]
            self.crop_dims = np.array(rand_crop[1:])
        else:
            self.crop_prob = 0.0
        self.fliplr = fliplr
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None
        self.camera_coords = camera_coords
        self.use_rgb = use_rgb

    def __getitem__(self, index):
        data_dict = self.data[index]

        pts_cam_coord = data_dict["pts_cam_coord"].copy()
        if self.camera_coords:
            points = pts_cam_coord.copy()
        else:
            points = data_dict["points"].copy()

        seg_label = data_dict["seg_labels"].astype(np.int64)
        intrinsics = np.array(
            [
                [1687.3369140625, 0.0, 965.43414055823814],
                [0.0, 1783.428466796875, 684.4193604186803],
                [0.0, 0.0, 1.0],
            ]
        ).copy()

        if self.label_mapping is not None:
            seg_label = self.label_mapping[seg_label]

        out_dict = {}

        points_img = data_dict["points_img"].copy()
        img_path = osp.join(self.preprocess_dir, data_dict["camera_path"])
        image = Image.open(img_path)

        if np.random.rand() < self.crop_prob:
            valid_crop = False
            for _ in range(10):
                # self.crop_dims is a tuple of floats in interval (0, 1):
                # (min_crop_height, max_crop_height, min_crop_width, max_crop_width)
                crop_height, crop_width = self.crop_dims[0::2] + np.random.rand(2) * (
                    self.crop_dims[1::2] - self.crop_dims[0::2]
                )
                top = np.random.rand() * (1 - crop_height) * image.size[1]
                left = np.random.rand() * (1 - crop_width) * image.size[0]
                bottom = top + crop_height * image.size[1]
                right = left + crop_width * image.size[0]
                top, left, bottom, right = int(top), int(left), int(bottom), int(right)

                # discard points outside of crop
                keep_idx = points_img[:, 0] >= top
                keep_idx = np.logical_and(keep_idx, points_img[:, 0] < bottom)
                keep_idx = np.logical_and(keep_idx, points_img[:, 1] >= left)
                keep_idx = np.logical_and(keep_idx, points_img[:, 1] < right)

                if np.sum(keep_idx) > 100:
                    valid_crop = True
                    break

            if valid_crop:
                # crop image
                image = image.crop((left, top, right, bottom))
                points_img = points_img[keep_idx]
                points_img[:, 0] -= top
                points_img[:, 1] -= left

                # update point cloud
                points = points[keep_idx]
                pts_cam_coord = pts_cam_coord[keep_idx]
                seg_label = seg_label[keep_idx]
            else:
                print("No valid crop found for image", data_dict["camera_path"])

        if self.resize:
            # always resize (crop or no crop)
            if not image.size == self.resize:
                # check if we do not enlarge downsized images
                assert image.size[0] > self.resize[0]

                # scale image points
                points_img[:, 0] = (
                    float(self.resize[1]) / image.size[1] * np.floor(points_img[:, 0])
                )
                points_img[:, 1] = (
                    float(self.resize[0]) / image.size[0] * np.floor(points_img[:, 1])
                )

                # resize image
                image = image.resize(self.resize, Image.BILINEAR)
                intrinsics[:2] /= 4

        img_indices = points_img.astype(np.int64)

        assert np.all(img_indices[:, 0] >= 0)
        assert np.all(img_indices[:, 1] >= 0)
        assert np.all(img_indices[:, 0] < image.size[1])
        assert np.all(img_indices[:, 1] < image.size[0])

        depth = np.zeros((image.size[1], image.size[0]))
        depth[img_indices[:, 0], img_indices[:, 1]] = pts_cam_coord[:, 2]
        seg_labels_2d = np.ones((image.size[1], image.size[0])) * (-100)
        seg_labels_2d[img_indices[:, 0], img_indices[:, 1]] = seg_label

        # 2D augmentation
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        # PIL to numpy
        image = np.array(image, dtype=np.float32, copy=False) / 255.0
        # 2D augmentation
        if np.random.rand() < self.fliplr:
            image = np.ascontiguousarray(np.fliplr(image))
            depth = np.ascontiguousarray(np.fliplr(depth))
            img_indices[:, 1] = image.shape[1] - 1 - img_indices[:, 1]
            intrinsics[0, 2] = image.shape[1] - intrinsics[0, 2]
            intrinsics[1, 2] = image.shape[0] - intrinsics[0, 1]
            seg_labels_2d = np.ascontiguousarray(np.fliplr(seg_labels_2d))

        # normalize image
        if self.image_normalizer:
            mean, std = self.image_normalizer
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            image = (image - mean) / std

        out_dict["img"] = np.moveaxis(image, -1, 0)
        out_dict["img_indices"] = img_indices

        # 3D data augmentation and scaling from points to voxel indices
        # A2D2 lidar coordinates (same as Kitti): x (front), y (left), z (up)
        coords, min_value, offset, rot_matrix = augment_and_scale_3d(
            points,
            self.scale,
            self.full_scale,
            noisy_rot=self.noisy_rot,
            flip_x=self.flip_x,
            rot_z=self.rot if not self.camera_coords else 0,
            rot_y=self.rot if self.camera_coords else 0,
            transl=self.transl,
        )

        # cast to integer
        coords = coords.astype(np.int64)

        # only use voxels inside receptive field
        idxs = (coords.min(1) >= 0) * (coords.max(1) < self.full_scale)

        out_dict["coords"] = coords[idxs]
        out_dict["points"] = points[idxs]
        out_dict["intrinsics"] = intrinsics
        out_dict["seg_label"] = seg_label[idxs]
        out_dict["img_indices"] = out_dict["img_indices"][idxs]
        if self.use_rgb:
            out_dict["feats"] = out_dict["img"][
                :, out_dict["img_indices"][:, 0], out_dict["img_indices"][:, 1]
            ].T
        else:
            out_dict["feats"] = np.ones(
                [len(idxs), 1], np.float32
            )  # simply use 1 as feature
        out_dict["depth"] = depth[None].astype(np.float32)
        out_dict["seg_labels_2d"] = seg_labels_2d
        out_dict["min_value"] = min_value
        out_dict["offset"] = offset
        out_dict["rot_matrix"] = rot_matrix

        return out_dict
