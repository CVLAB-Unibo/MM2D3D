import os.path as osp
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from lib.utils.augmentation_3d import augment_and_scale_3d
from lib.utils.refine_pseudo_labels import refine_pseudo_labels


class SemanticKITTIBase(Dataset):
    """SemanticKITTI dataset"""

    # https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
    id_to_class_name = {
        0: "unlabeled",
        1: "outlier",
        10: "car",
        11: "bicycle",
        13: "bus",
        15: "motorcycle",
        16: "on-rails",
        18: "truck",
        20: "other-vehicle",
        30: "person",
        31: "bicyclist",
        32: "motorcyclist",
        40: "road",
        44: "parking",
        48: "sidewalk",
        49: "other-ground",
        50: "building",
        51: "fence",
        52: "other-structure",
        60: "lane-marking",
        70: "vegetation",
        71: "trunk",
        72: "terrain",
        80: "pole",
        81: "traffic-sign",
        99: "other-object",
        252: "moving-car",
        253: "moving-bicyclist",
        254: "moving-person",
        255: "moving-motorcyclist",
        256: "moving-on-rails",
        257: "moving-bus",
        258: "moving-truck",
        259: "moving-other-vehicle",
    }

    class_name_to_id = {v: k for k, v in id_to_class_name.items()}

    # merging classes for common classes with A2D2 or VirtualKITTI
    categories = {
        "A2D2": {
            "car": ["car", "moving-car"],
            "truck": ["truck", "moving-truck"],
            "bike": [
                "bicycle",
                "motorcycle",
                "bicyclist",
                "motorcyclist",
                "moving-bicyclist",
                "moving-motorcyclist",
            ],  # riders are labeled as bikes in Audi dataset
            "person": ["person", "moving-person"],
            "road": ["road", "lane-marking"],
            "parking": ["parking"],
            "sidewalk": ["sidewalk"],
            "building": ["building"],
            "nature": ["vegetation", "trunk", "terrain"],
            "other-objects": ["fence", "pole", "traffic-sign", "other-object"],
        },
        "VirtualKITTI": {
            "vegetation_terrain": ["vegetation", "trunk", "terrain"],
            "building": ["building"],
            "road": ["road", "lane-marking"],
            "object": ["fence", "pole", "traffic-sign", "other-object"],
            "truck": ["truck", "moving-truck"],
            "car": ["car", "moving-car"],
        },
        "nuScenes": {
            "vehicle": [
                "truck",
                "moving-truck",
                "car",
                "moving-car",
                "bicycle",
                "motorcycle",
                "bicyclist",
                "motorcyclist",
                "moving-bicyclist",
                "moving-motorcyclist",
            ],
            "driveable_surface": ["road", "lane-marking", "parking"],
            "sidewalk": ["sidewalk"],
            "terrain": ["terrain"],
            "manmade": ["building", "fence", "pole", "traffic-sign", "other-object"],
            "vegetation": ["vegetation", "trunk"],
        },
    }

    def __init__(
        self,
        split,
        preprocess_dir,
        merge_classes_style=None,
        pselab_paths=None,
        short_run=False,
        reduce_factor=1,
    ):

        self.split = split
        self.preprocess_dir = preprocess_dir

        print("Initialize SemanticKITTI dataloader")

        print("Load", split)
        self.data = []
        for curr_split in split:
            with open(osp.join(self.preprocess_dir, curr_split + ".pkl"), "rb") as f:
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

        self.pselab_data = None
        if pselab_paths:
            # assert isinstance(pselab_paths, tuple)
            print("Load pseudo label data ", pselab_paths)
            self.pselab_data = []
            self.pselab_data.extend(np.load(pselab_paths, allow_pickle=True))

            # check consistency of data and pseudo labels
            assert len(self.pselab_data) == len(self.data)
            for i in range(len(self.pselab_data)):
                assert len(self.pselab_data[i]["pseudo_label_2d"]) == len(
                    self.data[i]["points"]
                )

            # refine 2d pseudo labels
            probs2d = np.concatenate([data["probs_2d"] for data in self.pselab_data])
            pseudo_label_2d = np.concatenate(
                [data["pseudo_label_2d"] for data in self.pselab_data]
            ).astype(np.int)
            pseudo_label_2d = refine_pseudo_labels(probs2d, pseudo_label_2d)

            # refine 3d pseudo labels
            # fusion model has only one final prediction saved in probs_2d
            if self.pselab_data[0]["probs_3d"] is not None:
                probs3d = np.concatenate(
                    [data["probs_3d"] for data in self.pselab_data]
                )
                pseudo_label_3d = np.concatenate(
                    [data["pseudo_label_3d"] for data in self.pselab_data]
                ).astype(np.int)
                pseudo_label_3d = refine_pseudo_labels(probs3d, pseudo_label_3d)
            else:
                pseudo_label_3d = None

            probs_ensemble = np.concatenate(
                [data["probs_ensemble"] for data in self.pselab_data]
            )
            pseudo_label_ensemble = np.concatenate(
                [data["pseudo_label_ensemble"] for data in self.pselab_data]
            ).astype(np.int)
            pseudo_label_ensemble = refine_pseudo_labels(
                probs_ensemble, pseudo_label_ensemble
            )

            # undo concat
            left_idx = 0
            for data_idx in range(len(self.pselab_data)):
                right_idx = left_idx + len(self.pselab_data[data_idx]["probs_2d"])
                self.pselab_data[data_idx]["pseudo_label_2d"] = pseudo_label_2d[
                    left_idx:right_idx
                ]
                if pseudo_label_3d is not None:
                    self.pselab_data[data_idx]["pseudo_label_3d"] = pseudo_label_3d[
                        left_idx:right_idx
                    ]
                else:
                    self.pselab_data[data_idx]["pseudo_label_3d"] = None
                self.pselab_data[data_idx][
                    "pseudo_label_ensemble"
                ] = pseudo_label_ensemble[left_idx:right_idx]

                left_idx = right_idx

        if merge_classes_style:
            highest_id = list(self.id_to_class_name.keys())[-1]
            self.label_mapping = -100 * np.ones(highest_id + 2, dtype=int)
            for cat_idx, cat_list in enumerate(
                self.categories[merge_classes_style].values()
            ):
                for class_name in cat_list:
                    self.label_mapping[self.class_name_to_id[class_name]] = cat_idx
            self.class_names = list(self.categories[merge_classes_style].keys())
        else:
            raise NotImplementedError(
                "The merge classes style needs to be provided, e.g. A2D2."
            )

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


class SemanticKITTISCN(SemanticKITTIBase):
    def __init__(
        self,
        split,
        preprocess_dir,
        semantic_kitti_dir="",
        pselab_paths=None,
        merge_classes_style=None,
        merge_classes=None,
        scale=20,
        full_scale=4096,
        image_normalizer=None,
        noisy_rot=0.0,  # 3D augmentation
        flip_x=0.0,  # 3D augmentation
        rot=0.0,  # 3D augmentation
        transl=False,  # 3D augmentation
        crop_size=tuple(),
        bottom_crop=False,
        rand_crop=tuple(),  # 2D augmentation
        fliplr=0.0,  # 2D augmentation
        color_jitter=None,  # 2D augmentation
        output_orig=False,
        resize=tuple(),
        downsample=(-1,),
        short_run=False,
        reduce_factor=1,
        camera_coords=False,
        use_rgb=False,
    ):
        super().__init__(
            split,
            preprocess_dir,
            merge_classes_style=merge_classes_style,
            pselab_paths=pselab_paths,
            short_run=short_run,
            reduce_factor=reduce_factor,
        )

        self.semantic_kitti_dir = semantic_kitti_dir
        self.output_orig = output_orig

        # point cloud parameters
        self.scale = scale
        self.full_scale = full_scale
        # 3D augmentation
        self.noisy_rot = noisy_rot
        self.flip_x = flip_x
        self.rot = rot
        self.transl = transl

        # image parameters
        self.image_normalizer = image_normalizer
        # 2D augmentation
        self.crop_size = crop_size
        if self.crop_size:
            assert bottom_crop != bool(
                rand_crop
            ), "Exactly one crop method needs to be active if crop size is provided!"
        else:
            assert (
                not bottom_crop and not rand_crop
            ), "No crop size, but crop method is provided is provided!"
        self.bottom_crop = bottom_crop
        self.rand_crop = np.array(rand_crop)
        assert len(self.rand_crop) in [0, 4]

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
        seg_label = data_dict["seg_labels"]
        intrinsics = data_dict["intrinsics"].copy()

        if seg_label is not None:
            seg_label = seg_label.astype(np.int64)

        if self.label_mapping is not None and seg_label is not None:
            seg_label = self.label_mapping[seg_label]

        out_dict = {}

        keep_idx = np.ones(len(points), dtype=np.bool)
        points_img = data_dict["points_img"].copy()
        img_path = osp.join(self.semantic_kitti_dir, data_dict["camera_path"])
        image = Image.open(img_path)

        if self.crop_size:
            # self.crop_size is a tuple (crop_width, crop_height)
            valid_crop = False
            for _ in range(10):
                if self.bottom_crop:
                    # self.bottom_crop is a boolean
                    left = int(
                        np.random.rand() * (image.size[0] + 1 - self.crop_size[0])
                    )
                    right = left + self.crop_size[0]
                    top = image.size[1] - self.crop_size[1]
                    bottom = image.size[1]
                elif len(self.rand_crop) > 0:
                    # self.rand_crop is a tuple of floats in interval (0, 1):
                    # (min_crop_height, max_crop_height, min_crop_width, max_crop_width)
                    crop_height, crop_width = self.rand_crop[0::2] + np.random.rand(
                        2
                    ) * (self.rand_crop[1::2] - self.rand_crop[0::2])
                    top = np.random.rand() * (1 - crop_height) * image.size[1]
                    left = np.random.rand() * (1 - crop_width) * image.size[0]
                    bottom = top + crop_height * image.size[1]
                    right = left + crop_width * image.size[0]
                    top, left, bottom, right = (
                        int(top),
                        int(left),
                        int(bottom),
                        int(right),
                    )

                # discard points outside of crop
                keep_idx = points_img[:, 0] >= top
                keep_idx = np.logical_and(keep_idx, points_img[:, 0] < bottom)
                keep_idx = np.logical_and(keep_idx, points_img[:, 1] >= left)
                keep_idx = np.logical_and(keep_idx, points_img[:, 1] < right)

                if np.sum(keep_idx) > 100:
                    valid_crop = True
                    break

            if valid_crop:
                # update intrinsics
                intrinsics[0, 2] -= top
                intrinsics[1, 2] -= left

                # crop image
                image = image.crop((left, top, right, bottom))
                points_img = points_img[keep_idx]
                points_img[:, 0] -= top
                points_img[:, 1] -= left

                # update point cloud
                points = points[keep_idx]
                pts_cam_coord = pts_cam_coord[keep_idx]
                if seg_label is not None:
                    seg_label = seg_label[keep_idx]

                if len(self.rand_crop) > 0:
                    # scale image points
                    points_img[:, 0] = (
                        float(self.crop_size[1])
                        / image.size[1]
                        * np.floor(points_img[:, 0])
                    )
                    points_img[:, 1] = (
                        float(self.crop_size[0])
                        / image.size[0]
                        * np.floor(points_img[:, 1])
                    )

                    # resize image (only during random crop, never during test)
                    image = image.resize(self.crop_size, Image.BILINEAR)
            else:
                print("No valid crop found for image", data_dict["camera_path"])

        img_indices = points_img.astype(np.int64)
        depth = np.zeros((image.size[1], image.size[0]))
        depth[img_indices[:, 0], img_indices[:, 1]] = pts_cam_coord[:, 2]
        seg_labels_2d = np.ones((image.size[1], image.size[0])) * (-100)
        seg_labels_2d[img_indices[:, 0], img_indices[:, 1]] = seg_label

        assert np.all(img_indices[:, 0] >= 0)
        assert np.all(img_indices[:, 1] >= 0)
        assert np.all(img_indices[:, 0] < image.size[1])
        assert np.all(img_indices[:, 1] < image.size[0])

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
        # Kitti lidar coordinates: x (front), y (left), z (up)
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

        if seg_label is not None:
            out_dict["seg_label"] = seg_label[idxs]
        out_dict["img_indices"] = out_dict["img_indices"][idxs]

        out_dict["depth"] = depth[None].astype(np.float32)
        out_dict["intrinsics"] = intrinsics
        out_dict["seg_labels_2d"] = seg_labels_2d
        out_dict["min_value"] = min_value
        out_dict["offset"] = offset
        out_dict["rot_matrix"] = rot_matrix

        if self.pselab_data is not None:
            out_dict["pseudo_label_2d"] = self.pselab_data[index]["pseudo_label_2d"][
                keep_idx
            ][idxs]
            if self.pselab_data[index]["pseudo_label_3d"] is None:
                out_dict["pseudo_label_3d"] = None
            else:
                out_dict["pseudo_label_3d"] = self.pselab_data[index][
                    "pseudo_label_3d"
                ][keep_idx][idxs]
            out_dict["pseudo_label_ensemble"] = self.pselab_data[index][
                "pseudo_label_ensemble"
            ][keep_idx][idxs]

        if self.output_orig:
            out_dict.update(
                {
                    "orig_seg_label": seg_label,
                    "orig_points_idx": idxs,
                }
            )

        if self.use_rgb:
            out_dict["feats"] = out_dict["img"][
                :, out_dict["img_indices"][:, 0], out_dict["img_indices"][:, 1]
            ].T
        else:
            out_dict["feats"] = np.ones(
                [len(idxs), 1], np.float32
            )  # simply use 1 as feature
        return out_dict
