import os
import os.path as osp
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nuscenes.eval.lidarseg.utils import LidarsegClassMapper
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

lib_path = Path(__file__).parent.parent.parent
print(lib_path)
sys.path.append(str(lib_path))
import lib.dataset.nuscenes_splits as splits


# modified from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py
def map_pointcloud_to_image(pc, im_shape, info, im=None):
    """
    Maps the lidar point cloud to the image.
    :param pc: (3, N)
    :param im_shape: image to check size and debug
    :param info: dict with calibration infos
    :param im: image, only for visualization
    :return:
    """
    pc = pc.copy()

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
    pc = Quaternion(info["lidar2ego_rotation"]).rotation_matrix @ pc
    pc = pc + np.array(info["lidar2ego_translation"])[:, np.newaxis]

    # Second step: transform to the global frame.
    pc = Quaternion(info["ego2global_rotation_lidar"]).rotation_matrix @ pc
    pc = pc + np.array(info["ego2global_translation_lidar"])[:, np.newaxis]

    # Third step: transform into the ego vehicle frame for the timestamp of the image.
    pc = pc - np.array(info["ego2global_translation_cam"])[:, np.newaxis]
    pc = Quaternion(info["ego2global_rotation_cam"]).rotation_matrix.T @ pc

    # Fourth step: transform into the camera.
    pc = pc - np.array(info["cam2ego_translation"])[:, np.newaxis]
    pc = Quaternion(info["cam2ego_rotation"]).rotation_matrix.T @ pc

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc[2, :]

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc, np.array(info["cam_intrinsic"]), normalize=True)

    # Cast to float32 to prevent later rounding errors
    points = points.astype(np.float32)

    # Remove points that are either outside or behind the camera.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    mask = np.logical_and(mask, points[0, :] > 0)
    mask = np.logical_and(mask, points[0, :] < im_shape[1])
    mask = np.logical_and(mask, points[1, :] > 0)
    mask = np.logical_and(mask, points[1, :] < im_shape[0])
    points = points[:, mask]

    # debug
    if im is not None:
        # Retrieve the color from the depth.
        coloring = depths
        coloring = coloring[mask]

        plt.figure(figsize=(9, 16))
        plt.imshow(im)
        plt.scatter(points[0, :], points[1, :], c=coloring, s=2)
        plt.axis("off")

        # plt.show()

    return mask, pc, points.T[:, :2]


def preprocess(
    nusc,
    split_names,
    root_dir,
    out_dir,
    keyword=None,
    keyword_action=None,
    subset_name=None,
    location=None,
):
    # cannot process day/night and location at the same time
    assert not (bool(keyword) and bool(location))
    if keyword:
        assert keyword_action in ["filter", "exclude"]

    # init dict to save
    pkl_dict = {}
    for split_name in split_names:
        pkl_dict[split_name] = []

    # fine to coarse label mapping (only 16 classes out of 32 are actually used in NuScenes lidarseg)
    class_mapper = LidarsegClassMapper(nusc)
    fine_2_carse_mapping_dict = class_mapper.get_fine_idx_2_coarse_idx()
    fine_2_coarse_mapping = np.array(
        [
            fine_2_carse_mapping_dict[fine_idx]
            for fine_idx in range(len(fine_2_carse_mapping_dict))
        ]
    )

    for i, sample in enumerate(nusc.sample):
        curr_scene_name = nusc.get("scene", sample["scene_token"])["name"]

        # get if the current scene is in train, val or test
        curr_split = None
        for split_name in split_names:
            if curr_scene_name in getattr(splits, split_name):
                curr_split = split_name
                break
        if curr_split is None:
            continue

        if subset_name == "night":
            if curr_split == "train":
                if curr_scene_name in splits.val_night:
                    curr_split = "val"
        if subset_name == "singapore":
            if curr_split == "train":
                if curr_scene_name in splits.val_singapore:
                    curr_split = "val"
        if subset_name == "all":
            if curr_split == "train":
                if curr_scene_name in splits.val_all:
                    curr_split = "val"

        # filter for day/night
        if keyword:
            scene_description = nusc.get("scene", sample["scene_token"])["description"]
            if keyword.lower() in scene_description.lower():
                if keyword_action == "exclude":
                    # skip sample
                    continue
            else:
                if keyword_action == "filter":
                    # skip sample
                    continue

        if location:
            scene = nusc.get("scene", sample["scene_token"])
            if location not in nusc.get("log", scene["log_token"])["location"]:
                continue

        lidar_token = sample["data"]["LIDAR_TOP"]
        cam_front_token = sample["data"]["CAM_FRONT"]
        lidar_path, _, _ = nusc.get_sample_data(lidar_token)
        cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_front_token)

        print(
            "{}/{} {} {}, current split: {}".format(
                i + 1, len(nusc.sample), curr_scene_name, lidar_path, curr_split
            )
        )

        sd_rec_lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cs_record_lidar = nusc.get(
            "calibrated_sensor", sd_rec_lidar["calibrated_sensor_token"]
        )
        pose_record_lidar = nusc.get("ego_pose", sd_rec_lidar["ego_pose_token"])
        sd_rec_cam = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        cs_record_cam = nusc.get(
            "calibrated_sensor", sd_rec_cam["calibrated_sensor_token"]
        )
        pose_record_cam = nusc.get("ego_pose", sd_rec_cam["ego_pose_token"])

        calib_infos = {
            "lidar2ego_translation": cs_record_lidar["translation"],
            "lidar2ego_rotation": cs_record_lidar["rotation"],
            "ego2global_translation_lidar": pose_record_lidar["translation"],
            "ego2global_rotation_lidar": pose_record_lidar["rotation"],
            "ego2global_translation_cam": pose_record_cam["translation"],
            "ego2global_rotation_cam": pose_record_cam["rotation"],
            "cam2ego_translation": cs_record_cam["translation"],
            "cam2ego_rotation": cs_record_cam["rotation"],
            "cam_intrinsic": cam_intrinsic,
        }

        # load lidar points
        pts = (
            np.fromfile(lidar_path, dtype=np.float32, count=-1)
            .reshape([-1, 5])[:, :3]
            .T
        )

        # map point cloud into front camera image
        pts_valid_flag, pts_cam_coord, pts_img = map_pointcloud_to_image(
            pts, (900, 1600, 3), calib_infos
        )
        # fliplr so that indexing is row, col and not col, row
        pts_img = np.ascontiguousarray(np.fliplr(pts_img))

        # only use lidar points in the front camera image
        pts = pts[:, pts_valid_flag]
        pts_cam_coord = pts_cam_coord[:, pts_valid_flag]

        # load segmentation labels
        lidarseg_labels_filename = osp.join(
            nusc.dataroot, nusc.get("lidarseg", lidar_token)["filename"]
        )
        seg_labels = np.fromfile(
            lidarseg_labels_filename, dtype=np.uint8
        )  # [num_points]
        seg_labels = seg_labels[
            pts_valid_flag
        ]  # discard labels of points outside of FoV
        seg_labels = fine_2_coarse_mapping[seg_labels]  # map from fine to coarse labels

        # convert to relative path
        lidar_path = lidar_path.replace(root_dir + "/", "")
        cam_path = cam_path.replace(root_dir + "/", "")

        # transpose to yield shape (num_points, 3)
        pts = pts.T
        pts_cam_coord = pts_cam_coord.T

        # append data to train, val or test list in pkl_dict
        data_dict = {
            "points": pts,
            "seg_labels": seg_labels.astype(np.uint8),
            "points_img": pts_img,  # row, col format, shape: (num_points, 2)
            "lidar_path": lidar_path,
            "camera_path": cam_path,
            "sample_token": sample["token"],
            "scene_name": curr_scene_name,
            "calib": calib_infos,
            "pts_cam_coord": pts_cam_coord,
        }
        pkl_dict[curr_split].append(data_dict)

    # save to pickle file
    save_dir = osp.join(out_dir, "preprocess")
    os.makedirs(save_dir, exist_ok=True)
    for split_name in split_names:
        save_path = osp.join(
            save_dir,
            "{}{}.pkl".format(split_name, "_" + subset_name if subset_name else ""),
        )
        with open(save_path, "wb") as f:
            pickle.dump(pkl_dict[split_name], f)
            print("Wrote preprocessed data to " + save_path)


if __name__ == "__main__":
    root_dir = "data/nuscenes"
    out_dir = "data/nuscenes/preprocessed_nuscenes_lidarseg"
    nusc = NuScenes(version="v1.0-trainval", dataroot=root_dir, verbose=True)
    # for faster debugging, the script can be run using the mini dataset
    # nusc = NuScenes(version='v1.0-mini', dataroot=root_dir, verbose=True)
    # We construct the splits by using the meta data of NuScenes:
    # USA/Singapore: We check if the location is Boston or Singapore.
    # Day/Night: We detect if "night" occurs in the scene description string.
    preprocess(
        nusc,
        ["train", "val", "test"],
        root_dir,
        out_dir,
        location="boston",
        subset_name="usa",
    )
    preprocess(
        nusc,
        ["train", "val", "test"],
        root_dir,
        out_dir,
        location="singapore",
        subset_name="singapore",
    )
    preprocess(
        nusc,
        ["train", "val", "test"],
        root_dir,
        out_dir,
        keyword="night",
        keyword_action="exclude",
        subset_name="day",
    )
    preprocess(
        nusc,
        ["train", "val", "test"],
        root_dir,
        out_dir,
        keyword="night",
        keyword_action="filter",
        subset_name="night",
    )
