"""
Load many available datasets
"""

import random
from dataclasses import dataclass
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler

from lib.dataset.a2d2 import A2D2SCN
from lib.dataset.nuscenes_dataloader import NuScenesLidarSegSCN
from lib.dataset.semantic_kitti import SemanticKITTISCN
from lib.dataset.virtual_kitti_dataloader import VirtualKITTISCN
from lib.utils.sampler import IterationBasedBatchSampler


def collate_scn_base(input_dict_list, output_orig, output_image=True):
    """
    Custom collate function for SCN. The batch size is always 1,
    but the batch indices are appended to the locations.
    :param input_dict_list: a list of dicts from the dataloader
    :param output_orig: whether to output original point cloud/labels/indices
    :param output_image: whether to output images
    :return: Collated data batch as dict
    """
    locs = []
    feats = []
    labels = []
    depths = []
    voxel_coords = []
    intrinsics = []
    seg_labels_2d = []
    min_values = []
    rotation_matrices = []
    offsets = []
    points_batch = []

    if output_image:
        imgs = []
        img_idxs = []

    if output_orig:
        orig_seg_label = []
        orig_points_idx = []

    output_pselab = "pseudo_label_2d" in input_dict_list[0].keys()
    if output_pselab:
        pseudo_label_2d = []
        pseudo_label_3d = []
        pseudo_label_ensemble = []

    for idx, input_dict in enumerate(input_dict_list):
        coords = torch.from_numpy(input_dict["coords"])
        points_batch.append(torch.from_numpy(input_dict["points"]))
        voxel_coords.append(coords)
        batch_idxs = torch.LongTensor(coords.shape[0], 1).fill_(idx)
        locs.append(torch.cat([coords, batch_idxs], 1))
        feats.append(torch.from_numpy(input_dict["feats"]))
        if "seg_label" in input_dict.keys():
            labels.append(torch.from_numpy(input_dict["seg_label"]))
        if output_image:
            imgs.append(torch.from_numpy(input_dict["img"]))
            depths.append(torch.from_numpy(input_dict["depth"]))
            img_idxs.append(input_dict["img_indices"])
        if output_orig:
            orig_seg_label.append(input_dict["orig_seg_label"])
            orig_points_idx.append(input_dict["orig_points_idx"])
        if output_pselab:
            pseudo_label_2d.append(torch.from_numpy(input_dict["pseudo_label_2d"]))
            if input_dict["pseudo_label_3d"] is not None:
                pseudo_label_3d.append(torch.from_numpy(input_dict["pseudo_label_3d"]))
            pseudo_label_ensemble.append(
                torch.from_numpy(input_dict["pseudo_label_ensemble"])
            )
        intrinsics.append(torch.from_numpy(input_dict["intrinsics"]))
        seg_labels_2d.append(torch.from_numpy(input_dict["seg_labels_2d"]))
        rotation_matrices.append(torch.from_numpy(input_dict["rot_matrix"]))
        min_values.append(torch.from_numpy(input_dict["min_value"]))
        offsets.append(torch.from_numpy(input_dict["offset"]))

    locs = torch.cat(locs, 0)
    feats = torch.cat(feats, 0)
    points_batch = torch.cat(points_batch, 0)
    voxel_coords = torch.cat(voxel_coords, 0)
    out_dict = {
        "x": [locs, feats],
        "rotation_matrices": torch.stack(rotation_matrices),
        "min_values": torch.stack(min_values),
        "offsets": torch.stack(offsets),
        "points": points_batch,
    }
    out_dict["coords"] = voxel_coords
    out_dict["intrinsics"] = torch.stack(intrinsics)
    out_dict["seg_labels_2d"] = torch.stack(seg_labels_2d).float()

    if labels:
        labels = torch.cat(labels, 0)
        out_dict["seg_label"] = labels
    if output_image:
        out_dict["img"] = torch.stack(imgs)
        out_dict["img_indices"] = img_idxs
        out_dict["depth"] = torch.stack(depths)
    if output_orig:
        out_dict["orig_seg_label"] = orig_seg_label
        out_dict["orig_points_idx"] = orig_points_idx
    if output_pselab:
        out_dict["pseudo_label_2d"] = torch.cat(pseudo_label_2d, 0)
        out_dict["pseudo_label_3d"] = (
            torch.cat(pseudo_label_3d, 0) if pseudo_label_3d else pseudo_label_3d
        )
        out_dict["pseudo_label_ensemble"] = torch.cat(pseudo_label_ensemble, 0)

    return out_dict


def get_collate_scn(is_train):
    return partial(
        collate_scn_base,
        output_orig=not is_train,
    )


# def set_random_seed(seed):
#     if seed < 0:
#         return
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     # torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.

    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

    """
    # base_seed = torch.IntTensor(1).random_().item()
    # print(worker_id, base_seed)
    np.random.seed(worker_id)


def load_datamodule(
    name: str,
    cfg_source: dict,
    cfg_target: dict,
    batch_size: int = 1,
    num_workers: int = cpu_count() // 4,
    max_iterations: int = 1,
    ds_args: dict = None,
    augmentations: dict = None,
    short_run: bool = False,
    reduce_factor: int = 1,
    pselab_paths=None,
):

    # choose dataset
    match name:
        case "nuscenes":
            train_ds_source = NuScenesLidarSegSCN(
                split=cfg_source.TRAIN,
                preprocess_dir=cfg_source.preprocess_dir,
                nuscenes_dir=cfg_source.nuscenes_dir,
                output_orig=False,
                short_run=short_run,
                reduce_factor=reduce_factor,
                **ds_args,
                **augmentations,
            )
            print(pselab_paths)
            train_ds_target = NuScenesLidarSegSCN(
                split=cfg_target.TRAIN,
                preprocess_dir=cfg_target.preprocess_dir,
                nuscenes_dir=cfg_target.nuscenes_dir,
                output_orig=False,
                short_run=short_run,
                reduce_factor=reduce_factor,
                pselab_paths=pselab_paths,
                **ds_args,
                **augmentations,
            )

            val_ds_target = NuScenesLidarSegSCN(
                split=cfg_target.VAL,
                preprocess_dir=cfg_source.preprocess_dir,
                nuscenes_dir=cfg_source.nuscenes_dir,
                output_orig=True,
                **ds_args,
            )
            test_ds = NuScenesLidarSegSCN(
                split=cfg_target.TEST,
                preprocess_dir=cfg_source.preprocess_dir,
                nuscenes_dir=cfg_source.nuscenes_dir,
                output_orig=True,
                **ds_args,
            )
        case "ad2d_semantic_kitti":
            train_ds_source = A2D2SCN(
                split=cfg_source.TRAIN,
                preprocess_dir=cfg_source.preprocess_dir,
                short_run=short_run,
                reduce_factor=reduce_factor,
                **ds_args,
                **augmentations,
            )

            train_ds_target = SemanticKITTISCN(
                split=cfg_target.TRAIN,
                preprocess_dir=cfg_target.preprocess_dir,
                semantic_kitti_dir=cfg_target.semantic_kitti_dir,
                output_orig=False,
                short_run=short_run,
                reduce_factor=reduce_factor,
                pselab_paths=pselab_paths,
                **ds_args,
                **augmentations,
            )

            val_ds_target = SemanticKITTISCN(
                split=cfg_target.VAL,
                preprocess_dir=cfg_target.preprocess_dir,
                semantic_kitti_dir=cfg_target.semantic_kitti_dir,
                output_orig=True,
                **ds_args,
            )
            test_ds = SemanticKITTISCN(
                split=cfg_target.TEST,
                preprocess_dir=cfg_target.preprocess_dir,
                semantic_kitti_dir=cfg_target.semantic_kitti_dir,
                output_orig=True,
                **ds_args,
            )
        case "vkitti_semantic_kitti":
            train_ds_source = VirtualKITTISCN(
                split=cfg_source.TRAIN,
                preprocess_dir=cfg_source.preprocess_dir,
                virtual_kitti_dir=cfg_source.virtual_kitti_dir,
                short_run=short_run,
                reduce_factor=reduce_factor,
                **ds_args,
                **augmentations,
            )

            train_ds_target = SemanticKITTISCN(
                split=cfg_target.TRAIN,
                preprocess_dir=cfg_target.preprocess_dir,
                semantic_kitti_dir=cfg_target.semantic_kitti_dir,
                output_orig=False,
                short_run=short_run,
                reduce_factor=reduce_factor,
                pselab_paths=pselab_paths,
                **ds_args,
                **augmentations,
            )

            val_ds_target = SemanticKITTISCN(
                split=cfg_target.VAL,
                preprocess_dir=cfg_target.preprocess_dir,
                semantic_kitti_dir=cfg_target.semantic_kitti_dir,
                output_orig=True,
                **ds_args,
            )
            test_ds = SemanticKITTISCN(
                split=cfg_target.TEST,
                preprocess_dir=cfg_target.preprocess_dir,
                semantic_kitti_dir=cfg_target.semantic_kitti_dir,
                output_orig=True,
                **ds_args,
            )

        case other:
            raise ValueError(f"not found datamodule {other}")

    return _DataModule(
        train_ds_source=train_ds_source,
        train_ds_target=train_ds_target,
        # val_ds_source=val_ds_source,
        val_ds_target=val_ds_target,
        test_ds=test_ds,
        batch_size=batch_size,
        max_iterations=max_iterations,
        num_workers=num_workers,
    )


@dataclass
class _DataModule(pl.LightningDataModule):
    train_ds_source: Dataset | dict[str, Dataset] | None = None
    train_ds_target: Dataset | dict[str, Dataset] | None = None
    # val_ds_source: Dataset | dict[str, Dataset] | None = None
    val_ds_target: Dataset | dict[str, Dataset] | None = None
    test_ds: Dataset | dict[str, Dataset] | None = None
    max_iterations: int = 100000
    batch_size: int = 1
    num_workers: int = cpu_count() // 2

    def train_dataloader(self):
        # sampler_source = RandomSampler(self.train_ds_source)
        # batch_sampler_source = BatchSampler(
        #     sampler_source, batch_size=self.batch_size, drop_last=True
        # )
        # batch_sampler_source = IterationBasedBatchSampler(
        #     batch_sampler_source, self.max_iterations, 0
        # )

        # sampler_target = RandomSampler(self.train_ds_target)
        # batch_sampler_target = BatchSampler(
        #     sampler_target, batch_size=self.batch_size, drop_last=True
        # )
        # batch_sampler_target = IterationBasedBatchSampler(
        #     batch_sampler_target, self.max_iterations, 0
        # )

        collate_fn = get_collate_scn(is_train=True)
        source_dl = DataLoader(
            self.train_ds_source,
            # batch_sampler=batch_sampler_source,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
        )
        target_dl = DataLoader(
            self.train_ds_target,
            # batch_sampler=batch_sampler_target,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
        )
        return {"source": source_dl, "target": target_dl}

    def val_dataloader(self):
        collate_fn = get_collate_scn(is_train=False)
        # source_dl = DataLoader(
        #     self.val_ds_source,
        #     batch_size=self.batch_size,
        #     drop_last=False,
        #     num_workers=self.num_workers // 2,
        #     worker_init_fn=worker_init_fn,
        #     collate_fn=collate_fn,
        # )
        target_dl = DataLoader(
            self.val_ds_target,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers // 2,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

        target_dl_test = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers // 2,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

        return [target_dl, target_dl_test]

    def test_dataloader(self):
        collate_fn = get_collate_scn(is_train=False)
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers // 2,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )
