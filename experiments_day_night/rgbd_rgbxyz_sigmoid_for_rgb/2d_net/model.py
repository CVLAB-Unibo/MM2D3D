from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import FrozenBatchNorm2d

from .backbones import Backbone

# each model must expose a signature describing its input
# and outputs

signature = (
    {
        "img": np.zeros([1, 3, 480, 640], dtype=np.float32),
    },
    {
        "segm": np.zeros([1, 1, 480, 640], dtype=np.float32),
    },
)

# its model must expose its dependencies, except for pytorch which
# is implicit

dependencies = [
    f"torchvision=={torchvision.__version__}",
    f"numpy>={np.__version__}",
]

# model definition


class Net2DSeg(nn.Module):
    def __init__(self, num_classes, pretrained=True, frozen_batch_norm=False):
        super(Net2DSeg, self).__init__()
        feat_channels = 64

        norm_layer = FrozenBatchNorm2d if frozen_batch_norm else None
        self.rgb_backbone = Backbone(pretrained=pretrained, norm_layer=norm_layer)
        self.depth_backbone = Backbone(num_channel=1, pretrained=False)
        # Decoder Sem
        _, self.dec_t_conv_stage5 = self.dec_stage(
            self.rgb_backbone.layer4, num_concat=1, num_concat_t=2
        )
        self.dec_conv_stage4, self.dec_t_conv_stage4 = self.dec_stage(
            self.rgb_backbone.layer3, num_concat=3
        )
        self.dec_conv_stage3, self.dec_t_conv_stage3 = self.dec_stage(
            self.rgb_backbone.layer2, num_concat=3
        )
        self.dec_conv_stage2, self.dec_t_conv_stage2 = self.dec_stage(
            self.rgb_backbone.layer1, num_concat=3
        )
        self.dec_conv_stage1 = nn.Conv2d(3 * 64, 64, kernel_size=3, padding=1)

        # classifier
        self.dow_avg = nn.AvgPool2d((5, 5), stride=(1, 1), padding=(2, 2))
        self.con1_1_avg = nn.Conv2d(64, num_classes, kernel_size=1, stride=1)

        self.aux = L2G_classifier_2D(feat_channels, num_classes)

    @staticmethod
    def dec_stage(enc_stage, num_concat, num_concat_t=1):
        in_channels = enc_stage[0].conv1.in_channels
        out_channels = enc_stage[-1].conv2.out_channels
        conv = nn.Sequential(
            nn.Conv2d(
                num_concat * out_channels, out_channels, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        t_conv = nn.Sequential(
            nn.ConvTranspose2d(
                out_channels * num_concat_t, in_channels, kernel_size=2, stride=2
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        return conv, t_conv

    def forward(self, data_batch):
        # (batch_size, 3, H, W)
        img = data_batch["img"]
        img_indices = data_batch["img_indices"]
        hints = data_batch["depth"]

        h, w = img.shape[2], img.shape[3]
        min_size = 16
        pad_h = int((h + min_size - 1) / min_size) * min_size - h
        pad_w = int((w + min_size - 1) / min_size) * min_size - w
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, [0, pad_w, 0, pad_h])
            hints = F.pad(hints, [0, pad_w, 0, pad_h])

        # encode
        inter_features = self.rgb_backbone(img)
        inter_features_depth = self.depth_backbone(hints)

        # decode segmentation **************************************************
        # upsample
        segm = self.dec_t_conv_stage5(
            torch.cat([inter_features_depth[4], inter_features[4]], dim=1)
        )
        segm = torch.cat([inter_features_depth[3], segm, inter_features[3]], dim=1)
        segm = self.dec_conv_stage4(segm)

        # upsample
        segm = self.dec_t_conv_stage4(segm)
        segm = torch.cat([inter_features_depth[2], segm, inter_features[2]], dim=1)
        segm = self.dec_conv_stage3(segm)

        # upsample
        segm = self.dec_t_conv_stage3(segm)
        segm = torch.cat([inter_features_depth[1], segm, inter_features[1]], dim=1)
        segm = self.dec_conv_stage2(segm)

        # upsample
        segm = self.dec_t_conv_stage2(segm)
        segm = torch.cat([inter_features_depth[0], segm, inter_features[0]], dim=1)
        segm = self.dec_conv_stage1(segm)

        # crop padding
        if pad_h > 0 or pad_w > 0:
            segm_last = segm[:, :, 0:h, 0:w]

        segm = self.dow_avg(segm_last)
        segm = self.con1_1_avg(segm)
        # 2D-3D feature lifting
        img_feats = []
        for i in range(segm.shape[0]):
            img_feats.append(
                segm.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]]
            )
        img_feats = torch.cat(img_feats, 0)

        aux_out = self.aux(segm_last, img_indices)
        preds = {"seg_logit": img_feats, "seg_logit_2d": segm}

        return preds, segm_last, img_indices, aux_out


class L2G_classifier_2D(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
    ):
        super(L2G_classifier_2D, self).__init__()

        # segmentation head
        self.con1_1_avg = nn.Conv2d(
            input_channels, num_classes, kernel_size=1, stride=1
        )
        self.linear = nn.Linear(input_channels, num_classes)
        self.dow_avg = nn.AvgPool2d((5, 5), stride=(1, 1), padding=(2, 2))

    def forward(self, input_2D_feature, img_indices):
        # (batch_size, 3, H, W)

        avg_feature = self.dow_avg(input_2D_feature)
        avg_feature = self.con1_1_avg(avg_feature)

        avg_line = []
        for i in range(avg_feature.shape[0]):
            avg_line.append(
                avg_feature.permute(0, 2, 3, 1)[i][
                    img_indices[i][:, 0], img_indices[i][:, 1]
                ]
            )
        avg_line = torch.cat(avg_line, 0)

        preds = {
            "seg_logit_avg": avg_line,
            "seg_logit_avg_2d": avg_feature,
        }

        return preds
