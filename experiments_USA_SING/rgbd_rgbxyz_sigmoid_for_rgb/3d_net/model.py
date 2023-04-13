import numpy as np
import torch
import torch.nn as nn
import torchvision

from .scn_unet import UNetSCN

# model definition

signature = (
    {
        "depth": np.zeros([1, 3, 3000], dtype=np.float32),
    },
    {"seg": np.zeros([1, 1, 3000], dtype=np.float32)},
)

dependencies = [
    f"torchvision=={torchvision.__version__}",
    f"numpy>={np.__version__}",
]


class Net3DSeg(nn.Module):
    def __init__(
        self,
        num_classes,
        dual_head,
        backbone_3d_kwargs,
    ):
        super(Net3DSeg, self).__init__()

        self.linear_rgb_mask = nn.Linear(3, 1)

        # 3D network
        self.net_3d = UNetSCN(**backbone_3d_kwargs)

        # segmentation head
        self.linear = nn.Linear(self.net_3d.out_channels, num_classes)

        # 2nd segmentation head
        self.dual_head = dual_head
        self.aux = L2G_classifier_3D(16, num_classes)

    def forward(self, data_batch):

        mask_rgb = self.linear_rgb_mask(data_batch["x"][1])
        mask_rgb = torch.sigmoid(mask_rgb)
        data_batch["x"][1] *= mask_rgb
        out_3D_feature = self.net_3d(data_batch["x"])
        x = self.linear(out_3D_feature)

        preds = {
            "seg_logit": x,
        }

        out_aux = self.aux(out_3D_feature)

        return preds, out_3D_feature, out_aux


class L2G_classifier_3D(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
    ):
        super(L2G_classifier_3D, self).__init__()

        # segmentation head
        self.linear_point = nn.Linear(input_channels, num_classes)
        self.linear_global = nn.Linear(input_channels, num_classes)
        self.dow = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        # self.dow_ = nn.AdaptiveAvgPool1d(8)

    def forward(self, input_3D_feature):
        # x = torch.transpose(input_3D_feature, 0, 1)
        # x = x.unsqueeze(0)
        # local_wise_line = self.dow(x).squeeze(0)
        # local_wise_line = torch.transpose(local_wise_line,0,1)

        # global_wise_line = self.dow_(x).squeeze(0)
        # global_wise_line = torch.transpose(global_wise_line, 0, 1)

        # linear
        point_wise_pre = self.linear_point(input_3D_feature)
        # local_wise_pre = self.linear(local_wise_line)
        # global_wise_pre = self.linear_global(global_wise_line)

        preds = {
            "feats": input_3D_feature,
            "seg_logit_point": point_wise_pre,
            # "seg_logit_global": global_wise_pre,
        }

        return preds
