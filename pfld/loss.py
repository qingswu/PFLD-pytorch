import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PFLDLoss(nn.Module):
    def __init__(self):
        super(PFLDLoss, self).__init__()

    def forward(
        self,
        attribute_gt,
        landmark_gt,
        euler_angle_gt,
        angle,
        landmarks,
    ):
        train_batchsize = landmark_gt.shape[0]
        weight_angle = torch.sum(1 - torch.cos(angle - euler_angle_gt), axis=1)
        attributes_w_n = attribute_gt[:, 1:6].float()
        mat_ratio = torch.mean(attributes_w_n, axis=0)
        mat_ratio = torch.Tensor(
            [1.0 / (x) if x > 0 else train_batchsize for x in mat_ratio]
        ).to(device)
        weight_attribute = torch.sum(attributes_w_n.mul(mat_ratio), axis=1)

        l2_distant = torch.sum(
            (landmark_gt - landmarks) * (landmark_gt - landmarks), axis=1
        )
        return torch.mean(weight_angle * weight_attribute * l2_distant), torch.mean(
            l2_distant
        )
