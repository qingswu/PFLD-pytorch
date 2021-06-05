import logging

import numpy as np
import torch

from pfld.utils import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    logging.info("Save checkpoint to {0:}".format(filename))


def train(
    train_loader,
    pfld_backbone,
    auxiliarynet,
    criterion,
    optimizer,
    epoch,
    train_batchsize,
):
    losses = AverageMeter()

    weighted_loss, loss = None, None
    for img, landmark_gt, attribute_gt, euler_angle_gt in train_loader:
        img = img.to(device)
        attribute_gt = attribute_gt.to(device)
        landmark_gt = landmark_gt.to(device)
        euler_angle_gt = euler_angle_gt.to(device)
        pfld_backbone = pfld_backbone.to(device)
        auxiliarynet = auxiliarynet.to(device)
        features, landmarks = pfld_backbone(img)
        angle = auxiliarynet(features)
        weighted_loss, loss = criterion(
            attribute_gt,
            landmark_gt,
            euler_angle_gt,
            angle,
            landmarks,
            train_batchsize,
        )
        logging.info("loss={}".format(weighted_loss.detach().cpu().item()))
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        losses.update(loss.item())
    return weighted_loss, loss


def validate(wlfw_val_dataloader, pfld_backbone, auxiliarynet, criterion):
    pfld_backbone.eval()
    auxiliarynet.eval()
    losses = []
    with torch.no_grad():
        for img, landmark_gt, attribute_gt, euler_angle_gt in wlfw_val_dataloader:
            img = img.to(device)
            attribute_gt = attribute_gt.to(device)
            landmark_gt = landmark_gt.to(device)
            euler_angle_gt = euler_angle_gt.to(device)
            pfld_backbone = pfld_backbone.to(device)
            auxiliarynet = auxiliarynet.to(device)
            _, landmark = pfld_backbone(img)
            loss = torch.mean(torch.sum((landmark_gt - landmark) ** 2, axis=1))
            losses.append(loss.cpu().numpy())
    logging.info("Eval set: Average loss: {:.4f} ".format(np.mean(losses)))
    return np.mean(losses)
