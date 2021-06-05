#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

from pfld.dataset.datasets import WLFWTarDatasets
from pfld.models.pfld import PFLDInference, AuxiliaryNet
from pfld.loss import PFLDLoss
from pfld.trainer import train, validate, save_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_args(args):
    for arg in vars(args):
        s = arg + ": " + str(getattr(args, arg))
        logging.info(s)


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")


def main(args):
    # Step 1: parse args config
    logging.basicConfig(
        format="[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode="w"),
            logging.StreamHandler(),
        ],
    )
    print_args(args)

    # Step 2: model, criterion, optimizer, scheduler
    pfld_backbone = PFLDInference().to(device)
    auxiliarynet = AuxiliaryNet().to(device)
    criterion = PFLDLoss()
    optimizer = torch.optim.Adam(
        [{"params": pfld_backbone.parameters()}, {"params": auxiliarynet.parameters()}],
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.lr_patience, verbose=True
    )
    if args.resume:
        checkpoint = torch.load(args.resume)
        auxiliarynet.load_state_dict(checkpoint["auxiliarynet"])
        pfld_backbone.load_state_dict(checkpoint["pfld_backbone"])
        args.start_epoch = checkpoint["epoch"]

    # step 3: data
    # argumetion
    transform = transforms.Compose([transforms.ToTensor()])
    wlfwdataset = WLFWTarDatasets(
        "D:/Documents/Drive/WLFW/WLFW_data.tar", "train_data/list.txt", transform
    )
    dataloader = DataLoader(
        wlfwdataset,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False,
    )

    wlfw_val_dataset = WLFWTarDatasets(
        "D:/Documents/Drive/WLFW/WLFW_data.tar", "test_data/list.txt", transform
    )
    wlfw_val_dataloader = DataLoader(
        wlfw_val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.workers,
    )

    # step 4: run
    writer = SummaryWriter(args.tensorboard)
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        weighted_train_loss, train_loss = train(
            dataloader,
            pfld_backbone,
            auxiliarynet,
            criterion,
            optimizer,
            epoch,
            args.train_batchsize,
        )
        filename = os.path.join(
            str(args.snapshot), "checkpoint_epoch_" + str(epoch) + ".pth.tar"
        )
        save_checkpoint(
            {
                "epoch": epoch,
                "pfld_backbone": pfld_backbone.state_dict(),
                "auxiliarynet": auxiliarynet.state_dict(),
            },
            filename,
        )

        val_loss = validate(wlfw_val_dataloader, pfld_backbone, auxiliarynet, criterion)

        scheduler.step(val_loss)
        writer.add_scalar("data/weighted_loss", weighted_train_loss, epoch)
        writer.add_scalars(
            "data/loss", {"val loss": val_loss, "train loss": train_loss}, epoch
        )
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="pfld")
    # general
    parser.add_argument("-j", "--workers", default=4, type=int)
    parser.add_argument("--devices_id", default="0", type=str)  # TBD
    parser.add_argument("--test_initial", default="false", type=str2bool)  # TBD

    # training
    ##  -- optimizer
    parser.add_argument("--base_lr", default=0.0001, type=int)
    parser.add_argument("--weight-decay", "--wd", default=1e-6, type=float)

    # -- lr
    parser.add_argument("--lr_patience", default=40, type=int)

    # -- epoch
    parser.add_argument("--start_epoch", default=1, type=int)
    parser.add_argument("--end_epoch", default=500, type=int)

    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument(
        "--snapshot", default="checkpoint/snapshot/", type=str, metavar="PATH"
    )
    parser.add_argument("--log_file", default="checkpoint/train.logs", type=str)
    parser.add_argument("--tensorboard", default="checkpoint/tensorboard", type=str)
    parser.add_argument("--resume", default="", type=str, metavar="PATH")

    # --dataset
    parser.add_argument(
        "--dataroot", default="C:/my_temp/WLFW_data/", type=str, metavar="PATH"
    )
    parser.add_argument("--train_batchsize", default=64, type=int)
    parser.add_argument("--val_batchsize", default=64, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
