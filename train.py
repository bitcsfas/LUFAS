import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataloader.oulu_dataloader import DataLoaderOULU
from dataloader.wmca_dataloader import DataLoaderWMCA
from model.model import DC_CDN, HybridModel as create_model 
from utils import train_one_epoch, evaluate
from sync_batchnorm import convert_model

import pandas as pd

def main(args):
    best_acer = 1

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()
    batch_size = args.batch_size

    if args.dataset == "oulu":
        # setup datasets
        oulu_root = "/media/disk3"
        oulu_train_list = oulu_root+"/OULU/Train_files_frames/categories-face-rotation.csv"
        oulu_val_list = oulu_root+"/OULU/Dev_files_frames/categories-face-rotation.csv"
        oulu_test_list = oulu_root+"/OULU/Test_files_frames/categories-face-rotation.csv"
        map_dir = True

        data_train = DataLoaderOULU(root=oulu_root, path=oulu_train_list, map_dir=map_dir, protocol=args.oulu_protocol, transform=True, imgType='hsv', imgSize=(256,256), if_alignment=True, enlarge=-1, training_state=0, sampling_size=args.sampling_size, leave_out_phone=1)
        train_loader= torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4)

        data_val = DataLoaderOULU(root=oulu_root, path=oulu_val_list, map_dir=True, protocol=args.oulu_protocol, transform=False, imgType='hsv', imgSize=(256,256), if_alignment=True, enlarge=-1, training_state=1, sampling_size=args.sampling_size, leave_out_phone=1)
        val_loader= torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False, num_workers=4)

        data_test = DataLoaderOULU(root=oulu_root, path=oulu_test_list, map_dir=True, protocol=args.oulu_protocol, transform=False, imgType='hsv', imgSize=(256,256), if_alignment=True, enlarge=-1, training_state=2, sampling_size=args.sampling_size, leave_out_phone=1)
        test_loader= torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=4)

    if args.dataset == "wmca":

        root = "/media/disk3"
        path ="/WMCA/preprocessed-face-station_RGB/face-station/"

        train_data = DataLoaderWMCA(root=root, path=path, map_dir=True, protocol=args.wmca_protocol, transform=True, imgType='hsv', imgSize=(128,128), training_state=0, sampling_size=args.sampling_size)
        train_loader= torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

        data_val = DataLoaderWMCA(root=root, path=path, map_dir=True, protocol=args.wmca_protocol, transform=True, imgType='hsv', imgSize=(128,128), training_state=1, sampling_size=args.sampling_size)
        val_loader= torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=True)

        test_data = DataLoaderWMCA(root=root, path=path, map_dir=True, protocol=args.wmca_protocol, transform=True, imgType='hsv', imgSize=(128,128), training_state=2, sampling_size=args.sampling_size)
        test_loader= torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    model = create_model(args.train_stage,args.vit_depths)
    # if len(args.gpu_list.split(',')) > 1 :
    #     model= convert_model(model)
    #     model= torch.nn.DataParallel(model, device_ids=list(map(int, args.gpu_list.split(','))) )
    model = model.cuda()

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights)

        model_dict = model.state_dict()
        state_dict = {k:v for k,v in weights_dict.items() if k in model_dict.keys()}
        print(state_dict.keys())
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    optimizer_Adam= optim.Adam(model.dccdn.parameters(), lr=args.lr, weight_decay=0.00005)
    optimizer_SGD = optim.SGD(model.vit.parameters(), lr=args.lr, momentum=0.9, weight_decay=5E-5)
            
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler_optimizer_Adam = lr_scheduler.LambdaLR(optimizer_Adam, lr_lambda=lf)
    scheduler_optimizer_SGD= lr_scheduler.LambdaLR(optimizer_SGD, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        loss_a,loss_b,loss_c, train_acc = train_one_epoch(model=model,
                                                optimizer_Adam=optimizer_Adam,
                                                optimizer_SGD=optimizer_SGD,
                                                data_loader=train_loader,
                                                epoch=epoch,
                                                train_stage=args.train_stage
                                                )

        scheduler_optimizer_Adam.step()
        scheduler_optimizer_SGD.step()
        print("Lr={}".format(scheduler_optimizer_Adam.get_last_lr()))

        if epoch % args.evaluate== args.evaluate-1:
            # validate
            val_ACC_cdcn,val_ACC_vit,val_APCER_cdcn,val_APCER_vit,val_BPCER_cdcn,val_BPCER_vit,val_ACER_cdcn,val_ACER_vit = evaluate(model=model,
                                        data_loader_val=val_loader,
                                        data_loader_test=test_loader,
                                        epoch=epoch,
                                        train_stage=args.train_stage,
                                        sampling_size=args.sampling_size,
                                        )

        # tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        # tb_writer.add_scalar(tags[0], train_loss, epoch)
        # tb_writer.add_scalar(tags[1], train_acc, epoch)
        # tb_writer.add_scalar(tags[2], val_loss, epoch)
        # tb_writer.add_scalar(tags[3], val_acc, epoch)
        # tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

            if args.train_stage == 1:
                # if best_acer>val_ACER_cdcn:
                torch.save(model.state_dict(), "./weights/{}-{}-APCER_{:.3f}-BPCER_{:.3f}-ACER_{:.3f}.pth".format(args.training_name,epoch,val_APCER_cdcn,val_BPCER_cdcn,val_ACER_cdcn))
                    # best_acer = val_ACER_cdcn
            else:
                # if best_acer>val_ACER_vit:
                torch.save(model.state_dict(), "./weights/{}-{}-APCER_{:.3f}-BPCER_{:.3f}-ACER_{:.3f}.pth".format(args.training_name,epoch,val_APCER_vit,val_BPCER_vit,val_ACER_vit))
                    # best_acer = val_ACER_vit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    parser.add_argument('--training-name', default='', help='create model name')
    parser.add_argument('--weights', type=str, default='',help='initial weights path')
    parser.add_argument('--oulu-protocol', type=str, default='protocol_1',help='oulu protocol')
    parser.add_argument('--wmca-protocol', type=str, default='LOO_fakehead',help='oulu protocol')
    parser.add_argument('--hqwmca-protocol', type=str, default='LOO_fakehead',help='oulu protocol')

    parser.add_argument('--vit-depths', type=int, default=4)
    parser.add_argument('--train-stage', type=int, default=1)
    parser.add_argument('--dataset', type=str, default="oulu")
    parser.add_argument('--sampling-size', type=int, default=5)
    parser.add_argument('--evaluate', type=int, default=10)
    parser.add_argument('--gpu_list', type=str, default="0", help="the list of gpu id seprated by commas")

    opt = parser.parse_args()

    main(opt)
