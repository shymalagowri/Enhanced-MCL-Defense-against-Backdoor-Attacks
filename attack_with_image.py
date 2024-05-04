
import torch
import torch.nn as nn
import pandas as pd
from utils.util import AverageMeter, accuracy
from data_loader import get_test_loader, get_backdoor_loader
from config import get_arguments
from models.selector import select_model
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os

def train_step(opt, train_loader, nets, optimizer, criterions, epoch):
    cls_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets['snet']

    criterionCls = criterions['criterionCls']
    snet.train()

    for idx, (img, target) in enumerate(train_loader, start=1):
        if opt.cpu:
            img = img.cpu()
            target = target.cpu()

        output_s = snet(img)

        cls_loss = criterionCls(output_s, target)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        cls_loss.backward()
        optimizer.step()

        if idx % opt.print_freq == 0:
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                  'cls_loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=cls_losses, top1=top1, top5=top5))

def show_image(img, title):
    plt.imshow(transforms.ToPILImage()(img[0]))  # Assuming img is a tensor in (C, H, W) format
    plt.title(title)
    plt.show()

def test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch):
    test_process = []
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets['snet']
    criterionCls = criterions['criterionCls']
    snet.eval()

    for idx, (img, target) in enumerate(test_clean_loader, start=1):
        img = img.cpu()
        target = target.cpu()

        with torch.no_grad():
            output_s = snet(img)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        if idx % opt.print_freq == 0:
            print('[Clean Sample] Target: {}, Predicted: {}'.format(target[0].item(), output_s.argmax(dim=1)[0].item()))
            show_image(img, title='Clean Sample')
       
    acc_clean = [top1.avg, top5.avg]

    cls_losses = AverageMeter()
    at_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (img, target) in enumerate(test_bad_loader, start=1):
        img = img.cpu()
        target = target.cpu()

        with torch.no_grad():
            output_s = snet(img)
            cls_loss = criterionCls(output_s, target)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        if idx % opt.print_freq == 0:
            print('[Bad Sample] Target: {}, Predicted: {}'.format(target[0].item(), output_s.argmax(dim=1)[0].item()))
            show_image(img, title='Bad Sample')
        if idx % opt.print_freq == 0:
            print('[Bad Samples]')
            if len(target.shape) == 0:  # Check if target is a 0-dimensional tensor
                print('Sample 1: Target: {}, Predicted: {}'.format(target.item(), output_s.argmax(dim=1)[0].item()))
                show_image(img[0], title='Bad Sample 1')
            else:
                for i in range(min(5, len(target))):  # Display up to 5 bad samples
                    print('Sample {}: Target: {}, Predicted: {}'.format(i+1, target[i].item(), output_s.argmax(dim=1)[i].item()))
                    show_image(img[i], title=f'Bad Sample {i+1}')

    acc_bd = [top1.avg, top5.avg, cls_losses.avg]

    print('[clean]Prec@1: {:.2f}'.format(acc_clean[0]))
    print('[bad]Prec@1: {:.2f}'.format(acc_bd[0]))

    # save training progress
    log_root = opt.log_root + '/backdoor_results2.csv'
    test_process.append(
        (epoch, acc_clean[0], acc_bd[0], acc_bd[2]))
    df = pd.DataFrame(test_process, columns=(
    "epoch", "test_clean_acc", "test_bad_acc", "test_bad_cls_loss"))
    df.to_csv(log_root, mode='a', index=False, encoding='utf-8')



    return acc_clean, acc_bd

def train(opt):
    # Load models
    print('----------- Network Initialization --------------')
    student = select_model(dataset=opt.data_name,
                           model_name=opt.s_name,
                           pretrained=False,
                           pretrained_models_path=opt.model,
                           n_classes=opt.num_class).to(opt.device)
    print('finished student model init...')

    nets = {'snet': student}

    # initialize optimizer
    optimizer = torch.optim.SGD(student.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

    # define loss functions
    if opt.cpu:
        criterionCls = nn.CrossEntropyLoss().cpu()
    else:
        criterionCls = nn.CrossEntropyLoss()

    print('----------- DATA Initialization --------------')
    train_loader = get_backdoor_loader(opt)
    test_clean_loader, test_bad_loader = get_test_loader(opt)

    print('----------- Train Initialization --------------')
    for epoch in range(1, opt.epochs):

        _adjust_learning_rate(optimizer, epoch, opt.lr)

        # train every epoch
        criterions = {'criterionCls': criterionCls}
        train_step(opt, train_loader, nets, optimizer, criterions, epoch)

        # evaluate on testing set
        print('testing the models......')
        acc_clean, acc_bad = test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch)

        # remember best precision and save checkpoint
        if opt.save:
            is_best = acc_bad[0] > opt.threshold_bad
            opt.threshold_bad = min(acc_bad[0], opt.threshold_bad)

            best_clean_acc = acc_clean[0]
            best_bad_acc = acc_bad[0]

            s_name = opt.s_name + '-' + opt.attack_method + '.pth'
         

def _adjust_learning_rate(optimizer, epoch, lr):
    if epoch < 20:
        lr = 0.1
    elif epoch < 70:
        lr = 0.01
    else:
        lr = 0.001
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    # Prepare arguments
    opt = get_arguments().parse_args()
    print("opt attributes:", vars(opt))
    train(opt)

import subprocess
if __name__ == '__main__':
    main()
    print("running without defense")
    subprocess.run(["python", "nodefense.py"])
