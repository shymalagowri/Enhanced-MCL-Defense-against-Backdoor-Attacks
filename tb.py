# Import necessary libraries
import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from models.selector import select_model
from utils.util import AverageMeter, accuracy, save_checkpoint
from data_loader import get_test_loader, get_backdoor_loader
from config import get_arguments

def calculate_metrics(y_true, y_pred, target_label):
    # Calculate metrics
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    return precision, recall, f1

def calculate_benign_accuracy(y_true, y_pred):
    # Calculate benign accuracy
    benign_accuracy = accuracy_score(y_true, y_pred)
    return benign_accuracy
def calculate_attack_success_rate(y_true, y_pred, target_label):
    # Calculate attack success rate
    cm = confusion_matrix(y_true, y_pred)

    if target_label < cm.shape[0]:
        attack_success_rate = cm[target_label, target_label] / sum(cm[:, target_label])
    else:
        print(f"Warning: target_label {target_label} is not present in the confusion matrix.")
        attack_success_rate = 0.0

    return attack_success_rate


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




def test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch):
    test_process = []
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets['snet']
    criterionCls = criterions['criterionCls']
    snet.eval()

    y_true_clean, y_pred_clean = [], []
    y_true_bad, y_pred_bad = [], []

    for idx, (img, target) in enumerate(test_clean_loader, start=1):
        img = img.cpu()
        target = target.cpu()

        with torch.no_grad():
            output_s = snet(img)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        y_true_clean.extend(target.tolist())
        y_pred_clean.extend(output_s.argmax(dim=1).cpu().tolist())

    acc_clean = [top1.avg, top5.avg]

    cls_losses = AverageMeter()
    at_losses = AverageMeter()
    top1 = AverageMeter()

    for idx, (img, target) in enumerate(test_bad_loader, start=1):
        img = img.cpu()
        target = target.cpu()

        with torch.no_grad():
            output_s = snet(img)
            cls_loss = criterionCls(output_s, target)

        prec1, _ = accuracy(output_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))

        y_true_bad.extend(target.tolist())
        y_pred_bad.extend(output_s.argmax(dim=1).cpu().tolist())

    acc_bd = [top1.avg, cls_losses.avg]

    print('[clean]Prec@1: {:.2f}'.format(acc_clean[0]))
    print('[bad]Prec@1: {:.2f}'.format(acc_bd[0]))

    # Calculate and print metrics
    precision_clean, recall_clean, f1_clean = calculate_metrics(y_true_clean, y_pred_clean, opt.target_label)
    benign_accuracy = calculate_benign_accuracy(y_true_clean, y_pred_clean)
    attack_success_rate = calculate_attack_success_rate(y_true_bad, y_pred_bad, opt.target_label)

    print('Clean Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}'.format(precision_clean, recall_clean, f1_clean))
    print('Benign Accuracy: {:.4f}'.format(benign_accuracy))
    print('Attack Success Rate: {:.4f}'.format(attack_success_rate))

    # Save training progress
    log_root = opt.log_root + '/backdoor_results.csv'
    test_process.append(
        (epoch, acc_clean[0], acc_bd[0], acc_bd[1], precision_clean, recall_clean, f1_clean, benign_accuracy, attack_success_rate))
    df = pd.DataFrame(test_process, columns=(
    "epoch", "test_clean_acc", "test_bad_acc", "test_bad_cls_loss", "precision_clean", "recall_clean", "f1_clean", "benign_accuracy", "attack_success_rate"))
    df.to_csv(log_root, mode='a', index=False, encoding='utf-8')

    return acc_clean, acc_bd

# ... (remaining code)


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
            save_checkpoint({
                'epoch': epoch,
                'state_dict': student.state_dict(),
                'best_clean_acc': best_clean_acc,
                'best_bad_acc': best_bad_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, os.path.join(opt.checkpoint_root, opt.dataset), s_name)



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
if (__name__ == '__main__'):
    main()
   