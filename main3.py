import random
import copy

from data_loader import get_backdoor_loader
from data_loader import get_train_loader, get_test_loader
from inversion_torch import PixelBackdoor
from utils.util import *
from models.selector import *
from config import get_arguments
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
import tqdm
import matplotlib.pyplot as plt
from utils import Normalizer, Denormalizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

normalize = None

def get_norm(args):
    global normalize
    normalize = Normalizer(args.dataset)
    print("Normalization function initialized.")

def inversion(args, model, target_label, train_loader):
    print('In inversion function')
    global normalize

    if args.dataset == 'imagenet':
        shape = (3, 224, 224)
    elif args.dataset == 'tinyImagenet':
        shape = (3, 64, 64)
    else:
        shape = (3, 32, 32)

    print("Processing label: {}".format(target_label))
    backdoor = PixelBackdoor(model,
                             shape=shape,
                             batch_size=args.batch_size,
                             normalize=normalize,
                             steps=100,
                             augment=False)

    pattern = backdoor.generate(train_loader, target_label, attack_size=args.attack_size)
    print('\n')
    print('before attack w trigger fn')

    attack_with_trigger(args, model, train_loader, target_label, pattern)

    print('@after attack w trigger function')

    return pattern

import cv2
import os.path as osp

def attack_with_trigger(args, model, train_loader, target_label, pattern):
    global normalize
    denormalize = Denormalizer(args.dataset)
    correct = 0
    total = 0
    pattern = pattern.to(device)
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm.tqdm(train_loader):
            images = images.to(device)
            trojan_images = torch.clamp(images + pattern, 0, 1)
            trojan_images = normalize(trojan_images)
            y_pred = model(trojan_images)
            y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(device)

            _, y_pred = y_pred.max(1)
            correct += y_pred.eq(y_target).sum().item()
            total += images.size(0)

        print("Accuracy on trojaned images:", correct / total)

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(y_true, y_pred, target_label):
    # Set zero_division=1 to suppress the warning and set the metric to 0 for divisions by zero
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    return precision, recall, f1





def train_step(opt, train_loader, nets, optimizer, criterions, pattern, epoch):
    global normalize

    model = nets['model']
    backup = nets['victimized_model']

    criterionCls = criterions['criterionCls']
    cos = torch.nn.CosineSimilarity(dim=-1)
    mse_loss = torch.nn.MSELoss()

    model.train()
    backup.eval()

    for idx, (data, label) in enumerate(train_loader, start=1):
        data, label = data.clone().cpu(), label.clone().cpu()

        negative_data = copy.deepcopy(data)
        negative_data = torch.clamp(negative_data + pattern, 0, 1)

        data = normalize(data)
        negative_data = normalize(negative_data)

        feature1 = model.get_final_fm(negative_data)
        feature2 = backup.get_final_fm(data)

        posi = cos(feature1, feature2.detach())
        logits = posi.reshape(-1, 1)

        feature3 = backup.get_final_fm(negative_data)
        nega = cos(feature1, feature3.detach())
        logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

        logits /= opt.temperature
        labels = torch.zeros(data.size(0)).cpu().long()
        cmi_loss = criterionCls(logits, labels)

        loss = cmi_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Training step completed.")

def fine_tuning(opt, train_loader, nets, optimizer, criterions, pattern, epoch):
    global normalize

    model = nets['model']
    backup = nets['victimized_model']

    criterionCls = criterions['criterionCls']

    cos = nn.CosineSimilarity(dim=1).cpu()

    model.train()
    backup.eval()

    for idx, (data, label) in enumerate(train_loader, start=1):
        data, label = data.clone().cpu(), label.clone().cpu()

        negative_data = copy.deepcopy(data)
        negative_data = torch.clamp(negative_data + pattern, 0, 1)

        data = normalize(data)
        negative_data = normalize(negative_data)

        feature1 = model.get_final_fm(negative_data)

        feature2 = backup.get_final_fm(data)

        loss = -cos(feature1, feature2.detach()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Fine-tuning completed.")

def test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch):
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets['model']

    criterionCls = criterions['criterionCls']

    snet.eval()

    y_true_clean, y_pred_clean = [], []
    y_true_bad, y_pred_bad = [], []

    for idx, (img, target) in enumerate(test_clean_loader, start=1):
        img = img.cpu()
        target = target.cpu()

        with torch.no_grad():
            output_s = snet(img)

        _, predicted = torch.max(output_s, 1)

        y_true_clean.extend(target.numpy())
        y_pred_clean.extend(predicted.numpy())

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_clean = [top1.avg, top5.avg]

    cls_losses = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (img, target) in enumerate(test_bad_loader, start=1):
        img = img.cpu()
        target = target.cpu()

        if opt.attack_method == 'wanet':
            grid_temps = (opt.identity_grid + 0.5 * opt.noise_grid / opt.input_height) * 1
            grid_temps = torch.clamp(grid_temps, -1, 1)

            img = F.grid_sample(img, grid_temps.repeat(img.shape[0], 1, 1, 1), align_corners=True)

        with torch.no_grad():
            output_s = snet(img)

            cls_loss = criterionCls(output_s, target)

        _, predicted = torch.max(output_s, 1)

        y_true_bad.extend(target.numpy())
        y_pred_bad.extend(predicted.numpy())

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_bd = [top1.avg, top5.avg, cls_losses.avg]

    print('[clean]Prec@1: {:.2f}'.format(acc_clean[0]))
    print('[bad]Prec@1: {:.2f}'.format(acc_bd[0]))

    # Calculate metrics for clean images
    precision_clean, recall_clean, f1_clean = calculate_metrics(y_true_clean, y_pred_clean, opt.target_label)
    benign_accuracy = accuracy_score(y_true_clean, y_pred_clean)

    # Calculate metrics for bad images
    precision_bad, recall_bad, f1_bad = calculate_metrics(y_true_bad, y_pred_bad, opt.target_label)
    attack_success_rate = accuracy_score(y_true_bad, y_pred_bad)

    print('Metrics for Clean Images:')
    print('Precision: {:.2f}, Recall: {:.2f}, F1 Score: {:.2f}, Benign Accuracy: {:.2f}'.format(
        precision_clean, recall_clean, f1_clean, benign_accuracy))

    print('Metrics for Bad Images:')
    print('Precision: {:.2f}, Recall: {:.2f}, F1 Score: {:.2f}, Attack Success Rate: {:.2f}'.format(
        precision_bad, recall_bad, f1_bad, attack_success_rate))

    print("Testing completed.")

    return acc_clean, acc_bd

def cl(model, opt, pattern, train_loader):
    test_clean_loader, test_bad_loader = get_test_loader(opt)

    nets = {'model': model, 'victimized_model':copy.deepcopy(model)}

    # initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.01,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

    # define loss functions
    if opt.cpu:
        criterionCls = nn.CrossEntropyLoss().cpu()
    else:
        criterionCls = nn.CrossEntropyLoss()

    print('----------- Train Initialization --------------')
    for epoch in range(0, opt.epochs):

        # train every epoch
        criterions = {'criterionCls': criterionCls}

        if epoch == 0:
            # before training test firstly
            test(opt, test_clean_loader, test_bad_loader, nets,
                 criterions, epoch)

        print("===Epoch: {}/{}===".format(epoch + 1, opt.epochs))

        fine_defense_adjust_learning_rate(optimizer, epoch, opt.lr, opt.dataset)

        train_step(opt, train_loader, nets, optimizer, criterions, pattern, epoch)

        # evaluate on testing set
        print('testing the models......')
        test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch + 1)

    print("Training completed.")

def reverse_engineer(opt):
    print('in rev engineering fn')
    model = select_model(dataset=opt.data_name,
                           model_name=opt.s_name,
                           pretrained=True,
                           pretrained_models_path=opt.model,
                           n_classes=opt.num_class).to(opt.device)
    print('@in if else in rev eng')
    if opt.attack_method == 'wanet':
        if opt.dataset == 'tinyImagenet':
            opt.input_height = 64
            identity_grid = torch.load('./trigger/ResNet18_tinyImagenet_WaNet_identity_grid.pth').to(opt.device)
            noise_grid = torch.load('./trigger/ResNet18_tinyImagenet_WaNet_noise_grid.pth').to(opt.device)
        else:
            identity_grid = torch.load('./trigger/WRN-16-1_CIFAR-10_WaNet_identity_grid.pth').to(opt.device)
            noise_grid = torch.load('./trigger/WRN-16-1_CIFAR-10_WaNet_noise_grid.pth').to(opt.device)
        opt.identity_grid = identity_grid
        opt.noise_grid = noise_grid
    print("Getting normalization function based on the dataset...")
    get_norm(args=opt)
    x=get_norm(args=opt)


    print('----------- DATA Initialization --------------')
    train_loader = get_train_loader(opt)
    print('\n')

    pattern = inversion(opt, model, opt.target_label, train_loader)
    cl(model, opt, pattern, train_loader)

if __name__ == '__main__':
    print("in main")
    device = torch.device("cpu:0" if torch.cuda.is_available() else "cpu")
    print('device used %s',device)
    opt = get_arguments().parse_args()
    print('arg used %s',opt)
    random.seed(opt.seed)  # torch transforms use this seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    print('reverse engineer trigger initiated')
    reverse_engineer(opt)
    print("end of main")
