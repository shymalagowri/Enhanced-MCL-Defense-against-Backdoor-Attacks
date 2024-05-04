import random
import copy
import torch
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torch.nn import CrossEntropyLoss
from inversion_torch import PixelBackdoor
from models.selector import select_model
from data_loader import get_train_loader, get_test_loader, get_backdoor_loader
from utils.util import AverageMeter, accuracy, save_checkpoint
from utils import Normalizer, Denormalizer
from config import get_arguments

# Global variable for normalization
normalize = None

# Function to initialize normalization function
def get_norm(args):
    global normalize
    normalize = Normalizer(args.dataset)
    print("Normalization function initialized.")

# Function for trigger pattern generation using inversion
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
    # print('@ inv fn, the pattern is '+pattern)
    print('\n')
    print('before attack w trigger fn')

    attack_with_trigger(args, model, train_loader, target_label, pattern)

    print('@after attack w trigger function')

    return pattern

# Function to perform attack with the generated trigger
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

# Function for training step during fine-tuning
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

# Function for fine-tuning
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
        print("negative data : ")
        print(negative_data)

        feature1 = model.get_final_fm(negative_data)
        print("feature  one: ")
        print(feature1)

        feature2 = backup.get_final_fm(data)
        print("feature two: ")
        print(feature2)

        loss = -cos(feature1, feature2.detach()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Fine-tuning completed.")

from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt

# ... (previous code)

# Function for testing the model
def test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch, save_dir):
    test_process = []

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

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        _, predicted = output_s.max(1)
        y_true_clean.extend(target.numpy())
        y_pred_clean.extend(predicted.cpu().numpy())

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

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        _, predicted = output_s.max(1)
        y_true_bad.extend(target.numpy())
        y_pred_bad.extend(predicted.cpu().numpy())

    acc_bd = [top1.avg, top5.avg, cls_losses.avg]

    print('[clean]Prec@1: {:.2f}'.format(acc_clean[0]))
    print('[bad]Prec@1: {:.2f}'.format(acc_bd[0]))

    # Save confusion matrix for clean images
    save_confusion_matrix(y_true_clean, y_pred_clean, "Clean Images", save_dir, "clean_confusion_matrix.png")

    # Save confusion matrix for bad images
    save_confusion_matrix(y_true_bad, y_pred_bad, "Bad Images", save_dir, "bad_confusion_matrix.png")

    print("Testing completed.")

    return acc_clean, acc_bd

def save_confusion_matrix(y_true, y_pred, title, save_dir, file_name):
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix for {}: \n".format(title))
    print(cm)

    # Plot confusion matrix and save as an image
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_true),
                yticklabels=np.unique(y_true))
    plt.title("Confusion Matrix for {}".format(title))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Save the confusion matrix image in the specified directory
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    plt.close()  # Close the plot to avoid displaying in the console

# ... (remaining code)


def fine_defense_adjust_learning_rate(optimizer, epoch, initial_lr, dataset):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if dataset == 'CIFAR10':
        if epoch < 60:
            lr = initial_lr
        elif epoch < 120:
            lr = initial_lr * 0.1
        else:
            lr = initial_lr * 0.01
    else:
        if epoch < 30:
            lr = initial_lr
        elif epoch < 60:
            lr = initial_lr * 0.1
        elif epoch < 90:
            lr = initial_lr * 0.01
        else:
            lr = initial_lr * 0.001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Function to reverse engineer and perform attack without defense
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
    x = get_norm(args=opt)

    print('----------- DATA Initialization --------------')
    train_loader = get_train_loader(opt)
    # print('train loader is '+train_loader)
    print('\n')

    pattern = inversion(opt, model, opt.target_label, train_loader)

    # Fine-tuning without defense
    cl(model, opt, pattern, train_loader)

    # Test the model on clean and bad data
    test_clean_loader, test_bad_loader = get_test_loader(opt)
    nets = {'model': model}
    criterionCls = nn.CrossEntropyLoss()
    criterions = {'criterionCls': criterionCls}
    test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch=0)

if __name__ == '__main__':
    print("in main")
    device = torch.device("cpu:0" if torch.cuda.is_available() else "cpu")
    print('device used %s', device)
    opt = get_arguments().parse_args()
    print('arg used %s', opt)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    print('reverse engineer trigger initiated')
    reverse_engineer(opt)
    print("end of main")
