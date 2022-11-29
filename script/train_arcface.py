import sys
sys.path.insert(0, '../')

from tqdm import tqdm
import multiprocessing
import os
import time
from argparse import ArgumentParser, Namespace

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pyeer.eer_info import get_eer_stats
from torchvision.models import resnet18, resnet34, resnet50, resnet101

from ArcFace.model import *
from ArcFace.dataset import TrainDataset, EvalDataset
import utils


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = ArgumentParser()
parser.add_argument('--name', type=str, default='test', help='experiment name')
parser.add_argument('-E', '--n_epochs', type=int,
                    default=500, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('-t', '--train', type=str,
                    default="/WebFace1k", help='training dataset')
parser.add_argument('-m', '--mask', type=str,
                    default="/WebFace1k_msk", help='masked training dataset')                   
parser.add_argument('-e', '--eval', type=str, default=None,
                    help='evaluation dataset')
parser.add_argument('-B', "--bs", type=int, default=480, help="Batch size")
parser.add_argument('--log_interval', type=int, default=10,
                    help='logging interval of saving results')
parser.add_argument('--resume', default=False, action='store_true',
                    help='toggle resuming')
parser.add_argument('--amp', default=False, action='store_true',
                    help="Toggle auto mixed precision")
parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Display tqdm bar")
parser.add_argument('--backbone', default='resnet18', help='backbone network type')
parser.add_argument('--metric', default='arc_margin', help='embedding head of model')
parser.add_argument('--easy_margin', default=False, action="store_true", help='margin type')
opt = parser.parse_args()


checkpoint_dir = os.path.join('../res',opt.name)
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

mask_args = Namespace()
mask_args.mask_type = "random"
mask_args.detector, mask_args.predictor = utils.msk_init()
mask_args.verbose = False
mask_args.code, mask_args.pattern, mask_args.color = "", "", ""


workers = min(8, multiprocessing.cpu_count())

print("Loading dataset")
train_dataset = TrainDataset(opt.train, opt.mask, mask_args)
eval_dataset = EvalDataset(opt.eval, mask_args)
train_loader = DataLoader(train_dataset, batch_size=opt.bs, num_workers=workers, shuffle=True, pin_memory=True)
eval_loader = DataLoader(eval_dataset, batch_size=opt.bs, num_workers=workers, shuffle=False, pin_memory=True)
identities = len(train_dataset.classes)


print("Creating model")

if opt.backbone == 'resnet18':
    model = resnet18(weights="ResNet18_Weights.DEFAULT")
elif opt.backbone == 'resnet34':
    model = resnet18(weights="ResNet34_Weights.DEFAULT")
elif opt.backbone == 'resnet50':
    model = resnet18(weights="ResNet50_Weights.DEFAULT")
elif opt.backbone == 'resnet101':
    model = resnet18(weights="ResNet101_Weights.DEFAULT")
else:
    assert 0, "Model not supported"
modules = list(model.children())[:-1]
model = nn.Sequential(*modules).to(device)

if opt.metric == 'add_margin':
    metric_fc = AddMarginProduct(512, identities, s=30, m=0.35).to(device)
elif opt.metric == 'arc_margin':
    metric_fc = ArcMarginProduct(512, identities, s=30, m=0.5, easy_margin=opt.easy_margin).to(device)
elif opt.metric == 'sphere':
    metric_fc = SphereProduct(512, identities, m=4).to(device)
else:
    metric_fc = nn.Linear(512, identities).to(device)
    

optimizer = optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=opt.lr)
MSE = nn.MSELoss()
CE = nn.CrossEntropyLoss()  
if opt.amp:
    scaler = torch.cuda.amp.GradScaler()

if opt.resume:
    resume_filename = os.path.join(checkpoint_dir, "model_best.pth")
    print("Resuming From ", resume_filename)

    state = torch.load(resume_filename)
    starting_epoch = state["epoch"]
    model.load_state_dict(state["model"])
    metric_fc.load_state_dict(state['fc'])
    optimizer.load_state_dict(state["optimizer"])
    if opt.amp:
        scaler.load_state_dict(state["scaler"])
else:
    starting_epoch = 0


print("Start training")
best_score = 0.5
training_losses = []
for epoch in range(starting_epoch, opt.n_epochs):
    # Train
    epoch_loss = 0
    raw_correct = 0
    msk_correct = 0
    model.train()

    time_start = time.time()
    for sample in tqdm(train_loader, disable=not opt.verbose):
        img_raw = sample['raw image'].to(device)
        img_msk = sample['masked image'].to(device)
        id = sample['identity'].to(device)
        mask = sample['mask'].to(device)
         
        # print(img_msk.shape)
        # print(img_raw.shape)
        # print(id)
        # print(mask)
        
        feature_raw = model(img_raw).squeeze()
        feature_msk = model(img_msk).squeeze()
        
        output_raw = metric_fc(feature_raw, id)
        output_msk = metric_fc(feature_msk, id)

        loss = CE(output_raw, id) + CE(output_msk, id)
        
        if opt.amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

        raw_class = output_raw.argmax(dim=1)
        msk_class = output_msk.argmax(dim=1)
        raw_correct += torch.sum(raw_class == id)
        msk_correct += torch.sum(msk_class == id)

    
    time_end = time.time()

    epoch_loss /= len(train_loader)
    training_losses.append(epoch_loss)
    if epoch % opt.log_interval == 0:        
        print("Training time: {:.2f}s".format(time_end - time_start))
        print('[{}/{}], training loss: {:.4f}'.format(epoch+1, opt.n_epochs, epoch_loss))
        print("Trianing raw image accuracy: {:.4f}".format(raw_correct / len(train_dataset)))
        print("Trianing mask image accuracy: {:.4f}".format(msk_correct / len(train_dataset)))


    # Evaluation
    print("Start evaluation")
    
    model.eval()
    with torch.no_grad():
        positives = []
        negatives = []
        accuracy = []
        # acc_metric = Accuracy()
        for sample in tqdm(eval_loader, disable=not opt.verbose):
            img_anchor = sample["anchor"].to(device)
            img_other = sample["other"].to(device)
            label_same = sample["same"]

            feature_anchor = model(img_anchor).squeeze()
            feature_other = model(img_other).squeeze()
            
            # feature_anchor = F.normalize(feature_anchor)
            # feature_other = F.normalize(feature_other)
            
            # dist = 1 - torch.cdist(feature_anchor, feature_other) / 2
            dist = F.cosine_similarity(feature_anchor, feature_other)
            
            for i in range(label_same.shape[0]):
                if label_same[i] == 1:
                    positives.append(dist[i].item())
                else:
                    negatives.append(dist[i].item())
        
        metrics = get_eer_stats(positives, negatives)
    
    auc = metrics.auc
    if epoch % opt.log_interval == 0:
        print("FMR 100:", metrics.fmr100)
        print("AUC:", auc)
        # print("Other metrics:metrics.fmr0, metrics.fmr100, metrics.fmr1000, metrics.gmean, metrics.imean, metrics.auc")
        # print(metrics.fmr0, metrics.fmr100, metrics.fmr1000, metrics.gmean, metrics.imean, metrics.auc)
        print()
        
    # Save checkpoint
    if auc > best_score:
        best_score = auc
        checkpoint = {"model": model.state_dict(),
            "fc": metric_fc.state_dict(),
            "optimizer": optimizer.state_dict(),
            # "scaler": scaler.state_dict(),
            "epoch": epoch
        }
        if opt.amp:
            checkpoint["scaler"] = scaler.state_dict()
        torch.save(checkpoint, os.path.join(checkpoint_dir, "model_best.pth"))
        print('best model saved.')
