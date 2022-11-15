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

from focus_face.metrics import calculate_metrics
from focus_face.model import FocusFace
from focus_face.dataset import TrainDataset, EvalDataset
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
train_dataset = TrainDataset(opt.train, mask_args)
eval_dataset = EvalDataset(opt.eval, mask_args)
train_loader = DataLoader(train_dataset, batch_size=opt.bs, num_workers=workers, shuffle=True, pin_memory=True)
eval_loader = DataLoader(eval_dataset, batch_size=opt.bs, num_workers=workers, shuffle=False, pin_memory=True)
identities = len(train_dataset.classes)


print("Creating model")
model = FocusFace(identities=identities).to(device)
optimizer = optim.Adam(model.parameters(),lr=opt.lr)
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
    optimizer.load_state_dict(state["optimizer"])
    if opt.amp:
        scaler.load_state_dict(state["scaler"])
else:
    starting_epoch = 0


print("Start training")
best_score = 100
training_losses = []
for epoch in range(starting_epoch, opt.n_epochs):
    # Train
    epoch_loss = 0
    model.train()

    time_start = time.time()
    for sample in tqdm(train_loader, disable=not opt.verbose):
        img_raw = sample['masked image'].to(device)
        img_msk = sample['raw image'].to(device)
        id = sample['identity'].to(device)
        mask = sample['mask'].to(device)

        id_raw_pred, emb_raw, _, mask_raw_pred = model(img_raw, label=id)
        id_msk_pred, emb_msk, _, mask_msk_pred = model(img_msk, label=id)
        loss = CE(id_raw_pred, id) +  0.1 * CE(mask_raw_pred, torch.zeros_like(mask_raw_pred))
        loss += CE(id_msk_pred, id) +  0.1 * CE(mask_msk_pred, mask)
        loss /= 2
        loss += MSE(emb_raw, emb_msk) * 0.3

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
    time_end = time.time()

    epoch_loss /= len(train_loader)
    training_losses.append(epoch_loss)
    if epoch % opt.log_interval == 0:        
        print("Training time: {:.2f}s".format(time_end - time_start))
        print('[{}/{}], training loss: {:.4f}'.format(epoch+1, opt.n_epochs, epoch_loss))

    # Evaluation
    print("Start evaluation")
    model.eval()
    with torch.no_grad():
        positives = []
        negatives = []
        for sample in tqdm(eval_loader, disable=not opt.verbose):
            img_anchor = sample["anchor"].to(device)
            img_other = sample["other"].to(device)
            label_same = sample["same"]

            _, emb_anchor, _, _ = model(img_anchor, inference=True) 
            _, emb_other, _, _ = model(img_other, inference=True) 
            emb_anchor = F.normalize(emb_anchor).unsqueeze(1)
            emb_other = F.normalize(emb_other).unsqueeze(1)
            dist = 1 - torch.cdist(emb_anchor, emb_other) / 2
            for i in range(label_same.shape[0]):
                if label_same[i] == 1:
                    positives.append(dist[i].squeeze().cpu().numpy())
                else:
                    negatives.append(dist[i].squeeze().cpu().numpy())

        metrics = get_eer_stats(positives, negatives)
    
    fmr100 = metrics.fmr100
    if epoch % opt.log_interval == 0:
        print("FMR 100:", fmr100)
        print("AUC:", metrics.auc)
        print()

    # Save checkpoint
    if fmr100 < best_score:
        best_score = fmr100
        checkpoint = {"model": model.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict(),
              "epoch": epoch
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, "model_best.pth"))
