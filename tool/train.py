import os
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from engine.Patchnet_trainer import Trainer
from metrics.losses import PatchLoss
from dataset.FAS_dataset import FASDataset
from dataset.FAS_labels import FASLabels
from utils.utils import read_cfg, get_optimizer, build_network, get_device, get_rank



cfg = read_cfg(cfg_file='config/config.yaml')

# fix the seed for reproducibility
seed = cfg['seed'] + get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
cudnn.benchmark = True

fas_labels = FASLabels(
    root_dir=cfg['dataset']['root'],
    label_dir=cfg['label_dir'],
    train_csv=cfg['dataset']['train_set'],
    val_csv=cfg['dataset']['val_set']
)

# build model and engine
device = get_device(cfg)
model = build_network(cfg, device)
model.to(device)
optimizer = get_optimizer(cfg, model)
lr_scheduler = StepLR(optimizer=optimizer, step_size=90, gamma=0.5)
criterion = PatchLoss(len(fas_labels.label_dict)).to(device=device)
writer = SummaryWriter(cfg['log_dir'])

train_transform = transforms.Compose([
    transforms.Resize(cfg['model']['image_size']),
    transforms.RandomCrop(cfg['dataset']['augmentation']['rand_crop_size']),
    transforms.RandomHorizontalFlip(cfg['dataset']['augmentation']['rand_hori_flip']),
    transforms.RandomRotation(cfg['dataset']['augmentation']['rand_rotation']),
    transforms.ToTensor(),
    transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
])

val_transform = transforms.Compose([
    transforms.Resize(cfg['model']['image_size']),
    transforms.RandomCrop(cfg['dataset']['augmentation']['rand_crop_size']),
    transforms.ToTensor(),
    transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
])

trainset = FASDataset(
    root_dir=cfg['dataset']['root'],
    csv_file=cfg['dataset']['train_set'],
    fas_labels=fas_labels,
    transform=train_transform,
)

valset = FASDataset(
    root_dir=cfg['dataset']['root'],
    csv_file=cfg['dataset']['val_set'],
    fas_labels=fas_labels,
    transform=val_transform,
)

trainloader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=cfg['train']['batch_size'],
    shuffle=True,
    num_workers=4
)

valloader = torch.utils.data.DataLoader(
    dataset=valset,
    batch_size=cfg['val']['batch_size'],
    shuffle=True,
    num_workers=4
)

trainer = Trainer(
    cfg=cfg,
    network=model,
    optimizer=optimizer,
    loss=criterion,
    lr_scheduler=lr_scheduler,
    device=device,
    label_dict=fas_labels.label_dict,
    trainloader=trainloader,
    valloader=valloader,
    writer=writer
)

print("Start training...")
trainer.train()

writer.close()
