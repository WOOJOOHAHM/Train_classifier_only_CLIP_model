import os
import pandas as pd
import math
import lightning.pytorch as pl
from typing import Any, Callable, List, Union
from tqdm import tqdm

import torch
from torch import Tensor, nn, optim
from torchmetrics.functional import accuracy, f1_score
from torchvision.transforms import transforms  as T
import torch.nn.functional as F
from torchvision.transforms._transforms_video import ToTensorVideo
from pytorchvideo.transforms import Normalize, Permute, RandAugment
from dataset import VideoDataset
def load_data(dataset_name: str, 
            path: str, 
            batch_size: int = 32,
            num_workers: int = 16,
            num_frames: int = 8,
            video_size: int = 224
                ):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = T.Compose(
        [
            ToTensorVideo(),  # C, T, H, W
            Permute(dims=[1, 0, 2, 3]),  # T, C, H, W
            RandAugment(magnitude=10, num_layers=2),
            Permute(dims=[1, 0, 2, 3]),  # C, T, H, W
            T.Resize(size=(video_size, video_size)),
            Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    test_transform = T.Compose(
        [
            ToTensorVideo(),
            T.Resize(size=(video_size, video_size)),
            Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    if dataset_name == 'hmdb_sample':
        dataframe = pd.read_csv(f'{path}/hmdb_sample.csv')
        num_classes = 51
    elif dataset_name == 'hmdb':
        dataframe = pd.read_csv(f'{path}/hmdb.csv')
        num_classes = 51
    elif dataset_name == 'ucf_sample':
        dataframe = pd.read_csv(f'{path}/ucf_sample.csv')
        num_classes = 101
    elif dataset_name == 'ucf':
        dataframe = pd.read_csv(f'{path}/ucf101.csv')
        num_classes = 101
    elif dataset_name == 'k400_sample':
        dataframe = pd.read_csv(f'{path}/k400_sample.csv')
        num_classes = 400
    elif dataset_name == 'k400':
        dataframe = pd.read_csv(f'{path}/k400.csv')
        num_classes = 400
    elif dataset_name == 'ssv2_sample':
        dataframe = pd.read_csv(f'{path}/ssv2_sample.csv')
        num_classes = 174
    elif dataset_name == 'ssv2':
        dataframe = pd.read_csv(f'{path}/ssv2.csv')
        num_classes = 174

    dataset_train = VideoDataset(
        dataframe,
        num_frames,
        'train',
        train_transform
    )
    dataset_val = VideoDataset(
        dataframe,
        num_frames,
        'valid',
        test_transform
    )

    dataset_test = VideoDataset(
        dataframe,
        num_frames,
        'test',
        test_transform
    )

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        # sampler=torch.utils.data.DistributedSampler(dataset_train),
        num_workers=num_workers,
        pin_memory=True,
        )
    dataloader_val = torch.utils.data.DataLoader(
        # torch.utils.data.Subset(dataset_val, range(dist.get_rank(), len(dataset_val), dist.get_world_size())),
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        )
    
    dataloader_test = torch.utils.data.DataLoader(
        # torch.utils.data.Subset(dataset_val, range(dist.get_rank(), len(dataset_val), dist.get_world_size())),
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        )
    
    return dataloader_train, dataloader_val, dataloader_test, num_classes

def evaluate(model, dataloader_val, test=None, dataset_name=None):
    model.eval()
    losses = []
    acc1s = []
    acc5s = []
    for data, labels in tqdm(dataloader_val, disable=True):
        data, labels = data.cuda(), labels.cuda()
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                logits, _, _, _, _, _ = model(data)
                loss = F.cross_entropy(logits, labels)
                acc1 = (logits.topk(1, dim=1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
                acc5 = (logits.topk(5, dim=1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
                losses.append(loss.item())
                acc1s.append(acc1)
                acc5s.append(acc5)
    print('             * Evaluate     Losses : {:.5f}    Acc1 : {:.3f}    Acc5 : {:.3f}'.format(sum(losses)/len(losses), sum(acc1s)/len(acc1s), sum(acc5s)/len(acc5s)))
    if test:
        result = pd.read_csv(f'/hahmwj/Video_KD/trained_model/{dataset_name}/result.csv')
        result['test_loss'] = sum(losses)/len(losses)
        result['test_acc1'] = sum(acc1s)/len(acc1s)
        result['test_acc5'] = sum(acc5s)/len(acc5s)
        result.to_csv(f'/hahmwj/Video_KD/trained_model/{dataset_name}/result.csv', index=False)
    return sum(losses)/len(losses), sum(acc1s)/len(acc1s), sum(acc5s)/len(acc5s)

def train(model, dataloader_train, dataloader_val, epochs, optimizer, dataset_name):
    def lr_func(step):
        epoch = step / len(dataloader_train)
        if epoch < 3:
            return epoch / 3
        else:
            return 0.5 + 0.5 * math.cos((epoch - 3) / (epochs - 3) * math.pi)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)

    model.cuda()
    train_losses = []
    train_acc1s = []
    train_acc5s = []
    lrs = []

    eval_losses = []
    eval_acc1s = []
    eval_acc5s = []
    for epoch in range(epochs):
        model.train()
        losses = []
        acc1s = []
        acc5s = []
        for data, labels in tqdm(dataloader_train):
            data, labels = data.cuda(), labels.cuda()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits, _, _, _, _, _ = model(data)
                loss = F.cross_entropy(logits, labels)
                acc1 = (logits.topk(1, dim=1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
                acc5 = (logits.topk(5, dim=1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
                losses.append(loss.item())
                acc1s.append(acc1)
                acc5s.append(acc5)
            loss.backward()
            optimizer.step()
            lr_sched.step()
        lr = optimizer.param_groups[0]['lr']

        print('Epochs : {}/{}   Train     Losses : {:.5f}    Acc1 : {:.3f}    Acc5 : {:.3f}     lr : {:5f}'.format(epochs, epoch, sum(losses)/len(losses), sum(acc1s)/len(acc1s), sum(acc5s)/len(acc5s), lr))
        eval_loss, eval_acc1, eval_acc5 = evaluate(model, dataloader_val)

        if not os.path.exists(f'/hahmwj/Video_KD/trained_model/{dataset_name}/'):
            os.makedirs(f'/hahmwj/Video_KD/trained_model/{dataset_name}/')

        torch.save(model.state_dict(), f'/hahmwj/Video_KD/trained_model/{dataset_name}/{epoch}.pt')
        
        train_losses.append(sum(losses)/len(losses))
        train_acc1s.append(sum(acc1s)/len(acc1s))
        train_acc5s.append(sum(acc5s)/len(acc5s))

        eval_losses.append(eval_loss)
        eval_acc1s.append(eval_acc1)
        eval_acc5s.append(eval_acc5)

        lrs.append(lr)
    result = pd.DataFrame({"train_losses" : train_losses, 
                           "train_acc1s" : train_acc1s, 
                           "train_acc5s" : train_acc5s, 
                           "eval_losses" : eval_losses, 
                           "eval_acc1s" : eval_acc1s, 
                           "eval_acc5s" : eval_acc5s,
                           "Learning rate" : lrs})
    result.to_csv(f'/hahmwj/Video_KD/trained_model/{dataset_name}/result.csv', index=False)
    return model

def get_difference(video, att1, att2, att3, att4, att5):
    frames = video.shape(2)
    video = video.flatten(0, 1).permute(1, 0, 2, 3).flatten(2, 3).flatten(1, 2)
    att1 = att1.permute(1, 0, 2).flatten(1, 2)
    att2 = att2.permute(1, 0, 2).flatten(1, 2)
    att3 = att3.permute(1, 0, 2).flatten(1, 2)
    att4 = att4.permute(1, 0, 2).flatten(1, 2)
    att5 = att5.permute(1, 0, 2).flatten(1, 2)

    video_differs = []
    att1_differs = []
    att2_differs = []
    att3_differs = []
    att4_differs = []
    att5_differs = []
    for i in range(frames//2):
        video_difference = sum(abs(video[i] - video[i+1])/max(abs(video[i] - video[i+1]))/video.shape[1])
        att1_difference = sum(abs(att1[i] - att1[i+1])/max(abs(att1[i] - att1[i+1]))/att1.shape[1])
        att2_difference = sum(abs(att2[i] - att2[i+1])/max(abs(att2[i] - att2[i+1]))/att2.shape[1])
        att3_difference = sum(abs(att3[i] - att3[i+1])/max(abs(att3[i] - att3[i+1]))/att3.shape[1])
        att4_difference = sum(abs(att4[i] - att4[i+1])/max(abs(att4[i] - att4[i+1]))/att4.shape[1])
        att5_difference = sum(abs(att5[i] - att5[i+1])/max(abs(att5[i] - att5[i+1]))/att5.shape[1])

        video_differs.append(video_difference)
        att1_differs.append(att1_difference)
        att2_differs.append(att2_difference)
        att3_differs.append(att3_difference)
        att4_differs.append(att4_difference)
        att5_differs.append(att5_difference)
    
    return video_differs, att1_differs, att2_differs, att3_differs, att4_differs, att5_differs