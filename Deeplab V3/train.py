import torch
import numpy as np
import utils
import torch.optim as optim
import torch.nn as nn
import test

from utils import set_model_mode,get_pseudo_label
import time

def train_transformer(args,transformer, teacher_model,source_train_loader, target_train_loader, optimizer, loss ,epoch, _print, writer):
    start = time.time()

    classifier_criterion = loss[0]
    domain_criterion = loss[1]

    transformer.train()
    len_dataloader = min(len(source_train_loader), len(target_train_loader))

    start_steps = epoch * len_dataloader
    total_steps = args.epochs * len_dataloader

    source_correct = 0
    total_source_loss = 0
    total_target_loss = 0
    source_total = 0
    
    total_domain_loss = 0
    domain_correct =0
    domain_total = 0
    
    for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):
        # 길이 짧은거 만큼 돌아감, 즉 길이가 긴건 매 에폭 마다 suffle 되서 앞에 일부만 들어감
        
        source_image, source_label = source_data
        target_image, _ = target_data

        source_image, source_label = source_image.cuda(), source_label.cuda()
        target_image  = target_image.cuda()
        target_pseudo_label = get_pseudo_label(teacher_model, target_image)

        source_pred, _ = transformer(source_image)
        _, target_pred = transformer(target_image)

        source_loss = classifier_criterion(source_pred, source_label)
        target_loss = classifier_criterion(target_pred, target_pseudo_label)
       
        total_loss = source_loss + target_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_source_loss += source_loss
        total_target_loss += target_loss
        #total_domain_loss += domain_loss


        ### train accuracy ###
        pred = source_pred.argmax(dim=1, keepdim=True)
        source_correct += pred.eq(source_label.long().view_as(pred)).sum().item()
        source_total += source_label.size(0)

    _print("(train) source loss : {:.4f} target loss : {:.4f}".format(
    total_source_loss / len_dataloader, total_target_loss / len_dataloader))
    _print("(train) source accuracy: {:.2f}% train time: {:.2f}s".format(
    (source_correct / source_total) * 100,time.time() - start))

    writer.add_scalar('train/acc/source', (source_correct / source_total) * 100,epoch)
    writer.add_scalar('train/loss/source', total_source_loss / len_dataloader, epoch)
    writer.add_scalar('train/loss/target', total_target_loss / len_dataloader, epoch)
