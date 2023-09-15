import wandb
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support


def pretrain(encoder,
             projector, 
            dataloader,
            val_dataloader,
            epochs,
            optimizer,
            device,
            loss_fn,
            backup_path = None,
            log = True):
    
    encoder.to(device)
    projector.to(device)
    for epoch in range(epochs):
        epoch_loss = 0
        encoder.train()
        projector.train()
        for i, data in enumerate(dataloader):
            x_1 = data[0].to(device).float()
            x_2 = data[1].to(device).float()
            optimizer.zero_grad()
            out_1 = encoder(x_1)
            out_1 = projector(out_1)

            out_2 = encoder(x_2)
            out_2 = projector(out_2)

            out = torch.cat((out_1.unsqueeze(1), out_2.unsqueeze(1)), dim = 1)
            loss = loss_fn(out)
            if len(loss) > 0:
                loss = loss[0]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss/(i+1)

        val_loss = 0
        encoder.eval()
        projector.eval()
        for i, data in enumerate(val_dataloader):
            x_1 = data[0].to(device).float()
            x_2 = data[1].to(device).float()

            out_1 = encoder(x_1)
            out_1 = projector(out_1)

            out_2 = encoder(x_2)
            out_2 = projector(out_2)

            out = torch.cat((out_1.unsqueeze(1), out_2.unsqueeze(1)), dim = 1)
            loss = loss_fn(out)
            if len(loss) > 0:
                loss = loss[0]

            val_loss += loss.item()


        if log:
            log_dict = {'val_loss': val_loss/(i+1), 'train_loss': train_loss}
            wandb.log(log_dict)

        if backup_path is not None:
            encoder_path = f'{backup_path}/encoder.pt'
            torch.save(encoder.state_dict(), encoder_path)
            projector_path = f'{backup_path}/projector.pt'
            torch.save(projector.state_dict(), projector_path)
