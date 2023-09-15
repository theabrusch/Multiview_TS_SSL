import torch
import torch.nn.functional as F
import numpy as np

class TS2VecLoss(torch.nn.Module):
    def __init__(self, alpha = 0.5, temporal_unit = 0) -> None:
        super().__init__()
        self.alpha = alpha
        self.temporal_unit = temporal_unit
        self.maxpool = torch.nn.MaxPool1d(2)
    
    def dual_loss(self, z1, z2, d):
        # z1, z2 : B x C x T
        dual_loss = torch.tensor(0.).to(z1.device)
        inst_loss = torch.tensor(0.)

        inst_loss = self.contrastive_loss(z1, z2)
        

        if self.alpha > 0:
            # compute instance loss
            dual_loss += self.alpha*inst_loss
        if self.alpha < 1 and d >= self.temporal_unit:
            # compute temporal loss
            temp_loss = self.contrastive_loss(z1.transpose(0,2), z2.transpose(0,2))
            dual_loss += (1-self.alpha)*temp_loss
        else:
            temp_loss = torch.tensor(0.).to(z1.device)

        return dual_loss, inst_loss.detach().cpu(), temp_loss.detach().cpu()

    def forward(self, z1, z2):
        # z1, z2 : B x C x T
        loss, inst_loss, temp_loss = self.dual_loss(z1, z2, d=0)
        d = 1
        while z1.shape[-1] > 1:
            z1, z2 = self.maxpool(z1), self.maxpool(z2)
            out = self.dual_loss(z1, z2, d)
            loss += out[0]
            inst_loss += out[1]
            temp_loss += out[2]
            d+=1

        return loss/d, inst_loss/d, temp_loss/d

    def contrastive_loss(self, z1, z2):
        '''
        The contrastive loss is computed across the first dimension.
        '''
        # z1, z2 : B x C x T
        B = z1.shape[0]
        z = torch.cat([z1,z2], dim = 0) # 2B x C x T
        z = z.permute((2,0,1)) # T x 2B x C
        sim = torch.matmul(z, z.transpose(1,2)) # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:] 
        logits = -F.log_softmax(logits, dim=-1) 
        
        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2 
        return loss
    
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # z1, z2 : B x C X L or B X L
        B = z1.shape[0]
        if len(z1.shape) > 2:
            z1 = z1.reshape(B, -1)
            z2 = z2.reshape(B, -1)
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        z = torch.cat([z1,z2], dim = 0) # 2B x C x T
        sim = torch.matmul(z, z.transpose(0,1)) # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :-1]    # T x 2B x (2B-1)
        logits += torch.triu(sim, diagonal=1)[:, 1:] 
        logits = -F.log_softmax(logits/self.temperature, dim=-1) 
        
        i = torch.arange(B, device=z1.device)
        loss = (logits[i, B + i - 1].mean() + logits[B + i, i].mean()) / 2 
        return loss


def compute_weights(targets):
    _, count = np.unique(targets, return_counts=True)
    weights = 1 / count
    weights = weights / weights.sum()
    return torch.tensor(weights).float()

class COCOAloss(torch.nn.Module):
    def __init__(self, temperature, scale_loss = 1/32, lambda_ = 3.9e-3):
        super(COCOAloss, self).__init__()
        self.temperature = temperature
        self.scale_loss = scale_loss
        self.lambda_ = lambda_
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z):
        z = z.reshape(z.shape[0], z.shape[1], -1)
        z = z.transpose(1, 0)
        batch_size, view_size = z.shape[1], z.shape[0]

        z = F.normalize(z, dim = -1)
        pos_error = []
        for i in range(batch_size):
            sim = torch.matmul(z[:, i, :], z[:, i, :].T)
            sim = torch.ones([view_size, view_size]).to(z.device)-sim
            sim = torch.exp(sim/self.temperature)
            pos_error.append(sim.mean())
        
        neg_error = 0
        for i in range(view_size):
            sim = torch.matmul(z[i], z[i].T)
            sim = torch.exp(sim / self.temperature)
            tri_mask = np.ones(batch_size ** 2, dtype=np.bool).reshape(batch_size, batch_size)
            tri_mask[np.diag_indices(batch_size)] = False
            tri_mask = torch.tensor(tri_mask).to(z.device)
            off_diag_sim = torch.reshape(torch.masked_select(sim, tri_mask), [batch_size, batch_size - 1])
            neg_error += off_diag_sim.mean(-1)

        pos_error = torch.stack(pos_error)
        error = torch.sum(pos_error)*self.scale_loss + self.lambda_ * torch.sum(neg_error)
        return error

class CMCloss(torch.nn.Module):
    def __init__(self, temperature, criterion = 'contrastive'):
        super(CMCloss, self).__init__()
        if criterion == 'contrastive':
            self.criterion = ContrastiveLoss(temperature)
        elif criterion == 'TS2Vec':
            self.criterion = TS2VecLoss(temperature)
    def forward(self, z):
        # make the number of views as the first dimension
        z = z.transpose(1, 0)
        batch_size, dim_size = z.shape[1], z.shape[0]
        loss = torch.tensor(0.).to(z.device)
        time_loss = torch.tensor(0.).to(z.device)
        inst_loss = torch.tensor(0.).to(z.device)
        d = 0
        for i in range(dim_size):
            for j in range(i+1, dim_size):
                l = self.criterion(z[i], z[j])
                if isinstance(l, tuple):
                    inst_loss += l[1]
                    time_loss += l[2]
                    loss += l[0]
                else:
                    loss += l
                d += 1
        return loss/d, time_loss/d, inst_loss/d




def get_loss(loss_function, device, temperature = 0.5):
    if loss_function == 'time_loss':
        loss_fn = CMCloss(temperature = temperature, criterion='TS2Vec').to(device)
    elif loss_function == 'contrastive':
        loss_fn = CMCloss(temperature = temperature, criterion='contrastive').to(device)
    elif loss_function == 'COCOA':
        loss_fn = COCOAloss(temperature = temperature).to(device)
    return loss_fn