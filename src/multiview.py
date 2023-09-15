import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models import wave2vecblock
import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from src.losses import COCOAloss, CMCloss
import wandb

class TimeClassifier(nn.Module):
    def __init__(self, in_features, num_classes, pool = 'adapt_avg', n_layers =1, orig_channels = 9, time_length = 33):
        super().__init__()
        self.pool = pool
        self.flatten = nn.Flatten()
        self.adpat_avg = nn.AdaptiveAvgPool1d(4)
        
        self.channelreduction = nn.Linear(in_features=orig_channels, out_features=1)
        if self.pool == 'adapt_avg':
            in_features = 4*in_features
        elif self.pool == 'flatten':
            in_features = in_features*time_length
        
        if n_layers == 1:
            self.classifier = nn.Linear(in_features=in_features, out_features=num_classes)
        elif n_layers == 2:
            self.classifier = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=in_features),
                nn.ReLU(),
                nn.Linear(in_features=in_features, out_features=num_classes)
            )

    def forward(self, latents):
        if len(latents.shape) > 3:
            latents = latents.permute(0,2,3,1)
            latents = self.channelreduction(latents).squeeze(-1)

        ts_length = latents.shape[2]
        if self.pool == 'max':
            latents = F.max_pool1d(latents, ts_length).squeeze(-1)
        elif self.pool == 'last':
            latents = latents[:,:,-1]
        elif self.pool == 'avg':
            latents = F.avg_pool1d(latents, ts_length).squeeze(-1)
        elif self.pool == 'adapt_avg':
            latents = self.flatten(self.adpat_avg(latents))
        else:
            latents = self.flatten(latents)
        
        return self.classifier(latents)

class TimeProjector(nn.Module):
    def __init__(self, in_features, output_dim, n_layers =1):
        super().__init__()
        if n_layers == 1:
            self.projector = nn.Linear(in_features=in_features, out_features=output_dim)
        elif n_layers == 2:
            self.projector = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=in_features),
                nn.ReLU(),
                nn.Linear(in_features=in_features, out_features=output_dim)
            )
    
    def forward(self, latents):
        latents = latents.permute(0,2,1)
        return self.projector(latents).permute(0,2,1)


class Classifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.classifier = nn.Linear(in_features=in_features, out_features=num_classes)
    def forward(self, x):
        return self.classifier(x)


def conv1D_out_shape(input_shape, kernel, stride, padding):
    kernel, stride, padding = np.array(kernel), np.array(stride), np.array(padding)
    shape = input_shape
    for kern, stri, padd in zip(kernel, stride, padding):
        shape = int((shape + 2*padd - kern)/stri + 1)
    return shape

class Wave2Vec(nn.Module):
    def __init__(self, 
                 channels, 
                 input_shape, 
                 out_dim = 64, 
                 hidden_channels = 512, 
                 nlayers = 6, do = 0.1, 
                 norm = 'group',
                 readout_layer = False,):
        super().__init__()
        self.channels = channels
        width = [3] + [2]*(nlayers-1)
        in_channels = [channels] + [hidden_channels]*(nlayers-1)
        out_channels = [hidden_channels]*(nlayers - 1) + [out_dim]
        self.convblocks = nn.Sequential(*[wave2vecblock(channels_in= in_channels[i], channels_out = out_channels[i], kernel = width[i], stride = width[i], norm = norm, dropout = do) for i in range(nlayers)])
        self.readout = nn.Conv1d(out_dim, out_dim, 1) if readout_layer else nn.Identity()
        self.out_shape = conv1D_out_shape(input_shape, width, width, [w//2 for w in width])
    def forward(self, x):
        return self.readout(self.convblocks(x))


class Multiview(nn.Module):
    def __init__(self, 
                 channels,
                 orig_channels,
                 num_classes,
                 conv_do = 0.1,
                 hidden_channels = 256, 
                 nlayers = 6,
                 out_dim = 64,
                 readout_layer = True,
                 projection_head = True,
                 n_layers_proj = 1,
                 embedding_dim = 32,
                 loss = 'time_loss',
                 **kwargs):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.out_dim = out_dim
        self.wave2vec = Wave2Vec(channels, input_shape = 33, out_dim = out_dim, 
                                 hidden_channels = hidden_channels, nlayers = nlayers, 
                                 norm = 'group', do = conv_do, readout_layer=readout_layer)
        self.classifier = TimeClassifier(in_features = out_dim, num_classes = num_classes, 
                                         pool = 'adapt_avg', orig_channels = orig_channels)
        self.projection_head = projection_head

        if projection_head:
            if not loss == 'time_loss':
                self.projector = TimeClassifier(in_features = out_dim, num_classes = embedding_dim,
                                                n_layers = n_layers_proj,
                                                pool = 'adapt_avg', orig_channels = orig_channels)
            else:
                self.projector = TimeProjector(in_features = out_dim, output_dim = embedding_dim, n_layers = n_layers_proj)
    
    def forward(self, x, classify = False):
        b, ch, ts = x.shape
        if ch > self.channels:
            x = x.view(b*ch, 1, ts)
        x = self.wave2vec(x)
        time_out = x.shape[-1]

        if self.projection_head and not classify:
            # only reshape after projection head
            out = self.projector(x)
            if len(out.shape) > 2:
                out = out.reshape(b, ch, -1, time_out)
            else:
                out = out.reshape(b, ch, -1)
            return out

        
        if ch > self.channels:
            x = x.view(b, ch, self.out_dim, time_out)
        if classify:
            return self.classifier(x)            
        else:
            return x

    def update_classifier(self, num_classes, orig_channels, seed = None):
        torch.manual_seed(seed)
        self.classifier = TimeClassifier(in_features = self.out_dim, num_classes = num_classes, pool = 'adapt_avg', orig_channels = orig_channels)

    def train_step(self, x, loss_fn, device):
        x = x.to(device)
        out = self.forward(x)
        loss = loss_fn(out)
        if isinstance(loss, tuple):
            return *loss
        else:
            return loss, *[torch.tensor(0)]*2

class GNNMultiview(nn.Module):
    def __init__(self, 
                 channels, 
                 time_length, 
                 num_classes, 
                 norm = 'group', 
                 conv_do = 0.1,
                 feat_do = 0.4,
                 num_message_passing_rounds = 2, 
                 hidden_channels = 256, 
                 nlayers = 6, 
                 out_dim = 64,
                 projection_head = True,
                 n_layers_proj = 1,
                 embedding_dim = 32,
                 readout_layer = True,
                 loss = 'time_loss',
                 **kwargs):
        super().__init__()
        self.channels = channels
        self.time_length = time_length
        self.num_classes = num_classes
        self.wave2vec = Wave2Vec(channels, input_shape = time_length, out_dim = out_dim, 
                                 hidden_channels = hidden_channels, nlayers = nlayers, norm = norm, 
                                 do = conv_do, readout_layer = readout_layer)


        self.out_dim = out_dim
        self.classifier = TimeClassifier(in_features = out_dim, num_classes = num_classes, 
                                         pool = 'adapt_avg', orig_channels = channels, 
                                         time_length = time_length)
        self.projection_head = projection_head

        if projection_head:
            if not loss == 'time_loss':
                self.projector = TimeClassifier(in_features = out_dim, num_classes = embedding_dim, 
                                            pool = 'adapt_avg', orig_channels = channels, n_layers=n_layers_proj,
                                            time_length = time_length)
            else:
                self.projector = TimeProjector(in_features = out_dim, output_dim = embedding_dim, n_layers=n_layers_proj)

        self.state_dim = out_dim
        print('out_dim', out_dim)
        
        # Message passing layers (from_state, to_state) -> message
        self.message_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_dim*2, out_dim),
                nn.Dropout(feat_do),
                nn.ReLU(),
        )
            for _ in range(num_message_passing_rounds)
        ])

        # Readout layer
        self.readout_net = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.Dropout(feat_do),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim) if readout_layer else nn.Identity(),
        )

    def forward(self, x, classify = False):
        b, ch, ts = x.shape
        x = x.view(b*ch, 1, ts)
        latents = self.wave2vec(x)

        view_id = torch.arange(b).unsqueeze(1).repeat(1, ch).view(-1).to(x.device)
        message_from = torch.arange(b*ch).unsqueeze(1).repeat(1, (ch-1)).view(-1).to(x.device)
        message_to = torch.arange(b*ch).view(b, ch).unsqueeze(1).repeat(1, ch, 1)
        idx = ~torch.eye(ch).view(1, ch, ch).repeat(b, 1, 1).bool()
        message_to = message_to[idx].view(-1).to(x.device)

        latents = latents.permute(0,2,1)

        for message_net in self.message_nets:
            # divide by ch-1 to take mean
            message = message_net(torch.cat([latents[message_from], latents[message_to]], dim=-1))/(ch-1)
            # Sum messages
            latents.index_add_(0, message_to.to(x.device), message)

        y = torch.zeros(b, *latents.shape[1:]).to(x.device)
        y.index_add_(0, view_id, latents)
        # divide by ch to take mean
        out_mpnn = self.readout_net(y)/ch

        out_mpnn = out_mpnn.permute(0,2,1)

        if classify:
            out = self.classifier(out_mpnn)
        elif self.projection_head:
            out = self.projector(out_mpnn)
        else:
            out = out_mpnn

        return out
        
    def get_view_pairs(self, view_id):
        """Yield all distinct pairs."""
        for i, vi in enumerate(view_id):
            for j, vj in enumerate(view_id):
                if vi == vj and i != j:
                    yield [i, j]

    def take_channel(self, A, indx):
        indx = indx.unsqueeze(2).repeat(1,1,A.shape[2])
        return torch.gather(A, 1, indx)

    def train_step(self, x, loss_fn, device):
        # partition the dataset into two views
        ch_size = np.random.randint(2, x.size(1)-1)
        random_channels = np.random.rand(x.size(1)).argpartition(x.size(1)-1)
        view_1_idx = random_channels[:ch_size] # randomly select ch_size channels per input
        view_2_idx = random_channels[ch_size:] # take the remaining as the second view
        #view_1 = self.take_channel(x, torch.tensor(view_1_idx).to(device))
        #view_2 = self.take_channel(x, torch.tensor(view_2_idx).to(device))
        view_1 = x[:, view_1_idx, :]
        view_2 = x[:, view_2_idx, :]

        out1 = self.forward(view_1)
        out2 = self.forward(view_2)

        out = torch.cat([out1.unsqueeze(1), out2.unsqueeze(1)], dim = 1)
        loss = loss_fn(out)

        sim = torch.tensor(0)
        avg_loss = torch.tensor(0)
        latent_loss = torch.tensor(0)

        if isinstance(loss, tuple):
            return *loss
        else:
            return loss, *[torch.tensor(0)]*2

def pretrain(model, 
            dataloader,
            val_dataloader,
            epochs,
            optimizer,
            device,
            loss_fn,
            backup_path = None,
            log = True):
    
    model.to(device)
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_inst = 0 
        epoch_temp = 0
        model.train()
        for i, data in enumerate(dataloader):
            x = data[0].to(device).float()
            optimizer.zero_grad()
            loss, inst_loss, temp_loss = model.train_step(x, loss_fn, device)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_inst += inst_loss.item()
            epoch_temp += temp_loss.item()

        train_loss = epoch_loss/(i+1)
        train_inst = epoch_inst/(i+1)
        train_temp = epoch_temp/(i+1)

        val_loss = 0
        val_inst = 0 
        val_temp = 0

        model.eval()
        for i, data in enumerate(val_dataloader):
            x = data[0].to(device).float()
            loss, inst_loss, temp_loss = model.train_step(x, loss_fn, device)
            val_loss += loss.item()
            val_inst += inst_loss.item()
            val_temp += temp_loss.item()

        if log:
            log_dict = {'val_loss': val_loss/(i+1), 'train_loss': train_loss}
            if inst_loss > 0:
                log_dict['train_inst_loss'] = train_inst
                log_dict['train_temp_loss'] = train_temp
                log_dict['val_inst_loss'] = val_inst/(i+1)
                log_dict['val_temp_loss'] = val_temp/(i+1)
            wandb.log(log_dict)

        if backup_path is not None:
            path = f'{backup_path}/pretrained_model.pt'
            torch.save(model.state_dict(), path)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    From https://github.com/Bjarten/early-stopping-pytorch """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def finetune(model,
            dataloader,
            val_dataloader,
            epochs,
            optimizer,
            weights,
            device,
            test_loader = None, 
            early_stopping_criterion = None,
            backup_path = None):
    model.to(device)
    loss = nn.CrossEntropyLoss(weight=weights)
    if early_stopping_criterion is not None:
        early_stopping = EarlyStopping(patience=7, verbose=True, path = f'{backup_path}/finetuned_model.pt')
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for i, data in enumerate(dataloader):
            x = data[0].to(device).float()
            y = data[-1].to(device)
            optimizer.zero_grad()
            out = model.forward(x, classify = True)
            loss_ = loss(out, y)
            loss_.backward()
            optimizer.step()
            epoch_loss += loss_.item()
        train_loss = epoch_loss/(i+1)
        val_loss = 0
        collect_y = []
        collect_pred = []

        model.eval()
        for i, data in enumerate(val_dataloader):
            x = data[0].to(device).float()
            y = data[-1].to(device)
            out = model.forward(x, classify = True)
            loss_ = loss(out, y)
            val_loss += loss_.item()
            collect_y.append(y.detach().cpu().numpy())
            collect_pred.append(out.argmax(dim=1).detach().cpu().numpy())
        collect_y = np.concatenate(collect_y)
        collect_pred = np.concatenate(collect_pred)
        acc = balanced_accuracy_score(collect_y, collect_pred)
        prec, rec, f, _ = precision_recall_fscore_support(collect_y, collect_pred)

        if test_loader is not None:
            test_acc, test_prec, test_rec, test_f = evaluate_classifier(model, test_loader, device)
            wandb.log({'train_class_loss': train_loss, 
                        'val_class_loss': val_loss/(i+1), 
                        'val_acc': acc, 
                        'val_prec': np.mean(prec), 
                        'val_rec': np.mean(rec), 
                        'val_f': np.mean(f),
                        'test_acc': test_acc,
                        'test_prec': np.mean(test_prec),
                        'test_rec': np.mean(test_rec),
                        'test_f': np.mean(test_f)
                        })
        else:
            wandb.log({'train_class_loss': train_loss, 
                        'val_class_loss': val_loss/(i+1), 
                        'val_acc': acc, 
                        'val_prec': np.mean(prec), 
                        'val_rec': np.mean(rec), 
                        'val_f': np.mean(f)
                        })
        if early_stopping_criterion is not None:
            if early_stopping_criterion == 'loss':
                early_stopping(val_loss/(i+1), model)
            elif early_stopping_criterion == 'acc':
                early_stopping(-acc, model)
            if early_stopping.early_stop:
                # load best model
                model.load_state_dict(torch.load(f'{backup_path}/finetuned_model.pt'))
                print("Early stopping")
                break

    if early_stopping.early_stop:
        # load best model
        model.load_state_dict(torch.load(f'{backup_path}/finetuned_model.pt'))

def evaluate_classifier(model,
                        test_loader,
                        device):
    model.eval()
    collect_y = []
    collect_pred = []
    for i, data in enumerate(test_loader):
        x = data[0].to(device).float()
        y = data[-1].to(device)
        out = model.forward(x, classify = True)
        collect_y.append(y.detach().cpu().numpy())
        collect_pred.append(out.argmax(dim=1).detach().cpu().numpy())
    collect_y = np.concatenate(collect_y)
    collect_pred = np.concatenate(collect_pred)
    acc = balanced_accuracy_score(collect_y, collect_pred)
    prec, rec, f, _ = precision_recall_fscore_support(collect_y, collect_pred)
    return acc, prec, rec, f


def load_model(pretraining_setup, device, channels, time_length, num_classes, model_args):
    torch.manual_seed(model_args.seed)
    if pretraining_setup == 'MPNN':
        model = GNNMultiview(channels = 1, time_length = time_length, num_classes = num_classes, **vars(model_args)).to(device)
    elif pretraining_setup == 'nonMPNN':
        model = Multiview(channels = 1, orig_channels=6, time_length = time_length, num_classes = num_classes, **vars(model_args)).to(device)
    elif pretraining_setup == 'None':
        model = Multiview(channels = channels, orig_channels=channels, time_length = time_length, num_classes = num_classes, **vars(model_args)).to(device)

        
    if model_args.loss == 'time_loss':
        loss_fn = CMCloss(temperature = 0.5, criterion='TS2Vec').to(device)
    elif model_args.loss == 'contrastive':
        loss_fn = CMCloss(temperature = 0.5, criterion='contrastive').to(device)
    elif model_args.loss == 'COCOA':
        loss_fn = COCOAloss(temperature = 0.5).to(device)

    return model, loss_fn
