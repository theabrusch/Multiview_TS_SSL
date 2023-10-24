import torch
from torch import nn
import torch.nn.functional as F
import numpy as np 

class conv_block(nn.Module):
    def __init__(self, channels_in, channels_out, kernel, stride, dropout):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(channels_in, channels_out, kernel_size=kernel, stride = stride, padding = kernel//2, bias = False),
            nn.BatchNorm1d(channels_out),
            nn.ReLU(),
            nn.MaxPool1d(2,2),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.layer(x)

class conv_block2(nn.Module):
    def __init__(self, channels_in, channels_out, kernel, stride, dropout):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(channels_in, channels_out, kernel_size=kernel, stride = stride, padding = kernel//2, bias = False),
            nn.BatchNorm1d(channels_out),
            nn.ReLU(),
            nn.MaxPool1d(4,4),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.layer(x)

class wave2vecblock(nn.Module):
    def __init__(self, channels_in, channels_out, kernel, stride, dropout = 0.1, norm = 'group'):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(channels_in, channels_out, kernel_size=kernel, stride=stride, padding = kernel // 2),
            nn.Dropout1d(dropout),
            nn.GroupNorm(channels_out // 2, channels_out) if norm == 'group' else nn.BatchNorm1d(channels_out),
            nn.GELU()
        )
    def forward(self, x):
        return self.layer(x)

def conv1D_out_shape(input_shape, kernel, stride, padding):
    kernel, stride, padding = np.array(kernel), np.array(stride), np.array(padding)
    shape = input_shape
    for kern, stri, padd in zip(kernel, stride, padding):
        shape = int((shape + 2*padd - kern)/stri + 1)
    return shape

class PrintShape(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        print(x.shape)
        return x

class TFC_single_encoder(nn.Module):
    def __init__(self, 
                 in_channels, 
                 input_size, 
                 stride = 1, 
                 conv_dropout = 0., 
                 linear_dropout = 0.5, 
                 avg_channels_before = False, 
                 avg_channels_after = False, 
                 time_or_freq = 'time',
                 nlayers = 6, 
                 encoder_type = 'TFC'):
        super().__init__()
        self.avg_channels_before = avg_channels_before
        self.avg_channels_after = avg_channels_after
        self.time_or_freq = time_or_freq

        if self.avg_channels_before or self.avg_channels_after:
            in_channels = 1
        if encoder_type == 'TFC':
            out_shape = conv1D_out_shape(input_size, [8,2,8,2,8,2], [stride,2,1,2,1,2], [4,0,4,0,4,0])
            self.TimeEncoder = nn.Sequential(
                conv_block(channels_in = in_channels, channels_out = 32, kernel = 8, stride = stride, dropout = conv_dropout),
                conv_block(channels_in = 32, channels_out = 64, kernel = 8, stride = 1, dropout = conv_dropout),
                conv_block(channels_in = 64, channels_out = 128, kernel = 8, stride = 1, dropout = conv_dropout),
                nn.Flatten(),
                nn.ReLU(),
                nn.Linear(out_shape*128, 256)
                )
            
            channels_out = 128

        elif encoder_type == 'TFC2':
            out_shape = conv1D_out_shape(input_size, [8,4,8,4,8,4], [1,4,1,4,1,4], [4,0,4,0,4,0])
            self.TimeEncoder = nn.Sequential(
                conv_block2(channels_in = in_channels, channels_out = 32, kernel = 8, stride = 1, dropout = conv_dropout),
                conv_block2(channels_in = 32, channels_out = 64, kernel = 8, stride = 1, dropout = conv_dropout),
                conv_block2(channels_in = 64, channels_out = 128, kernel = 8, stride = 1, dropout = conv_dropout),
                nn.Flatten(),
                nn.ReLU(),
                nn.Linear(out_shape*128, 256)
                )
            
            channels_out = 128

        elif encoder_type == 'wave2vec':
            self.TimeEncoder = nn.Sequential()
            channels_out = 512
            out_shape = conv1D_out_shape(input_size, width, width, np.array(width)//2)
            width = [3] + [2]*(nlayers-1)
            in_channels = [in_channels] + [512]*(nlayers-1)
            for i, (w, in_ch) in enumerate(zip(width, in_channels)):
                self.TimeEncoder.add_module(f'Time_encoder_{i}', wave2vecblock(in_ch, channels_out, kernel = w, stride = w))
            self.TimeEncoder.add_module('flatten', nn.Flatten())
            self.TimeEncoder.add_module('gelu', nn.GELU())
            self.TimeEncoder.add_module('linear', nn.Linear(out_shape*channels_out, 256))
            

    def forward(self, x_t, x_f, finetune = False):
        if self.time_or_freq == 'freq':
            x = x_f
        else:
            x = x_t
            
        if self.avg_channels_before or self.avg_channels_after:
            batch_size = x.shape[0]
            n_channels = x.shape[1]
            time_len = x.shape[2]
            x = torch.reshape(x, (batch_size*n_channels, 1, time_len))
        
        h = self.TimeEncoder(x)
        if self.avg_channels_before:
            n_h_features = h.shape[-1]
            h = torch.reshape(h, (batch_size, n_channels, n_h_features)).mean(dim = 1)

        if finetune and self.avg_channels_after:
            n_h_features = h.shape[-1]
            h = torch.reshape(h, (batch_size, n_channels, n_h_features))

        if self.time_or_freq == 'freq':
            h_f = h
            h_t = None
        else:
            h_t = h
            h_f = None

        return h_t, None, h_f, None


class TFC_encoder(nn.Module):
    def __init__(self, in_channels, input_size, stride = 1, conv_dropout = 0., linear_dropout = 0.5, avg_channels_before = False, avg_channels_after = False, nlayers = 6, encoder_type = 'TFC'):
        super().__init__()
        self.avg_channels_before = avg_channels_before
        self.avg_channels_after = avg_channels_after

        if self.avg_channels_before or self.avg_channels_after:
            in_channels = 1
        if encoder_type == 'TFC':
            self.TimeEncoder = nn.Sequential(
                conv_block(channels_in = in_channels, channels_out = 32, kernel = 8, stride = stride, dropout = conv_dropout),
                conv_block(channels_in = 32, channels_out = 64, kernel = 8, stride = 1, dropout = conv_dropout),
                conv_block(channels_in = 64, channels_out = 128, kernel = 8, stride = 1, dropout = conv_dropout),
                nn.Flatten()
                )
            
            self.FrequencyEncoder = nn.Sequential(
                conv_block(channels_in = in_channels, channels_out = 32, kernel = 8, stride = stride, dropout = conv_dropout),
                conv_block(channels_in = 32, channels_out = 64, kernel = 8, stride = 1, dropout = conv_dropout),
                conv_block(channels_in = 64, channels_out = 128, kernel = 8, stride = 1, dropout = conv_dropout),
                nn.Flatten()
                )
            out_shape = conv1D_out_shape(input_size, [8,2,8,2,8,2], [stride,2,1,2,1,2], [4,0,4,0,4,0])
            channels_out = 128

        elif encoder_type == 'TFC2':
            self.TimeEncoder = nn.Sequential(
                conv_block2(channels_in = in_channels, channels_out = 32, kernel = 8, stride = 1, dropout = conv_dropout),
                conv_block2(channels_in = 32, channels_out = 64, kernel = 8, stride = 1, dropout = conv_dropout),
                conv_block2(channels_in = 64, channels_out = 128, kernel = 8, stride = 1, dropout = conv_dropout),
                nn.Flatten()
                )
            
            self.FrequencyEncoder = nn.Sequential(
                conv_block2(channels_in = in_channels, channels_out = 32, kernel = 8, stride = 1, dropout = conv_dropout),
                conv_block2(channels_in = 32, channels_out = 64, kernel = 8, stride = 1, dropout = conv_dropout),
                conv_block2(channels_in = 64, channels_out = 128, kernel = 8, stride = 1, dropout = conv_dropout),
                nn.Flatten()
                )
            out_shape = conv1D_out_shape(input_size, [8,4,8,4,8,4], [1,4,1,4,1,4], [4,0,4,0,4,0])
            channels_out = 128

        elif encoder_type == 'wave2vec':
            self.TimeEncoder = nn.Sequential()
            self.FrequencyEncoder = nn.Sequential()
            channels_out = 512
            width = [3] + [2]*(nlayers-1)
            in_channels = [in_channels] + [512]*(nlayers-1)
            for i, (w, in_ch) in enumerate(zip(width, in_channels)):
                self.TimeEncoder.add_module(f'Time_encoder_{i}', wave2vecblock(in_ch, channels_out, kernel = w, stride = w))
                self.FrequencyEncoder.add_module(f'Freq_encoder_{i}', wave2vecblock(in_ch, channels_out, kernel = w, stride = w))
            self.TimeEncoder.add_module('flatten', nn.Flatten())
            self.FrequencyEncoder.add_module('flatten', nn.Flatten())
            out_shape = conv1D_out_shape(input_size, width, width, np.array(width)//2)

        self.TimeCrossSpace = nn.Sequential(
            nn.Linear(in_features=out_shape*channels_out, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(linear_dropout),
            nn.Linear(in_features=256, out_features=128),            
        )
        self.FreqCrossSpace = nn.Sequential(
            nn.Linear(in_features=out_shape*channels_out, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(linear_dropout),
            nn.Linear(in_features=256, out_features=128),            
        )

    def forward(self, x_t, x_f, finetune = False):
        if self.avg_channels_before or self.avg_channels_after:
            batch_size = x_t.shape[0]
            n_channels = x_t.shape[1]
            time_len = x_t.shape[2]
            x_t = torch.reshape(x_t, (batch_size*n_channels, 1, time_len))
            x_f = torch.reshape(x_f, (batch_size*n_channels, 1, time_len))
        
        h_t = self.TimeEncoder(x_t)
        if self.avg_channels_before:
            n_h_features = h_t.shape[-1]
            h_t = torch.reshape(h_t, (batch_size, n_channels, n_h_features)).mean(dim = 1)
        z_t = self.TimeCrossSpace(h_t)

        h_f = self.FrequencyEncoder(x_f)
        if self.avg_channels_before:
            h_f = torch.reshape(h_f, (batch_size, n_channels, n_h_features)).mean(dim = 1)
        z_f = self.FreqCrossSpace(h_f)
        
        if finetune and self.avg_channels_after:
            n_h_features = h_t.shape[-1]
            n_z_features = z_t.shape[-1]
            h_t = torch.reshape(h_t, (batch_size, n_channels, n_h_features))
            h_f = torch.reshape(h_f, (batch_size, n_channels, n_h_features))
            z_t = torch.reshape(z_t, (batch_size, n_channels, n_z_features))
            z_f = torch.reshape(z_f, (batch_size, n_channels, n_z_features))


        return h_t, z_t, h_f, z_f

class ClassifierModule(nn.Module):
    def __init__(self, num_classes, avg_channels = False):
        super().__init__()
        self.avg_channels = avg_channels
        self.classifier = nn.Linear(in_features=256, out_features=num_classes)
    def forward(self, x):
        if self.avg_channels:
            x = x.mean(dim = 1)
        return self.classifier(x)


class ContrastiveLoss(nn.Module):

    def __init__(self, device, tau, use_cosine_similarity = True):
        super(ContrastiveLoss, self).__init__()
        self.temperature = tau
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs, reduce = True):
        if zis is None or zjs is None:
            return torch.tensor(0.).to(self.device)
        
        batch_size = zis.shape[0]
        representations = torch.cat([zjs, zis], dim=0)

        mask_samples_from_same_repr = self._get_correlated_mask(batch_size=batch_size).type(torch.bool)

        similarity_matrix = self.similarity_function(representations, representations)
        if len(similarity_matrix.shape) > 2:
            similarity_matrix = similarity_matrix.mean(dim = 2)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        """Criterion has an internal one-hot function. Here, make all positives as 1 while all negatives as 0. """
        labels = torch.zeros(2 * batch_size).to(self.device).long()
        CE = self.criterion(logits, labels)

        onehot_label = torch.cat((torch.ones(2 * batch_size, 1),torch.zeros(2 * batch_size, negatives.shape[-1])),dim=-1).to(self.device).long()
        # Add poly loss
        pt = torch.mean(onehot_label* torch.nn.functional.softmax(logits,dim=-1))

        epsilon = batch_size
        # loss = CE/ (2 * self.batch_size) + epsilon*(1-pt) # replace 1 by 1/self.batch_size
        loss = CE / (2 * batch_size) + epsilon * (1/batch_size - pt)
        # loss = CE / (2 * self.batch_size)

        return loss


class ContrastiveLoss2(nn.Module):
    def __init__(self, tau, device, reduce):
        super().__init__()
        self.tau = tau
        self.device = device
        self.cosine_similarity = nn.CosineSimilarity(dim = -1)
        self.reduce = reduce
        if self.reduce:
            reduction = 'sum'
        else:
            reduction = 'none'
        self.criterion = nn.CrossEntropyLoss(reduction = reduction)


    def forward(self, z_orig, z_augment, reduce = True):
        if z_orig is None or z_augment is None:
            return torch.tensor(0.).to(self.device)
        
        batchsize = z_orig.shape[0]
        collect_z = torch.cat([z_augment, z_orig], dim = 0)
        # calculate cosine similarity between all augmented and
        # non-augmented latent representations
        similarities = self.cosine_similarity(collect_z.unsqueeze(1), collect_z.unsqueeze(0))
        # compute negative mask
        upper = np.eye(2*batchsize, 2*batchsize, k = batchsize)
        lower = np.eye(2*batchsize, 2*batchsize, k = -batchsize)
        diag = np.eye(2*batchsize)

        self.negative_mask = torch.from_numpy(1-(upper+lower+diag)).bool().to(self.device)

        # get the positive samples (upper and lower diagonal)
        upper_pos = torch.diag(similarities, batchsize)
        lower_pos = torch.diag(similarities, -batchsize)
        positive = torch.cat((upper_pos, lower_pos), dim = 0).unsqueeze(1)

        # get the negative samples by masking out the diagonal and the upper 
        # and lower diagonal
        negative = similarities[self.negative_mask].view(2*batchsize, -1)

        # concatenate the logits
        logits = torch.cat((positive, negative), dim = 1)
        logits /= self.tau
        
        # calculate loss using cross entropy by setting the "labels" to zero:
        # we have the positive samples on the 0th index in the logits and all 
        # other samples on the remaining indices - i.e. force the 0th "class"
        # to be large and all other to be smal
        labels = torch.zeros(2*batchsize).to(self.device).long()
        loss = self.criterion(logits, labels) 
        if reduce:
            loss = loss / (2*batchsize)
        return loss 
