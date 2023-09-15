import torch
from torch import nn
import torch.nn.functional as F
import numpy as np 
from src.multiview import Wave2Vec, TimeClassifier

class GRU_resblock(nn.Module):
    def __init__(self, input_shape):
        super(GRU_resblock, self).__init__()
        self.layer_norm = nn.LayerNorm(input_shape)
        self.GRU = nn.GRU(input_shape, input_shape, batch_first = True)
    
    def forward(self, x):
        out = self.layer_norm(x)
        out, _ = self.GRU(x)
        return x + out

class GRU_encoder(nn.Module):
    def __init__(self):
        super(GRU_encoder, self).__init__()
        self.GRU_1 = nn.GRU(1, 256, batch_first = True)
        self.GRU_2 = nn.GRU(1, 128, batch_first = True)
        self.GRU_3 = nn.GRU(1, 64, batch_first = True)
    
    def forward(self, x):
        out1, _ = self.GRU_1(x)
        x_down = F.interpolate(x.transpose(1,2), scale_factor = 0.5).transpose(1,2)
        out2, _ = self.GRU_2(x_down)
        x_down = F.interpolate(x_down.transpose(1,2), scale_factor = 0.5).transpose(1,2)
        out3, _ = self.GRU_3(x_down)

        # upsample out2 and out3
        out2 = F.interpolate(out2.transpose(1,2), scale_factor = 2).transpose(1,2)
        out3 = F.interpolate(out3.transpose(1,2), scale_factor = 4).transpose(1,2)
        return torch.cat([out1, out2, out3], dim = -1)

class SeqCLR_R(nn.Module):
    def __init__(self):
        super(SeqCLR_R, self).__init__()
        self.encoder = nn.Sequential(GRU_encoder(),
                                     nn.Linear(448, 128),
                                     nn.ReLU(),
                                     GRU_resblock(128),
                                     nn.Linear(128, 4))
    def forward(self, x):
        return self.encoder(x)

class Conv_resblock(nn.Module):
    def __init__(self):
        super(Conv_resblock, self).__init__()
        self.linear = nn.Linear(250, 250)
        self.conv_layer = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(250), 
            nn.Conv1d(250, 250, kernel_size=33, stride = 1, padding = 16, padding_mode='reflect'),
        )
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.linear(x)
        x = x.transpose(1,2)
        return x + self.conv_layer(x)

class Conv_encoder(nn.Module):
    def __init__(self):
        super(Conv_encoder, self).__init__()
        self.conv_1 = nn.Conv1d(1, 100, kernel_size=65, stride = 1, padding = 32, padding_mode='reflect')
        self.conv_2 = nn.Conv1d(1, 100, kernel_size=33, stride = 1, padding = 16, padding_mode='reflect')
        self.conv_3 = nn.Conv1d(1, 50, kernel_size=17, stride = 1, padding = 8, padding_mode='reflect')
    def forward(self, x):
        out1 = self.conv_1(x)
        out2 = self.conv_2(x)
        out3 = self.conv_3(x)
        return torch.cat([out1, out2, out3], dim = 1)

class SeqCLR_C(nn.Module):
    def __init__(self):
        super(SeqCLR_C, self).__init__()
        self.encoder = Conv_encoder()
        resblocks = []
        for i in range(4):
            resblocks.append(Conv_resblock())
        self.resblocks = nn.Sequential(*resblocks)
        self.output_layer = nn.Sequential(
            nn.ReLU(), 
            nn.BatchNorm1d(250),
            nn.Conv1d(250, 4, kernel_size=33, stride = 1, padding = 16, padding_mode='reflect'))

    def forward(self, x):
        x = x.transpose(1,2)
        x = self.encoder(x)
        x = self.resblocks(x)
        return self.output_layer(x).transpose(1,2)
    
class SeqCLR_W(nn.Module):
    def __init__(self, 
                 channels = 1,
                 conv_do = 0.1,
                 hidden_channels = 256, 
                 nlayers = 6,
                 out_dim = 64,
                 readout_layer = True,
                 **kwargs):
        super().__init__()
        self.channels = channels
        self.out_dim = out_dim
        self.wave2vec = Wave2Vec(channels, input_shape = 33, out_dim = out_dim, 
                                 hidden_channels = hidden_channels, nlayers = nlayers, 
                                 norm = 'group', do = conv_do, readout_layer=readout_layer)
    def forward(self, x, classify = False):
        x = x.transpose(1,2)
        x = self.wave2vec(x)
        return x
        
class SeqProjector(nn.Module):
    def __init__(self, input_dim = 4, output_dim = 32):
        super(SeqProjector, self).__init__()
        self.LSTM_1 = nn.LSTM(input_dim, 256, batch_first = True, bidirectional = True)
        self.LSTM_2 = nn.LSTM(input_dim, 128, batch_first = True, bidirectional = True)
        self.LSTM_3 = nn.LSTM(input_dim, 64, batch_first = True,  bidirectional = True)

        self.linear_layer = nn.Sequential(
            nn.Linear(896, 128), 
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        _, (h_1, _) = self.LSTM_1(x)
        x_down = F.interpolate(x.transpose(1,2), scale_factor = 0.5).transpose(1,2)
        _, (h_2, _) = self.LSTM_2(x_down)
        x_down = F.interpolate(x_down.transpose(1,2), scale_factor = 0.5).transpose(1,2)
        _, (h_3, _) = self.LSTM_3(x_down)
        # flatten the hidden states
        h_1, h_2, h_3 = h_1.transpose(0,1), h_2.transpose(0,1), h_3.transpose(0,1)
        h_1, h_2, h_3 = h_1.reshape(h_1.shape[0], -1), h_2.reshape(h_2.shape[0], -1), h_3.reshape(h_3.shape[0], -1)
        out = torch.cat([h_1, h_2, h_3], dim = -1)
        
        out = self.linear_layer(out)
        return out
    

class SeqCLR_classifier(nn.Module):
    def __init__(self, encoder, channels, num_classes, out_dim, classifier = 'SeqProjector'):
        super(SeqCLR_classifier, self).__init__()
        self.encoder = encoder
        self.channels = channels
        self.classifier_type = classifier
        self.out_dim = out_dim

        if self.classifier_type == 'SeqProjector':
            self.classifier = SeqProjector(input_dim = 4*channels, output_dim = num_classes)
        elif self.classifier_type == 'TimeClassifier':
            self.classifier = TimeClassifier(in_features = out_dim, num_classes = num_classes, 
                                             pool = 'adapt_avg', orig_channels = channels)
    def forward(self, x, classify = True):
        b_size = x.shape[0]
        x = x.reshape(b_size*self.channels, -1, 1)
        x = self.encoder(x)
        if self.classifier_type == 'SeqProjector':
            x = x.reshape(b_size, self.channels, -1, 4).transpose(2,3).reshape(b_size, 4*self.channels, -1).transpose(1,2)
        else:
            x = x.reshape(b_size, self.channels, self.out_dim, -1)

        return self.classifier(x)


class DummyProjector(nn.Module):
    def __init__(self):
        super(DummyProjector, self).__init__()

    def forward(self, x):
        return x