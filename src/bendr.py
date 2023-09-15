import copy
import mne
import parse
import tqdm
from src.multiview import TimeClassifier

import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from math import ceil
from pathlib import Path

from dn3.trainable.processes import StandardClassification, BaseProcess
from dn3.trainable.layers import Flatten, Permute
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
import wandb


class BendingCollegeWav2Vec(BaseProcess):
    """
    A more wav2vec 2.0 style of constrastive self-supervision, more inspired-by than exactly like it.
    """
    def __init__(self, encoder, context_fn, mask_rate=0.1, mask_span=6, learning_rate=0.01, temp=0.5,
                 permuted_encodings=False, permuted_contexts=False, enc_feat_l2=0.001, multi_gpu=False,
                 l2_weight_decay=1e-4, unmasked_negative_frac=0.25, encoder_grad_frac=1.0,
                 num_negatives=100, **kwargs):
        self.predict_length = mask_span
        self._enc_downsample = encoder.downsampling_factor
        if multi_gpu:
            encoder = nn.DataParallel(encoder)
            context_fn = nn.DataParallel(context_fn)
        if encoder_grad_frac < 1:
            encoder.register_backward_hook(lambda module, in_grad, out_grad:
                                           tuple(encoder_grad_frac * ig for ig in in_grad))
        super(BendingCollegeWav2Vec, self).__init__(encoder=encoder, context_fn=context_fn,
                                                    loss_fn=nn.CrossEntropyLoss(), lr=learning_rate,
                                                    l2_weight_decay=l2_weight_decay,
                                                    metrics=dict(Accuracy=self._contrastive_accuracy,
                                                                 Mask_pct=self._mask_pct), **kwargs)
        self.best_metric = None
        self.mask_rate = mask_rate
        self.mask_span = mask_span
        self.temp = temp
        self.permuted_encodings = permuted_encodings
        self.permuted_contexts = permuted_contexts
        self.beta = enc_feat_l2
        self.start_token = getattr(context_fn, 'start_token', None)
        self.unmasked_negative_frac = unmasked_negative_frac
        self.num_negatives = num_negatives

    def description(self, sequence_len):
        encoded_samples = self._enc_downsample(sequence_len)
        desc = "{} samples | mask span of {} at a rate of {} => E[masked] ~= {}".format(
            encoded_samples, self.mask_span, self.mask_rate,
            int(encoded_samples * self.mask_rate * self.mask_span))
        return desc

    def _generate_negatives(self, z):
        """Generate negative samples to compare each sequence location against"""
        batch_size, feat, full_len = z.shape
        z_k = z.permute([0, 2, 1]).reshape(-1, feat)
        with torch.no_grad():
            # candidates = torch.arange(full_len).unsqueeze(-1).expand(-1, self.num_negatives).flatten()
            negative_inds = torch.randint(0, full_len-1, size=(batch_size, full_len * self.num_negatives))
            # From wav2vec 2.0 implementation, I don't understand
            # negative_inds[negative_inds >= candidates] += 1

            for i in range(1, batch_size):
                negative_inds[i] += i * full_len

        z_k = z_k[negative_inds.view(-1)].view(batch_size, full_len, self.num_negatives, feat)
        return z_k, negative_inds

    def _calculate_similarity(self, z, c, negatives):
        c = c[..., 1:].permute([0, 2, 1]).unsqueeze(-2)
        z = z.permute([0, 2, 1]).unsqueeze(-2)

        # In case the contextualizer matches exactly, need to avoid divide by zero errors
        negative_in_target = (c == negatives).all(-1)
        targets = torch.cat([c, negatives], dim=-2)

        logits = F.cosine_similarity(z, targets, dim=-1) / self.temp
        if negative_in_target.any():
            logits[1:][negative_in_target] = float("-inf")

        return logits.view(-1, logits.shape[-1])

    def forward(self, *inputs):
        z = self.encoder(inputs[0])

        if self.permuted_encodings:
            z = z.permute([1, 2, 0])

        unmasked_z = z.clone()

        batch_size, feat, samples = z.shape

        if self._training:
            mask = _make_mask((batch_size, samples), self.mask_rate, samples, self.mask_span)
        else:
            mask = torch.zeros((batch_size, samples), requires_grad=False, dtype=torch.bool)
            half_avg_num_seeds = max(1, int(samples * self.mask_rate * 0.5))
            if samples <= self.mask_span * half_avg_num_seeds:
                raise ValueError("Masking the entire span, pointless.")
            mask[:, _make_span_from_seeds((samples // half_avg_num_seeds) * np.arange(half_avg_num_seeds).astype(int),
                                              self.mask_span)] = True

        c = self.context_fn(z, mask)

        # Select negative candidates and generate labels for which are correct labels
        negatives, negative_inds = self._generate_negatives(z)

        # Prediction -> batch_size x predict_length x predict_length
        logits = self._calculate_similarity(unmasked_z, c, negatives)
        return logits, z, mask

    @staticmethod
    def _mask_pct(inputs, outputs):
        return outputs[2].float().mean().item()

    @staticmethod
    def _contrastive_accuracy(inputs, outputs):
        logits = outputs[0]
        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        return StandardClassification._simple_accuracy([labels], logits)

    def calculate_loss(self, inputs, outputs):
        logits = outputs[0]
        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        # Note the loss_fn here integrates the softmax as per the normal classification pipeline (leveraging logsumexp)
        return self.loss_fn(logits, labels) + self.beta * outputs[1].pow(2).mean()



class _BENDREncoder(nn.Module):
    def __init__(self, in_features, encoder_h=256,):
        super().__init__()
        self.in_features = in_features
        self.encoder_h = encoder_h

    def load(self, filename, strict=True):
        state_dict = torch.load(filename, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def freeze_features(self, unfreeze=False):
        for param in self.parameters():
            param.requires_grad = unfreeze


class ConvEncoderBENDR(_BENDREncoder):
    def __init__(self, in_features, encoder_h=256, out_dim = 64, enc_width=(3, 2, 2, 2, 2, 2),
                 dropout=0.1, enc_downsample=(3, 2, 2, 2, 2, 2), orig_bendr = False):
        super().__init__(in_features, encoder_h)
        self.encoder_h = encoder_h
        if not isinstance(enc_width, (list, tuple)):
            enc_width = [enc_width]
        if not isinstance(enc_downsample, (list, tuple)):
            enc_downsample = [enc_downsample]
        assert len(enc_downsample) == len(enc_width)

        self._downsampling = enc_downsample
        self._width = enc_width
        if orig_bendr:
            enc_width = [e if e % 2 else e+1 for e in enc_width]

        self.encoder = nn.Sequential()
        for i, (width, downsample) in enumerate(zip(enc_width, enc_downsample)):
            if i == len(enc_width)-1:
                encoder_h = out_dim
            self.encoder.add_module("Encoder_{}".format(i), nn.Sequential(
                nn.Conv1d(in_features, encoder_h, width, stride=downsample, padding=width // 2),
                nn.Dropout2d(dropout),
                nn.GroupNorm(encoder_h // 2, encoder_h),
                nn.GELU(),
            ))
            in_features = encoder_h

    def description(self, sfreq=None, sequence_len=None):
        widths = list(reversed(self._width))[1:]
        strides = list(reversed(self._downsampling))[1:]

        rf = self._width[-1]
        for w, s in zip(widths, strides):
            rf = rf if w == 1 else (rf - 1) * s + 2 * (w // 2)

        desc = "Receptive field: {} samples".format(rf)
        if sfreq is not None:
            desc += ", {:.2f} seconds".format(rf / sfreq)

        ds_factor = np.prod(self._downsampling)
        desc += " | Downsampled by {}".format(ds_factor)
        if sfreq is not None:
            desc += ", new sfreq: {:.2f} Hz".format(sfreq / ds_factor)
        desc += " | Overlap of {} samples".format(rf - ds_factor)
        if sequence_len is not None:
            desc += " | {} encoded samples/trial".format(sequence_len // ds_factor)
        return desc

    def downsampling_factor(self, samples):
        for factor in self._downsampling:
            samples = ceil(samples / factor)
        return samples

    def forward(self, x):
        return self.encoder(x)



def _make_span_from_seeds(seeds, span, total=None):
    inds = list()
    for seed in seeds:
        for i in range(seed, seed + span):
            if total is not None and i >= total:
                break
            elif i not in inds:
                inds.append(int(i))
    return np.array(inds)


def _make_mask(shape, p, total, span, allow_no_inds=False):
    # num_mask_spans = np.sum(np.random.rand(total) < p)
    # num_mask_spans = int(p * total)
    mask = torch.zeros(shape, requires_grad=False, dtype=torch.bool)

    for i in range(shape[0]):
        mask_seeds = list()
        while not allow_no_inds and len(mask_seeds) == 0 and p > 0:
            mask_seeds = np.nonzero(np.random.rand(total) < p)[0]

        mask[i, _make_span_from_seeds(mask_seeds, span, total=total)] = True

    return mask


class _Hax(nn.Module):
    """T-fixup assumes self-attention norms are removed"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class BENDRContextualizer(nn.Module):

    def __init__(self, in_features, hidden_feedforward=3076, heads=8, layers=8, dropout=0.15, activation='gelu',
                 position_encoder=25, layer_drop=0.0, mask_p_t=0.1, mask_p_c=0.004, mask_t_span=6, mask_c_span=64,
                 start_token=-5, finetuning=False):
        super().__init__()

        self.dropout = dropout
        self.in_features = in_features
        self._transformer_dim = in_features * 3

        encoder = nn.TransformerEncoderLayer(d_model=in_features * 3, nhead=heads, dim_feedforward=hidden_feedforward,
                                             dropout=dropout, activation=activation)
        encoder.norm1 = _Hax()
        encoder.norm2 = _Hax()

        self.norm = nn.LayerNorm(self._transformer_dim)

        # self.norm_layers = nn.ModuleList([copy.deepcopy(norm) for _ in range(layers)])
        self.transformer_layers = nn.ModuleList([copy.deepcopy(encoder) for _ in range(layers)])
        self.layer_drop = layer_drop
        self.p_t = mask_p_t
        self.p_c = mask_p_c
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        self.start_token = start_token
        self.finetuning = finetuning

        # Initialize replacement vector with 0's
        self.mask_replacement = torch.nn.Parameter(torch.normal(0, in_features**(-0.5), size=(in_features,)),
                                                   requires_grad=True)

        self.position_encoder = position_encoder > 0
        if position_encoder:
            conv = nn.Conv1d(in_features, in_features, position_encoder, padding=position_encoder // 2, groups=16)
            nn.init.normal_(conv.weight, mean=0, std=2 / self._transformer_dim)
            nn.init.constant_(conv.bias, 0)
            conv = nn.utils.weight_norm(conv, dim=2)
            self.relative_position = nn.Sequential(conv, nn.GELU())

        self.input_conditioning = nn.Sequential(
            Permute([0, 2, 1]),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            Permute([0, 2, 1]),
            nn.Conv1d(in_features, self._transformer_dim, 1),
            Permute([2, 0, 1]),
        )

        self.output_layer = nn.Conv1d(self._transformer_dim, in_features, 1)
        self.apply(self.init_bert_params)

    def init_bert_params(self, module):
        if isinstance(module, nn.Linear):
            # module.weight.data.normal_(mean=0.0, std=0.02)
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
            # Tfixup
            module.weight.data = 0.67 * len(self.transformer_layers) ** (-0.25) * module.weight.data

        # if isinstance(module, nn.Conv1d):
        #     # std = np.sqrt((4 * (1.0 - self.dropout)) / (self.in_features * self.in_features))
        #     # module.weight.data.normal_(mean=0.0, std=std)
        #     nn.init.xavier_uniform_(module.weight.data)
        #     module.bias.data.zero_()

    def forward(self, x, mask_t=None, mask_c=None):
        bs, feat, seq = x.shape
        if self.training and self.finetuning:
            if mask_t is None and self.p_t > 0:
                mask_t = _make_mask((bs, seq), self.p_t, x.shape[-1], self.mask_t_span)
            if mask_c is None and self.p_c > 0:
                mask_c = _make_mask((bs, feat), self.p_c, x.shape[1], self.mask_c_span)

        if mask_t is not None:
            x = x*(1-mask_t.unsqueeze(1).int()) + (mask_t.unsqueeze(1).repeat(1,x.shape[1],1)).int()*self.mask_replacement.unsqueeze(0).unsqueeze(2).repeat(x.shape[0],1,x.shape[2])
            # x.transpose(2,1)[mask_t] = self.mask_replacement
        if mask_c is not None:
            x[mask_c] = 0

        if self.position_encoder:
            x = x + self.relative_position(x)
        x = self.input_conditioning(x)

        if self.start_token is not None:
            in_token = self.start_token * torch.ones((1, 1, 1), requires_grad=True).to(x.device).expand([-1, *x.shape[1:]])
            x = torch.cat([in_token, x], dim=0)

        for layer in self.transformer_layers:
            if not self.training or torch.rand(1) > self.layer_drop:
                x = layer(x)

        return self.output_layer(x.permute([1, 2, 0]))

    def freeze_features(self, unfreeze=False, finetuning=False):
        for param in self.parameters():
            param.requires_grad = unfreeze
        if self.finetuning or finetuning:
            self.mask_replacement.requires_grad = False

    def load(self, filename, strict=True):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

class BENDRClassifier(nn.Module):
    def __init__(self, encoder, num_classes, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.classifier = TimeClassifier(hidden_dim, num_classes, 'adapt_avg', orig_channels = 2, time_length = 32)
    
    def forward(self, x, classify = True):
        x = self.encoder(x)
        return self.classifier(x)

