import torch
import numpy as np

def remove_frequency(x, pertub_ratio=0.0):
    mask = torch.FloatTensor(x.shape).uniform_() > pertub_ratio # maskout_ratio are False
    mask = mask
    return x*mask

def remove_frequency_abs_budget(x_f, E = 1):
    remove_freqs = torch.rand(x_f.shape).argsort(2)[:,:,:E]
    return x_f.scatter(-1, remove_freqs, 0)

def add_frequency(x, pertub_ratio=0.0):
    mask = torch.FloatTensor(x.shape).uniform_() > (1-pertub_ratio) # only pertub_ratio of all values are True
    mask = mask.to(x.device)
    max_amplitude = x.max()
    random_am = torch.rand(mask.shape)*(max_amplitude*0.1)
    pertub_matrix = mask*random_am
    return x+pertub_matrix

def add_frequency_abs_budget(x_f, alpha = 0.5, E = 1):
    
    for i, sample in enumerate(x_f):
        for j, row in enumerate(sample):
            max_freq = torch.max(row)*alpha
            low_freqs = torch.nonzero(row < max_freq).squeeze()
            add_freqs = torch.rand(low_freqs.shape).argsort()[:E]
            x_f[i,j,low_freqs[add_freqs]] = max_freq

    return x_f

def frequency_augmentation(freq_cont, keep_all = True, return_ifft = True, abs_budget = True):
    x_f = freq_cont.abs()

    if abs_budget:
        x_f_add = add_frequency_abs_budget(x_f)
        x_f_rem = remove_frequency_abs_budget(x_f)
    else:
        x_f_add = add_frequency(x_f)
        x_f_rem = remove_frequency(x_f)

    if keep_all:
        collect_x_f = torch.cat((x_f.unsqueeze(1), x_f_add.unsqueeze(1), x_f_rem.unsqueeze(1)), axis = 1)
        if return_ifft:
            x_f_phase = freq_cont.imag
            collect_x_t = torch.fft.ifft(collect_x_f + x_f_phase.unsqueeze(1), axis = -1).real
            return collect_x_f, collect_x_t
    else:
        collect_x_f = torch.cat((x_f_add.unsqueeze(1), x_f_rem.unsqueeze(1)), axis = 1)
    
    return collect_x_f

# Temporal augmentations

def jitter(x, sigma = 0.8):
    return x + np.random.normal(0, sigma, size = x.shape)

def scaling(x, sigma = 1.1):
    factor = np.random.normal(2, sigma, size = [x.shape[0], 1, x.shape[2]])
    return x * factor


def permutation(x, max_segments = 8, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

def time_augmentation(x, keep_all = True, return_fft = True):
    x_jitter = jitter(x.clone())
    x_scale = scaling(x.clone())
    x_perm = permutation(x.clone())

    if keep_all:
        collect_x_t = torch.cat((x.unsqueeze(1), x_jitter.unsqueeze(1), x_scale.unsqueeze(1), x_perm.unsqueeze(1)), axis = 1)
        if return_fft:
            collect_x_f = torch.fft.fft(collect_x_t, axis = -1).abs()
            return collect_x_t, collect_x_f
    else:
        collect_x_t = torch.cat((x_jitter.unsqueeze(1), x_scale.unsqueeze(1), x_perm.unsqueeze(1)), axis = 1)
    
    return collect_x_t

