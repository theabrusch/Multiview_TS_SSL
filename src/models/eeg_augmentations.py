import numpy as np
from scipy.signal import butter, sosfiltfilt
import torch
from src.models.ecg_augmentations import ChoiceEnum


PERTURBATION_CHOICES = ChoiceEnum(
    [
        "time_shift",
        "amplitude_scaling",
        "DCshift",
        "zero_masking",
        "gaussian_noise",
        "band_stop_filter",
    ]
)
MASKING_LEADS_STRATEGY_CHOICES = ChoiceEnum(["random", "conditional"])

def instantiate_from_name(str: PERTURBATION_CHOICES, **kwargs):
    if str == "time_shift":
        return TimeShift(**kwargs)
    elif str == "amplitude_scaling":
        return AmplitudeScaling(**kwargs)
    elif str == "DCshift":
        return DCShift(**kwargs)
    elif str == "zero_masking":
        return ZeroMasking(**kwargs)
    elif str == "gaussian_noise":
        return GaussianNoise(**kwargs)
    elif str == "band_stop_filter":
        return BandStopFilter(**kwargs)
    else:
        raise ValueError(f"inappropriate perturbation choices: {str}")

class TimeShift(object):
    def __init__(
        self,
        max_shift=80,
        window_length = 3000,
        p=1.0,
        **kwargs,
    ):
        self.max_shift = max_shift
        self.window_length = window_length
        self.p = p
    
    def __call__(self, sample):
        new_sample = sample.clone()
        if self.p > np.random.uniform(0,1):
            padding = int((new_sample.shape[1]-self.window_length)/2)
            max_shift = int(self.window_length/self.max_shift)
            shift = np.random.randint(1, max_shift)
            sign = np.random.choice([-1, 1])
            new_sample = new_sample[:,padding + shift * sign: padding + shift * sign + self.window_length]
        return new_sample.float()

class AmplitudeScaling(object):
    def __init__(
        self,
        max_scale=2,
        min_scale=0.5,
        window_length = 3000,
        p=1.0,
        **kwargs,
    ):
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.window_length = window_length
        self.p = p
    
    def __call__(self, sample):
        new_sample = sample.clone()
        if self.p > np.random.uniform(0,1):
            padding = int((new_sample.shape[1]-self.window_length)/2)
            scale = np.random.uniform(self.min_scale, self.max_scale)
            new_sample = new_sample[:,padding:padding+self.window_length] * scale
        return new_sample.float()

class DCShift(object):
    def __init__(
        self,
        max_shift=10,
        window_length = 3000,
        p=1.0,
        **kwargs,
    ):
        self.max_shift = max_shift
        self.window_length = window_length
        self.p = p
    
    def __call__(self, sample):
        new_sample = sample.clone()
        if self.p > np.random.uniform(0,1):
            padding = int((new_sample.shape[1]-self.window_length)/2)
            shift = np.random.uniform(-self.max_shift, self.max_shift)
            new_sample = new_sample[:,padding:padding+self.window_length] + shift
        return new_sample.float()

class ZeroMasking(object):
    def __init__(
        self,
        max_mask=0.0375,
        window_length = 3000,
        p=1.0,
        **kwargs,
    ):
        self.max_mask = max_mask
        self.window_length = window_length
        self.p = p
    
    def __call__(self, sample):
        new_sample = sample.clone()
        if self.p > np.random.uniform(0,1):
            padding = int((new_sample.shape[1]-self.window_length)/2)
            max_mask = int(self.window_length*self.max_mask)
            mask_length = np.random.randint(1, max_mask)
            mask_start = np.random.randint(0, self.window_length - mask_length)
            new_sample = new_sample[:,padding:padding+self.window_length]
            new_sample[:,mask_start:mask_start+mask_length] = 0
        return new_sample.float()
    
class GaussianNoise(object):
    def __init__(
        self,
        max_sigma=0.2,
        window_length = 3000,
        p=1.0,
        **kwargs,
    ):
        self.max_sigma = max_sigma
        self.window_length = window_length
        self.p = p
    
    def __call__(self, sample):
        new_sample = sample.clone()
        if self.p > np.random.uniform(0,1):
            padding = int((new_sample.shape[1]-self.window_length)/2)
            sigma = np.random.uniform(0, self.max_sigma)
            new_sample = new_sample[:,padding:padding+self.window_length] + np.random.normal(0, sigma, size = (new_sample.shape[0], self.window_length))
        return new_sample.float()

class BandStopFilter(object):
    def __init__(
        self,
        window_length = 3000,
        min_freq = 2.8,
        max_freq = 41.3,
        p=1.0,
        **kwargs,
    ):
        self.window_length = window_length
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.p = p
    
    def __call__(self, sample):
        new_sample = sample.clone()
        if self.p > np.random.uniform(0,1):
            padding = int((new_sample.shape[1]-self.window_length)/2)
            new_sample = new_sample[:,padding:padding+self.window_length]
            fs = 100
            center = np.random.uniform(self.min_freq, self.max_freq)
            width = 5
            low = (center - width/2) 
            high = (center + width/2)
            sos = butter(5, [low, high], btype='bandstop', fs=fs, output='sos')
            new_sample = sosfiltfilt(sos, new_sample)
            new_sample = torch.tensor(new_sample.copy())
        return new_sample.float()


class EEG_augmentations():
    def __init__(self, augmentation_choices = ["time_shift","amplitude_scaling","DCshift","zero_masking","gaussian_noise","band_stop_filter"], **kwargs):
        self.augmentations = [instantiate_from_name(augmentation_choice, **kwargs) for augmentation_choice in augmentation_choices]
    
    def __call__(self, sample):
        for augmentation in self.augmentations:
            sample = augmentation(sample)
        return sample.float()
