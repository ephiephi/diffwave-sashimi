
import torch
import torchaudio
from glob import glob

filenames = glob(f'/data/ephraim/datasets/sc09/wavs/*.wav')
for f in filenames:
    noisy_signal_, sr =torchaudio.load(f)
    power_x0_a = 1 / noisy_signal_.shape[1] * torch.sum(noisy_signal_**2)
    power_x0_b =  1 / noisy_signal_.shape[1] * torch.sum(noisy_signal_**2)