import os
from glob import glob
from tqdm import tqdm

exp_dir = "/data/ephraim/datasets/known_noise/enhanced_diffwave_1sec/"
snr_dirs = glob(exp_dir+"*/")
for snr_dir in tqdm(snr_dirs): 
    # snr_dir = "/data/ephraim/datasets/known_noise/enhanced_diffwave_1sec/snr10/"
    s_dirs =glob(snr_dir+"*/")
    for d in tqdm(s_dirs):
        command = "python measure.py -enhanced_dir={} -out_stats_dir={}".format(d,snr_dir)
        os.system(command)

