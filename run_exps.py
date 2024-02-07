from glob import glob
import os

outdirname="enhanced_diffwave_1sec"
LEN='dataset.segment_length=16000'

MDL='model=wavenet'

noiswavs = glob("/data/ephraim/datasets/known_noise/noisy_wav/*")
outbasedir = "/data/ephraim/datasets/known_noise/{}/".format(outdirname)
for wav in noiswavs:
    snr = wav.split("snr")[1].split("_var")[0]
    s_array=[0.0000001, 0.000005, 0.000001, 0.00005, 0.00001, 0.0005, 0.0001, 0.005, 0.001, 0.05, 0.01, 0.1]
    if snr in ["5","10", "0", "-5", "-10"]:
        s_array=[0.05, 0.01, 0.1]
    OUTPATH = outbasedir+"snr{}".format(outdirname,snr)

# #'CUDA_LAUNCH_BLOCKING=1 '

    # for i in [0.00005, 0.00003, 0.00001, 0.000008, 0.000006, 0.000004, 0.000002, 0.000001]: 
    #     command = "HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0,1 python generate.py {} {} generate.addedoutputpath={} generate.noisy_signal={} generate.guid_s={}".format(MDL,LEN,OUTPATH, wav,i)
    #     os.system(command)
        
    for i in s_array: 
        command = "HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2,3 python generate.py {} {} generate.addedoutputpath={} generate.noisy_signal={} generate.guid_s={}".format(MDL,LEN,OUTPATH, wav,i)
        os.system(command)
        
command = "cd measure; python run_measure.py -exp_dir {}".format(outbasedir )
os.system(command)
        
