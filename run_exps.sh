LEN='dataset.segment_length=16000'
SNR=5

NOISYWAV=/data/ephraim/datasets/known_noise/noisy_wav/digits_snr5_var0.0015030844369903207.wav

OUTPATH="/data/ephraim/datasets/known_noise/enhanced_diffwave_1sec/snr${SNR}"
# #'CUDA_LAUNCH_BLOCKING=1 '
MDL='model=wavenet'

for i in 0.00005 0.00003 0.00001 0.000008 0.000006 0.000004 0.000002 0.000001; 
do HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0,1 python generate.py ${MDL} ${LEN} generate.addedoutputpath=${OUTPATH} generate.noisy_signal=${NOISYWAV} generate.guid_s=${i} ; 
done



