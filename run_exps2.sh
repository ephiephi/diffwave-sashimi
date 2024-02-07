LEN='dataset.segment_length=16000'
MDL='model=wavenet'
# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2,3 python generate.py ${MDL} ${LEN} generate.guid_s=0.000002
# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2,3 python generate.py ${MDL} ${LEN} generate.guid_s=0.000003
# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2,3 python generate.py ${MDL} ${LEN} generate.guid_s=0.000004

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2,3 python generate.py ${MDL} ${LEN} generate.guid_s=0.00002
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2,3 python generate.py ${MDL} ${LEN} generate.guid_s=0.00003
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2,3 python generate.py ${MDL} ${LEN} generate.guid_s=0.00005
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2,3 python generate.py ${MDL} ${LEN} generate.guid_s=0.00007

