for i in $(ls -d /data/ephraim/datasets/known_noise/enhanced_diffwave/*/); do python measure.py -enhanced_dir=${i}; done
