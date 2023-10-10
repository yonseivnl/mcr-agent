export ALFRED_ROOT=$(pwd)
CUDA_VISIBLE_DEVICES=9 python models/utils/extract_resnet.py	\
	--batch 32	\
	--filename feat_conv_onlyAutoAug4_panoramic.pt	\
	--randomization auto_aug
