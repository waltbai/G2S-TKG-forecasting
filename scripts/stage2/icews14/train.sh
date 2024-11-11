CUDA_VISIBLE_DEVICES=0,1 torchrun \
	--standalone \
	--nnodes=1 \
	--nproc-per-node=2 \
	-m src.stage2.train config/stage2/ICEWS14/train.yaml