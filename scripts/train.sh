cfg='/code/GALIP-main/code/cfg/birds.yml'
batch_size=48

state_epoch=1
pretrained_model_path='/data/wym123/24RSP_GALIP/selfckptdir/saved_models/coco/GALIP_nf64_gpuMP_True_coco_256_2024_07_10_10_15_05/'
log_dir='new'

multi_gpus=False
mixed_precision=True

num_workers=8
stamp=gpu${nodes}MP_${mixed_precision}

python -m torch.distributed.launch /code/GALIP-main/code/src/train.py \
                    --stamp $stamp \
                    --cfg $cfg \
                    --mixed_precision $mixed_precision \
                    --log_dir $log_dir \
                    --batch_size $batch_size \
                    --state_epoch $state_epoch \
                    --num_workers $num_workers \
                    --multi_gpus $multi_gpus \
                    --pretrained_model_path $pretrained_model_path \
