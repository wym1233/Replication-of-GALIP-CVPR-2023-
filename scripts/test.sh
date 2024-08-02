cfg='/code/GALIP-main/code/cfg/birds.yml'
batch_size=64

pretrained_model='/data/wym123/24RSP_GALIP/pretrainckpts/pre_coco.pth'
multi_gpus=True
mixed_precision=True

nodes=1
num_workers=8
master_port=11277
stamp=gpu${nodes}MP_${mixed_precision}

python -m torch.distributed.launch /code/GALIP-main/code/src/test.py \
                    --stamp $stamp \
                    --cfg $cfg \
                    --mixed_precision $mixed_precision \
                    --batch_size $batch_size \
                    --num_workers $num_workers \
                    --multi_gpus $multi_gpus \
                    --pretrained_model_path $pretrained_model \