python ../tools/train.py --config_file='../configs/joint.yml' \
MODEL.IF_IBN_A True \
TRAIN.MAX_EPOCHS 300 \
MODEL.DEVICE_ID  1

python ../tools/train.py --config_file='../configs/joint.yml' \
MODEL.IF_IBN_A True \
TRAIN.MAX_EPOCHS 300 \
LOSS.IF_LEARNING_WEIGHT = True \
MODEL.DEVICE_ID  1


python ../tools/train.py --config_file='../configs/msmt17.yml' \
MODEL.IF_IBN_A True \
TRAIN.MAX_EPOCHS 300 \
MODEL.DEVICE_ID  1


python ../tools/train.py --config_file='../configs/dukemtmc.yml' \
MODEL.IF_IBN_A True \
TRAIN.MAX_EPOCHS 300 \
MODEL.DEVICE_ID  1
