#python ../tools/continuous_train.py --config_file='../configs/continue.yml' CONTINUATION.T 0.05
#python ../tools/train.py --config_file='../configs/sgd.yml'
python ../tools/continuous_train.py --config_file='../configs/continue.yml' CONTINUATION.LOSS_TYPE "ce_dist"

python ../tools/continuous_train.py --config_file='../configs/continue.yml' \
CONTINUATION.LOSS_TYPE "ce_dist" \
LOSS.ID_LOSS_WEIGHT 0.2

python ../tools/continuous_train.py --config_file='../configs/continue.yml' \
CONTINUATION.LOSS_TYPE "tr_dist"

python ../tools/continuous_train.py --config_file='../configs/continue.yml' \
CONTINUATION.LOSS_TYPE "ce_dist tr_dist"

python ../tools/continuous_train.py --config_file='../configs/continue.yml' \
CONTINUATION.LOSS_TYPE "tr_dist" \
TEST.IF_CLASSIFT_FEATURE False \
LOSS.LOSS_TYPE 'triplet' \
TRAIN.MAX_EPOCHS 300