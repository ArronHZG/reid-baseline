#python ../tools/train.py --config_file='../configs/autoencoder.yml'

python ../tools/continuous_train.py --config_file='../configs/EBLL.yml' \
EBLL.LOSS_TYPE "ae_loss ae_l1 ae_l2" \
EBLL.DIST_TYPE "mse"

python ../tools/continuous_train.py --config_file='../configs/continue.yml' \
CONTINUATION.LOSS_TYPE "ce_dist"

python ../tools/continuous_train.py --config_file='../configs/continue.yml' \
CONTINUATION.LOSS_TYPE "tr_dist"

python ../tools/continuous_train.py --config_file='../configs/continue.yml' \
CONTINUATION.LOSS_TYPE "ce_dist tr_dist"