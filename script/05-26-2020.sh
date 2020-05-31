#python ../tools/train.py --config_file='../configs/dukemtmc.yml'
#
#python ../tools/train.py --config_file='../configs/dukemtmc.yml' \
#MODEL.IF_IBN_A = False

#python ../tools/train.py --config_file='../configs/joint.yml'

#python ../tools/continuous_train.py --config_file='../configs/continue.yml' \
#CONTINUATION.LOSS_TYPE "ce_dist"
#
#python ../tools/continuous_train.py --config_file='../configs/continue.yml' \
#CONTINUATION.LOSS_TYPE "tr_dist"
#
#python ../tools/continuous_train.py --config_file='../configs/continue.yml' \
#CONTINUATION.LOSS_TYPE "ce_dist tr_dist"