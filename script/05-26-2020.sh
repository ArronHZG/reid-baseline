#python ../tools/supervisedComponent.py --config_file='../config-yml/dukemtmc.yml'
#
#python ../tools/supervisedComponent.py --config_file='../config-yml/dukemtmc.yml' \
#MODEL.IF_IBN_A = False

#python ../tools/supervisedComponent.py --config_file='../config-yml/joint.yml'

#python ../tools/continuous_train.py --config_file='../config-yml/continue.yml' \
#CONTINUATION.LOSS_TYPE "ce_dist"
#
#python ../tools/continuous_train.py --config_file='../config-yml/continue.yml' \
#CONTINUATION.LOSS_TYPE "tr_dist"
#
#python ../tools/continuous_train.py --config_file='../config-yml/continue.yml' \
#CONTINUATION.LOSS_TYPE "ce_dist tr_dist"