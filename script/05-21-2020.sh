#python ../tools/train.py --config_file='../config-yml/joint.yml'
#
#python ../tools/train.py --config_file='../config-yml/joint.yml' \
#LOSS.IF_LEARNING_WEIGHT = True

python ../tools/train.py --config_file='../config-yml/msmt17.yml'

python ../tools/train.py --`config_file='../config-yml/dukemtmc.yml'