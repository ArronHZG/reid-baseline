#python ../tools/train.py --config_file='../configs/joint.yml'
#
#python ../tools/train.py --config_file='../configs/joint.yml' \
#LOSS.IF_LEARNING_WEIGHT = True

python ../tools/train.py --config_file='../configs/msmt17.yml'

python ../tools/train.py --`config_file='../configs/dukemtmc.yml'