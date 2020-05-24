python ../tools/train.py --config_file='../configs/autoencoder.yml'

python ../tools/train.py --config_file='../configs/autoencoder.yml' \
EBLL.CODE_SIZE 512

python ../tools/train.py --config_file='../configs/autoencoder.yml' \
EBLL.LAMBDA 0.1

python ../tools/train.py --config_file='../configs/autoencoder.yml' \
EBLL.LAMBDA 0.05