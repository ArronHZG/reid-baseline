python ../tools/supervisedComponent.py --config_file='../config-yml/autoencoder.yml'

python ../tools/supervisedComponent.py --config_file='../config-yml/autoencoder.yml' \
EBLL.CODE_SIZE 512

python ../tools/supervisedComponent.py --config_file='../config-yml/autoencoder.yml' \
EBLL.LAMBDA 0.1

python ../tools/supervisedComponent.py --config_file='../config-yml/autoencoder.yml' \
EBLL.LAMBDA 0.05