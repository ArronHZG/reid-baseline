python ../tools/continuous_train.py --config_file='../config-yml/EBLL.yml' \
EBLL.LOSS_TYPE "ae_loss" \
EBLL.DIST_TYPE "mse"

python ../tools/continuous_train.py --config_file='../config-yml/EBLL.yml' \
EBLL.LOSS_TYPE "ae_l1" \
EBLL.DIST_TYPE "mse"


python ../tools/continuous_train.py --config_file='../config-yml/EBLL.yml' \
EBLL.LOSS_TYPE "ae_l2" \
EBLL.DIST_TYPE "mse"

python ../tools/continuous_train.py --config_file='../config-yml/EBLL.yml' \
EBLL.LOSS_TYPE "ae_loss ae_l1 ae_l2" \
EBLL.DIST_TYPE "mse"