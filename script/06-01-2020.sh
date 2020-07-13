python ../tools/ebll_train.py \
CONTINUATION.LOSS_TYPE "ae_dict ce_dist" \
MODEL.DEVICE_ID 0

python ../tools/ebll_train.py \
CONTINUATION.LOSS_TYPE "ce_dist" \
MODEL.DEVICE_ID 0

python ../tools/ebll_train.py \
CONTINUATION.LOSS_TYPE "ae_dict tr_dist" \
MODEL.DEVICE_ID 1

python ../tools/ebll_train.py \
CONTINUATION.LOSS_TYPE "tr_dist" \
MODEL.DEVICE_ID 1

python ../tools/ebll_train.py \
CONTINUATION.LOSS_TYPE "ae_dict ce_dist tr_dist" \
MODEL.DEVICE_ID 2

python ../tools/ebll_train.py \
CONTINUATION.LOSS_TYPE "ce_dist tr_dist" \
MODEL.DEVICE_ID 2