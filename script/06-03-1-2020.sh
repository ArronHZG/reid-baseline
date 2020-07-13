python ../tools/ebll_train.py \
CONTINUATION.LOSS_TYPE "ae_dict ce_dist" \
MODEL.DEVICE_ID 0

python ../tools/ebll_train.py \
CONTINUATION.LOSS_TYPE "ce_dist" \
MODEL.DEVICE_ID 0