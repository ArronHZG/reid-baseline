python ../tools/ebll_train.py \
CONTINUATION.LOSS_TYPE "ae_dict tr_dist" \
MODEL.DEVICE_ID 1

python ../tools/ebll_train.py \
CONTINUATION.LOSS_TYPE "tr_dist" \
MODEL.DEVICE_ID 1