python ../tools/ebll_train.py \
CONTINUATION.LOSS_TYPE "ce_dist"

python ../tools/ebll_train.py \
CONTINUATION.LOSS_TYPE "tr_dist"

python ../tools/ebll_train.py \
CONTINUATION.LOSS_TYPE "ce_dist tr_dist"