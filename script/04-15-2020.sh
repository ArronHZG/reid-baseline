python ../tools/train.py --config_file='../configs/arcface.yml'
python ../tools/train.py --config_file='../configs/apex.yml' LOSS.METRIC_LOSS_WEIGHT 2.0
python ../tools/continuous_train.py --config_file='../configs/continue.yml' CONTINUATION.T 0.05
python ../tools/continuous_train.py --config_file='../configs/continual_learning1.yml' CONTINUATION.T 0.05
python ../tools/continuous_train.py --config_file='../configs/continue.yml' CONTINUATION.T 0.05 OPTIMIZER.BASE_LR 0.000035
python ../tools/continuous_train.py --config_file='../configs/continual_learning1.yml' CONTINUATION.T 0.05 OPTIMIZER.BASE_LR 0.000035
python ../tools/continuous_train.py --config_file='../configs/continue.yml' CONTINUATION.T 0.05 LOSS.LOSS_TYPE = 'softmax_arcface_triplet'
python ../tools/continuous_train.py --config_file='../configs/continue.yml' CONTINUATION.T 0.06
python ../tools/continuous_train.py --config_file='../configs/continue.yml' CONTINUATION.T 0.07
python ../tools/continuous_train.py --config_file='../configs/continue.yml' CONTINUATION.T 0.08
python ../tools/continuous_train.py --config_file='../configs/continue.yml' CONTINUATION.T 0.09
python  ../tools/train.py --config_file='../configs/ranger.yml'