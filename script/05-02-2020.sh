python ../tools/train.py --config_file='../configs/resnet50.yml'
python ../tools/train.py --config_file='../configs/resnet101.yml'
python ../tools/train.py --config_file='../configs/resnext50_32x4d.yml'
python ../tools/train.py --config_file='../configs/resnext101_32x8d.yml'
python ../tools/train.py --config_file='../configs/wide_resnet50_2.yml'
python ../tools/train.py --config_file='../configs/wide_resnet101_2.yml'

python ../tools/train.py --config_file='../configs/resnet50.yml' MODEL.IF_SE True
python ../tools/train.py --config_file='../configs/resnet101.yml' MODEL.IF_SE True
python ../tools/train.py --config_file='../configs/resnext50_32x4d.yml' MODEL.IF_SE True
python ../tools/train.py --config_file='../configs/resnext101_32x8d.yml' MODEL.IF_SE True
python ../tools/train.py --config_file='../configs/wide_resnet50_2.yml' MODEL.IF_SE True
python ../tools/train.py --config_file='../configs/wide_resnet101_2.yml' MODEL.IF_SE True

python ../tools/train.py --config_file='../configs/resnet50.yml' MODEL.IF_IBN_A True
python ../tools/train.py --config_file='../configs/resnet101.yml' MODEL.IF_IBN_A True
python ../tools/train.py --config_file='../configs/resnext50_32x4d.yml' MODEL.IF_IBN_A True
python ../tools/train.py --config_file='../configs/resnext101_32x8d.yml' MODEL.IF_IBN_A True
python ../tools/train.py --config_file='../configs/wide_resnet50_2.yml' MODEL.IF_IBN_A True
python ../tools/train.py --config_file='../configs/wide_resnet101_2.yml' MODEL.IF_IBN_A True

python ../tools/train.py --config_file='../configs/resnet50.yml' MODEL.IF_IBN_B True
python ../tools/train.py --config_file='../configs/resnet101.yml' MODEL.IF_IBN_B True
python ../tools/train.py --config_file='../configs/resnext50_32x4d.yml' MODEL.IF_IBN_B True
python ../tools/train.py --config_file='../configs/resnext101_32x8d.yml' MODEL.IF_IBN_B True
python ../tools/train.py --config_file='../configs/wide_resnet50_2.yml' MODEL.IF_IBN_B True
python ../tools/train.py --config_file='../configs/wide_resnet101_2.yml' MODEL.IF_IBN_B True

python ../tools/train.py --config_file='../configs/resnet50.yml' MODEL.IF_SE True MODEL.IF_IBN_A True
python ../tools/train.py --config_file='../configs/resnet101.yml' MODEL.IF_SE True MODEL.IF_IBN_A True
python ../tools/train.py --config_file='../configs/resnext50_32x4d.yml' MODEL.IF_SE True MODEL.IF_IBN_A True
python ../tools/train.py --config_file='../configs/resnext101_32x8d.yml' MODEL.IF_SE True MODEL.IF_IBN_A True
python ../tools/train.py --config_file='../configs/wide_resnet50_2.yml' MODEL.IF_SE True MODEL.IF_IBN_A True
python ../tools/train.py --config_file='../configs/wide_resnet101_2.yml' MODEL.IF_SE True MODEL.IF_IBN_A True

python ../tools/train.py --config_file='../configs/resnet50.yml' MODEL.IF_SE True MODEL.IF_IBN_B True
python ../tools/train.py --config_file='../configs/resnet101.yml' MODEL.IF_SE True MODEL.IF_IBN_B True
python ../tools/train.py --config_file='../configs/resnext50_32x4d.yml' MODEL.IF_SE True MODEL.IF_IBN_B True
python ../tools/train.py --config_file='../configs/resnext101_32x8d.yml' MODEL.IF_SE True MODEL.IF_IBN_B True
python ../tools/train.py --config_file='../configs/wide_resnet50_2.yml' MODEL.IF_SE True MODEL.IF_IBN_B True
python ../tools/train.py --config_file='../configs/wide_resnet101_2.yml' MODEL.IF_SE True MODEL.IF_IBN_B True


#MODEL.IF_SE True
#MODEL.IF_IBN_A True
#MODEL.IF_IBN_B True
#MODEL.IF_SE True MODEL.IF_IBN_A True
#MODEL.IF_SE True MODEL.IF_IBN_B True
