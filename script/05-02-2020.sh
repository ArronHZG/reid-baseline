python ../tools/supervisedComponent.py --config_file='../config-yml/resnet50.yml'
python ../tools/supervisedComponent.py --config_file='../config-yml/resnet101.yml'
python ../tools/supervisedComponent.py --config_file='../config-yml/resnext50_32x4d.yml'
python ../tools/supervisedComponent.py --config_file='../config-yml/resnext101_32x8d.yml'
python ../tools/supervisedComponent.py --config_file='../config-yml/wide_resnet50_2.yml'
python ../tools/supervisedComponent.py --config_file='../config-yml/wide_resnet101_2.yml'

python ../tools/supervisedComponent.py --config_file='../config-yml/resnet50.yml' MODEL.IF_SE True
python ../tools/supervisedComponent.py --config_file='../config-yml/resnet101.yml' MODEL.IF_SE True
python ../tools/supervisedComponent.py --config_file='../config-yml/resnext50_32x4d.yml' MODEL.IF_SE True
python ../tools/supervisedComponent.py --config_file='../config-yml/resnext101_32x8d.yml' MODEL.IF_SE True
python ../tools/supervisedComponent.py --config_file='../config-yml/wide_resnet50_2.yml' MODEL.IF_SE True
python ../tools/supervisedComponent.py --config_file='../config-yml/wide_resnet101_2.yml' MODEL.IF_SE True

python ../tools/supervisedComponent.py --config_file='../config-yml/resnet50.yml' MODEL.IF_IBN_A True
python ../tools/supervisedComponent.py --config_file='../config-yml/resnet101.yml' MODEL.IF_IBN_A True
python ../tools/supervisedComponent.py --config_file='../config-yml/resnext50_32x4d.yml' MODEL.IF_IBN_A True
python ../tools/supervisedComponent.py --config_file='../config-yml/resnext101_32x8d.yml' MODEL.IF_IBN_A True
python ../tools/supervisedComponent.py --config_file='../config-yml/wide_resnet50_2.yml' MODEL.IF_IBN_A True
python ../tools/supervisedComponent.py --config_file='../config-yml/wide_resnet101_2.yml' MODEL.IF_IBN_A True

python ../tools/supervisedComponent.py --config_file='../config-yml/resnet50.yml' MODEL.IF_IBN_B True
python ../tools/supervisedComponent.py --config_file='../config-yml/resnet101.yml' MODEL.IF_IBN_B True
python ../tools/supervisedComponent.py --config_file='../config-yml/resnext50_32x4d.yml' MODEL.IF_IBN_B True
python ../tools/supervisedComponent.py --config_file='../config-yml/resnext101_32x8d.yml' MODEL.IF_IBN_B True
python ../tools/supervisedComponent.py --config_file='../config-yml/wide_resnet50_2.yml' MODEL.IF_IBN_B True
python ../tools/supervisedComponent.py --config_file='../config-yml/wide_resnet101_2.yml' MODEL.IF_IBN_B True

python ../tools/supervisedComponent.py --config_file='../config-yml/resnet50.yml' MODEL.IF_SE True MODEL.IF_IBN_A True
python ../tools/supervisedComponent.py --config_file='../config-yml/resnet101.yml' MODEL.IF_SE True MODEL.IF_IBN_A True
python ../tools/supervisedComponent.py --config_file='../config-yml/resnext50_32x4d.yml' MODEL.IF_SE True MODEL.IF_IBN_A True
python ../tools/supervisedComponent.py --config_file='../config-yml/resnext101_32x8d.yml' MODEL.IF_SE True MODEL.IF_IBN_A True
python ../tools/supervisedComponent.py --config_file='../config-yml/wide_resnet50_2.yml' MODEL.IF_SE True MODEL.IF_IBN_A True
python ../tools/supervisedComponent.py --config_file='../config-yml/wide_resnet101_2.yml' MODEL.IF_SE True MODEL.IF_IBN_A True

python ../tools/supervisedComponent.py --config_file='../config-yml/resnet50.yml' MODEL.IF_SE True MODEL.IF_IBN_B True
python ../tools/supervisedComponent.py --config_file='../config-yml/resnet101.yml' MODEL.IF_SE True MODEL.IF_IBN_B True
python ../tools/supervisedComponent.py --config_file='../config-yml/resnext50_32x4d.yml' MODEL.IF_SE True MODEL.IF_IBN_B True
python ../tools/supervisedComponent.py --config_file='../config-yml/resnext101_32x8d.yml' MODEL.IF_SE True MODEL.IF_IBN_B True
python ../tools/supervisedComponent.py --config_file='../config-yml/wide_resnet50_2.yml' MODEL.IF_SE True MODEL.IF_IBN_B True
python ../tools/supervisedComponent.py --config_file='../config-yml/wide_resnet101_2.yml' MODEL.IF_SE True MODEL.IF_IBN_B True


#MODEL.IF_SE True
#MODEL.IF_IBN_A True
#MODEL.IF_IBN_B True
#MODEL.IF_SE True MODEL.IF_IBN_A True
#MODEL.IF_SE True MODEL.IF_IBN_B True
