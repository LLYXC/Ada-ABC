# SbP90
python abc.py --dataset_tag Source_pneumonia_bias90 --exp_name LAD_SbP90_check --model_tag DenseNet121 --lamda 5 --xrv_weight --seed 1 --device 5 --main_batch_size 8 --main_num_epochs 50 --num_heads 2 --proportion 0.8 --bias_mlp_classifier 

# SbP95 
# python abc.py --dataset_tag Source_pneumonia_bias95 --exp_name LAD_SbP95_check --model_tag DenseNet121 --lamda 10 --xrv_weight --seed 1 --device 5 --main_batch_size 4 --main_num_epochs 50 --num_heads 64 --proportion 0.8 --bias_mlp_classifier 

# SbP99
# python abc.py --dataset_tag Source_pneumonia_bias99 --exp_name LAD_SbP99_check --model_tag DenseNet121 --lamda 100 --xrv_weight --seed 1 --device 5 --main_batch_size 8 --main_num_epochs 50 --num_heads 16 --proportion 0.8 --bias_mlp_classifier 

# DbP
# python abc.py --dataset_tag Drain_pneumothorax_case1 --exp_name LAD_DbP_check --model_tag DenseNet121 --lamda 1 --xrv_weight --seed 1 --device 5 --main_batch_size 8 --main_num_epochs 50 --num_heads 4 --proportion 0.8 --bias_mlp_classifier 

# GbP1
# python abc.py --dataset_tag Gender_pneumothorax_case1 --exp_name LAD_GbP1_check --model_tag DenseNet121 --lamda 1 --xrv_weight --seed 1 --device 4 --main_batch_size 4 --main_num_epochs 50 --num_heads 128 --proportion 0.8 --bias_mlp_classifier 

# GbP2
# python abc.py --dataset_tag Gender_pneumothorax_case2 --exp_name LAD_GbP2_check --model_tag DenseNet121 --lamda 0.1 --xrv_weight --seed 1 --device 4 --main_batch_size 4 --main_num_epochs 50 --num_heads 64 --proportion 0.8 --bias_mlp_classifier 

# OL3I
# python abc.py --dataset_tag Age_heart_disease --exp_name LAD_OL3I_check --model_tag ResNet18 --lamda 300 --imagenet_weight --seed 1 --device 5 --main_batch_size 16 --main_num_epochs 50 --num_heads 8 --proportion 0.8 --bias_mlp_classifier --debias_mlp_classifier
