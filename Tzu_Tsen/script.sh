python train.py \
  --seed 42 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note 0.3weightforOARSIloss

python inference.py \
  --current_ckpt demo_ckpt \
  --seed 42 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note demo


python inference.py \
  --current_ckpt model_checkpoints_20260127_004207_epoch200_MIL_MultiTask_imedslab_LCEoM_MA_C0_Fo_lr1e-04_b16 \
  --seed 42 \
  --inference_targe avg \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note demo


python inference.py \
  --current_ckpt model_checkpoints_20260127_033448_epoch200_MIL_MultiTask_imedslab_LCEoM_MA_C0_Fo_lr1e-04_b16 \
  --seed 42 \
  --inference_targe avg \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note demo

python inference.py \
  --current_ckpt model_checkpoints_20260128_042926_epoch200_MIL_MultiTask_imedslab_LCEoM_MA_C0_Fo_lr1e-04_b16 \
  --seed 42 \
  --inference_targe avg \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note demo

python train.py \
  --seed 42 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train_wMask_0.2weightforOARSIloss

python inference.py \
  --current_ckpt model_checkpoints_20260129_000705_epoch200_MIL_MultiTask_imedslab_LCEoM_MA_C0_Fo_lr1e-04_b16 \
  --seed 42 \
  --inference_targe avg \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note demo

python train.py \
  --seed 42 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train_wMask



python inference.py \
  --current_ckpt model_checkpoints_tnc_final \
  --seed 42 \
  --inference_targe kl \
  --model_type MIL_ORG \
  --lossfcn_type CrossEntropy \
  --predict_criteria Max \
  --classweight_type inv \
  --multitask_type off \
  --feedback_type off \
  --feedback_cam off \
  --note demo

python inference_img.py \
  --current_ckpt model_checkpoints_tnc_final \
  --seed 42 \
  --inference_targe kl \
  --model_type MIL_ORG \
  --lossfcn_type CrossEntropy \
  --predict_criteria Max \
  --classweight_type inv \
  --multitask_type off \
  --feedback_type off \
  --feedback_cam off \
  --note demo


python inference_img.py \
  --current_ckpt model_checkpoints_20260127_041244_epoch200_MIL_MultiTask_imedslab_LCEoM_MA_C0_Fo_lr1e-04_b16_GOOD \
  --seed 42 \
  --inference_targe avg \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note demo

python train.py \
  --seed 42 \
  --inference_targe kl \
  --model_type MIL \
  --lossfcn_type CrossEntropy \
  --predict_criteria Max \
  --classweight_type inv \
  --multitask_type off \
  --feedback_type off \
  --feedback_cam off \
  --note onlyKL

python inference_img.py \
  --current_ckpt model_checkpoints_20260129_032606_epoch200_MIL_LCE_M0_C0_Fo_lr1e-04_b16 \
  --seed 42 \
  --inference_targe kl \
  --model_type MIL \
  --lossfcn_type CrossEntropy \
  --predict_criteria Max \
  --classweight_type inv \
  --multitask_type off \
  --feedback_type off \
  --feedback_cam off \
  --note demo


python train_k_fold.py \
  --seed 42 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note k_fold_42


python inference.py \
  --current_ckpt model_checkpoints_20260129_030919_epoch200_MIL_MultiTask_imedslab_LCEoM_MA_C0_Fo_lr1e-04_b16\
  --seed 42 \
  --inference_targe avg \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note demo

python train.py \
  --seed 42 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note train_0.2weightforOARSIloss


python inference.py \
  --current_ckpt model_checkpoints_20260129_082114_epoch200_MIL_MultiTask_imedslab_LCEoM_MA_C0_Fo_lr1e-04_b16\
  --seed 42 \
  --inference_targe avg \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note 
  
python train.py \
  --seed 42 \
  --inference_targe kl \
  --model_type MIL \
  --lossfcn_type CrossEntropy \
  --predict_criteria Max \
  --classweight_type inv \
  --multitask_type off \
  --feedback_type off \
  --feedback_cam off \
  --note aggregation_type_mean



python inference_k_fold.py \
  --current_ckpt model_checkpoints_20260129_042950_epoch200_MIL_MultiTask_imedslab_LCEoM_MA_C0_Fo_lr1e-04_b16 \
  --seed 42 \
  --model_type MIL_MultiTask_imedslab \
  --lossfcn_type CrossEntropy_MultiTask \
  --predict_criteria Max_Multitask \
  --classweight_type all_metrics_inv \
  --multitask_type all \
  --feedback_type off \
  --feedback_cam off \
  --note k_fold_42