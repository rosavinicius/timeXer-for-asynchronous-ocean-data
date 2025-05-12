export CUDA_VISIBLE_DEVICES=1

model_name=TimeXer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/PSOD/ \
  --data_path  multivariate_psod_full_sync_hourly.csv \
  --model_id timemixer_psod_multivariate_full \
  --model $model_name \
  --data custom \
  --target ssh_prat_ssh \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 14 \
  --dec_in 14 \
  --c_out 14 \
  --d_model 512 \
  --batch_size 32 \
  --des 'exp' \
  --itr 1

