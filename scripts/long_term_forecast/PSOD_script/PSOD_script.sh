export CUDA_VISIBLE_DEVICES=1

model_name=TimeXer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/PSOD/ \
  --data_path ssh_psod_sync_hourly.csv \
  --model_id timemixer_psod \
  --model $model_name \
  --data custom \
  --target ssh \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --d_model 512 \
  --batch_size 32 \
  --des 'exp' \
  --itr 1

