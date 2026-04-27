export CUDA_VISIBLE_DEVICES=0

model_name=DualMAE4TS

for PRED_LEN in 96 192 336 720; do
  if [ "$PRED_LEN" -le 96 ]; then
    d_model=64
  elif [ "$PRED_LEN" -le 192 ]; then
    d_model=768
  else
    d_model=512
  fi

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /remote-home/data/ETT/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_1440_$PRED_LEN \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len 1440 \
    --label_len 48 \
    --pred_len $PRED_LEN \
    --e_layers 2 \
    --d_layers 1 \
    --d_model $d_model \
    --factor 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --learning_rate 0.000002 \
    --des 'Exp' \
    --n_heads 2 \
    --itr 1 \
    --periodicity 24 \
    --percent 100
done
