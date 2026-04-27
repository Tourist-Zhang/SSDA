export CUDA_VISIBLE_DEVICES=0

model_name=DualMAE4TS

for PRED_LEN in 96 192 336 720; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /remote-home/data/ETT/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_1440_$PRED_LEN \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --seq_len 1440 \
    --label_len 48 \
    --pred_len $PRED_LEN \
    --e_layers 2 \
    --d_layers 1 \
    --factor 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --learning_rate 0.000002 \
    --patience 3 \
    --des 'Exp' \
    --n_heads 2 \
    --itr 1 \
    --periodicity 96 \
    --percent 100
done

