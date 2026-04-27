export CUDA_VISIBLE_DEVICES=0

model_name=DualMAE4TS

for PRED_LEN in 96 192 336 720; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /remote-home/data/weather/ \
    --data_path weather.csv \
    --model_id weather_2160_$PRED_LEN \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 2160 \
    --label_len 48 \
    --pred_len $PRED_LEN \
    --e_layers 2 \
    --d_layers 1 \
    --factor 1 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --learning_rate 0.000003 \
    --patience 2 \
    --des 'Exp' \
    --n_heads 2 \
    --itr 1 \
    --periodicity 144 \
    --percent 10
done


for PRED_LEN in 96 192 336 720; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /remote-home/data/weather/ \
    --data_path weather.csv \
    --model_id weather_2160_$PRED_LEN \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 2160 \
    --label_len 48 \
    --pred_len $PRED_LEN \
    --e_layers 2 \
    --d_layers 1 \
    --factor 1 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --learning_rate 0.000003 \
    --patience 2 \
    --des 'Exp' \
    --n_heads 2 \
    --itr 1 \
    --periodicity 144 \
    --percent 5
done