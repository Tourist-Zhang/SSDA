export CUDA_VISIBLE_DEVICES=0

model_name=DualMAE4TS

for PRED_LEN in 96 192 336 720; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /remote-home/data/electricity/ \
    --data_path electricity.csv \
    --model_id electricity_2304_$PRED_LEN \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 2304 \
    --label_len 48 \
    --pred_len $PRED_LEN \
    --e_layers 2 \
    --d_layers 1 \
    --factor 1 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --learning_rate 0.000005 \
    --patience 2 \
    --train_epochs 3 \
    --des 'Exp' \
    --n_heads 2 \
    --itr 1 \
    --periodicity 96 \
    --percent 100
done
