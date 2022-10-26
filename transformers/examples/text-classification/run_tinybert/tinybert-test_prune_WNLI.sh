export GLUE_DIR=data/glue
export TASK_NAME=WNLI
export PENALTY=penalty_tinybert-3
#mv output_distilbert_128/$TASK_NAME/pretrain/checkpoint-10/pytorch_model.bin output_distilbert_128/$TASK_NAME/pretrain/
# mv output_distilbert_128/WN$TASK_NAMELI/pretrain/checkpoint-10/config.json output_distilbert_128/$TASK_NAME/pretrain/
CUDA_VISIBLE_DEVICES=3 python run_glue.py \
  --model_name_or_path /data/ZLKong/tinybert/finetune/WNLI \
  --task_name $TASK_NAME \
  --do_train \
  --rew \
  --penalty_config_file ${PENALTY} \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 5.0 \
  --evaluate_during_training \
  --block_row_division 3 \
  --block_row_width 8 \
  --output_dir /data/ZLKong/tinybert/prune/WNLI \
  --overwrite_output_dir \
  --logging_steps 100 \
  --logging_dir /data/ZLKong/tinybert/prune/WNLI \
  --save_steps 400