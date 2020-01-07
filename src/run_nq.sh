export CUDA_VISIBLE_DEVICES=0,1,2,3

python run_nq.py \
--model_type bert \
--model_name_or_path ./bert_base/ \
--do_train \
--data_dir ./data/fold_5/data_0/ \
--output_dir ./output_version2/ \
--max_seq_length 384 \
--eval_steps 200 \
--chunksize 1000 \
--per_gpu_train_batch_size 10 \
--per_gpu_eval_batch_size 50 \
--learning_rate 5e-6 \