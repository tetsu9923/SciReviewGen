CUDA_VISIBLE_DEVICES=0 \
    python run_summarization.py \
    --model_name_or_path facebook/bart-large-cnn \
    --do_train \
    --do_predict \
    --train_file ../dataset/train_qfid.csv \
    --validation_file ../dataset/val_qfid.csv \
    --test_file ../dataset/test_qfid.csv \
    --text_column reference \
    --summary_column target \
    --output_dir ./test \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --num_train_epochs=1 \
    --predict_with_generate \
    --save_strategy epoch \
    --learning_rate 5e-05 \
    --evaluation_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model=eval_rouge2 \
    --greater_is_better True \
    --save_total_limit=1 \
    --max_source_length 2048 \
    --max_target_length 512 \
    --generation_max_length 256 \
    --num_beams 4 \
