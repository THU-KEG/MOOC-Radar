cd ..
python baselines.py \
  --data_process \
  --saved_train_dir '../data/DKTplus/train.json' \
  --saved_test_dir '../data/DKTplus/test.json' \
  --model_path '../data/DKTplus/dktplus.params' \
  --models 'DKT+' \
  --num_questions 580 \
  --mode "Middle" \
  --epoch 10 \
  --batch_size 64 \
  --logger_dir '../log/' 