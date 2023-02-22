DATA_ROOT='../data'
cd ..
python baseline-cognitive.py \
  --models 'DKVMN' \
  --data_split 0.8 \
  --data_shuffle \
  --saved_train_dir $DATA_ROOT/DKVMN/train-pid.txt \
  --saved_test_dir $DATA_ROOT/DKVMN/test-pid.txt \
  --num_questions 580 \
  --num_cognitive 5 \
  --batch_size 256 \
  --epoch 50 \
  --logger_dir $DATA_ROOT/../log/