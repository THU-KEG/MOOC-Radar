DATA_ROOT='../data'
cd ..
python baseline-video.py \
  --models 'DKVMN' \
  --data_split 0.8 \
  --data_shuffle \
  --saved_train_dir $DATA_ROOT/DKVMN/train-pid6.txt \
  --saved_test_dir $DATA_ROOT/DKVMN/test-pid6.txt \
  --num_questions 443 \
  --num_cognitive 1486 \
  --batch_size 256 \
  --seqlen 50 \
  --epoch 50 \
  --logger_dir $DATA_ROOT/../log/