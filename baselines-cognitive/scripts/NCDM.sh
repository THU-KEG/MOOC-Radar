DATA_ROOT='../data'
cd ..
python baseline-cognitive.py \
  --data_dir $DATA_ROOT/student-problem-new-1.json \
  --data_split 0.8 \
  --data_shuffle \
  --saved_train_dir $DATA_ROOT/cdbd/train.csv \
  --saved_dev_dir $DATA_ROOT/cdbd/dev.csv \
  --saved_test_dir $DATA_ROOT/cdbd/test.csv \
  --saved_item_dir $DATA_ROOT/cdbd/item.csv \
  --saved_cog_dir $DATA_ROOT/cdbd/cog.csv \
  --model_path $DATA_ROOT/NCDM/ncdm.snapshot \
  --models 'NCDM' \
  --epoch 20 \
  --batch_size 256 \
  --logger_dir $DATA_ROOT/../log/