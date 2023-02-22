DATA_ROOT='../data'
cd ..
python baseline-video.py \
  --data_split 0.8 \
  --data_shuffle \
  --saved_train_dir $DATA_ROOT/cdbd2/train.csv \
  --saved_dev_dir $DATA_ROOT/cdbd2/dev.csv \
  --saved_test_dir $DATA_ROOT/cdbd2/test.csv \
  --saved_item_dir $DATA_ROOT/cdbd2/item.csv \
  --saved_cog_dir $DATA_ROOT/cdbd2/video.csv \
  --model_path $DATA_ROOT/NCDM/ncdm.snapshot \
  --models 'NCDM' \
  --epoch 30 \
  --batch_size 256 \
  --logger_dir $DATA_ROOT/../log/