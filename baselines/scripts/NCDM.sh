cd ..
python baselines.py \
  --data_dir '../data/student-problem-new-1.json' \
  --data_split 0.8 \
  --data_shuffle \
  --saved_train_dir '../data/cdbd2/train.csv' \
  --saved_dev_dir '../data/cdbd2/dev.csv' \
  --saved_test_dir '../data/cdbd2/test.csv' \
  --saved_item_dir '../data/cdbd2/item.csv' \
  --model_path '../data/NCDM/ncdm.snapshot' \
  --models 'NCDM' \
  --mode "Middle" \
  --epoch 30 \
  --batch_size 256 \
  --logger_dir '../log/' 