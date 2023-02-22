cd ..
python baselines.py \
  --data_split 0.8 \
  --data_shuffle \
  --saved_train_dir '../data/cdbd/train.csv' \
  --saved_dev_dir '../data/cdbd/dev.csv' \
  --saved_test_dir '../data/cdbd/test.csv' \
  --saved_item_dir '../data/cdbd/item.csv' \
  --model_path '../data/MIRT/mirt.params' \
  --models 'MIRT' \
  --mode "Middle" \
  --epoch 30 \
  --batch_size 256 \
  --logger_dir '../log/' 