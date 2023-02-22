cd ..
python baselines.py \
  --data_dir '../data/student-problem-new-1.json' \
  --data_process \
  --data_split 0.8 \
  --data_shuffle \
  --saved_train_dir '../data/GDIRT/train.csv' \
  --saved_dev_dir '../data/GDIRT/dev.csv' \
  --saved_test_dir '../data/GDIRT/test.csv' \
  --saved_item_dir '../data/GDIRT/item.csv' \
  --model_path '../data/GDIRT/irt.params' \
  --models 'GDIRT' \
  --mode "Middle" \
  --epoch 40 \
  --batch_size 256 \
  --logger_dir '../log/' 