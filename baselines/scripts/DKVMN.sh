# set the DKVMN.sh in line 394
cd ..
DATA_ROOT='../data'
python baselines.py \
  --models 'DKVMN' \
  --saved_train_dir $DATA_ROOT/DKVMN/train-pid6.txt \
  --saved_test_dir $DATA_ROOT/DKVMN/test-pid6.txt \
  --num_questions 443 \
  --mode "Middle" \
  --batch_size 256 \
  --epoch 50 \
  --logger_dir $DATA_ROOT/../log/