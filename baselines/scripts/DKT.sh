cd ..
python baselines.py \
  --data_dir '../data/student-problem-new-1-2exercise.json' \
  --data_process \
  --data_split 0.8 \
  --data_shuffle \
  --saved_train_dir '../data/DKT/train2.txt' \
  --saved_test_dir '../data/DKT/test2.txt' \
  --encoded_train_dir '../data/DKT/train_data2.npy' \
  --encoded_test_dir '../data/DKT/test_data2.npy' \
  --model_path '../data/DKT/dkt2.params' \
  --models 'DKT' \
  --max_step 50 \
  --num_questions 580 \
  --mode "Middle" \
  --epoch 25 \
  --batch_size 256 \
  --logger_dir '../log/' 
