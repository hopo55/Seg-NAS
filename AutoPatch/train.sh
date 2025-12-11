python search.py --accelerator auto \
  --study_name 2023_03_11_13_00_00_n2000_k1_s0_carpet_False \
  --n_trials 2000 \
  --k 1 \
  --seed 0 \
  --category carpet \
  --test_set_search False \
  --dataset_dir "../MVTec" \
  --db_url sqlite:///studies.db
