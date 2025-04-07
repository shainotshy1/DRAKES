for seed in 0
do
    echo "Running Ours... with seed=$seed"
    CUDA_VISIBLE_DEVICES=1 python eval_finetune.py --decoding=original --base_model=old --seed=$seed --base_path=../../data/data_and_model
done
