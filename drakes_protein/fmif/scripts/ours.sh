for seed in 0
do
    echo "Running Ours... with seed=$seed"
    CUDA_VISIBLE_DEVICES=0 python eval_finetune.py --decoding=original --base_model=new --seed=$seed --base_path=../../data/data_and_model
done
