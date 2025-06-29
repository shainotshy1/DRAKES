if [ "$#" -lt 5 ]; then
  echo "Usage: $0 <BASE_MODEL=new/old> <ALIGN_TYPE=bon/beam/spectral> <ALIGN_ORACLE> <N_ALIGN> <GPU>"
  exit 1
fi

BASE_MODEL=$1
ALIGN_TYPE=$2
ALIGN_ORACLE=$3
N_ALIGN=$4
GPU=$5

if [ "$#" -eq 6 ]; then
  ALPHA=$6
else
  ALPHA=0.01
fi

for seed in 0
do
    echo "Running Ours... with seed=$seed"
    python eval_finetune.py --decoding=original --base_model=$BASE_MODEL --seed=$seed --base_path=../../data/data_and_model --align_type $ALIGN_TYPE --align_oracle $ALIGN_ORACLE --n_align $N_ALIGN --gpu $GPU --balanced_alpha $ALPHA
done