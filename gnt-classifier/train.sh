#!/usr/bin/zsh
#SBATCH --job-name=gnt_cls            # Job name
#SBATCH --output=logs/%A.out             # Output file
#SBATCH --partition=boost_usr_prod                  # Specify the partition name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks (processes) per node
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1                # Number of tasks (processes) per node
#SBATCH --time=06:00:00                   # Walltime limit (hh:mm:ss)

export TOKENIZERS_PARALLELISM=false

#module load cuda
source ~/.zshrc
source $FAST/py310/bin/activate
export HF_HOME=/leonardo_scratch/large/userexternal/gattanas/huggingface

export WANDB_MODE=offline
export WANDB_PROJECT="gnt-classifier"

# Inspiration from: https://github.com/huggingface/blog/blob/main/Lora-for-sequence-classification-with-Roberta-Llama-Mistral.md
    # --model_name_or_path "mistralai/Mistral-7B-v0.1" \
srun python train.py \
    --model_name_or_path "Musixmatch/umberto-commoncrawl-cased-v1" \
    --output_dir $FAST/gnt-classifier-umberto-reproduce-v3 \
    --train_file ./data/train_L2.tsv \
    --validation_file ./data/dev_L2.tsv \
    --dataloader_num_workers 8 \
    --shuffle_train_dataset \
    --metric_name accuracy \
    --text_column_name SENTENCE \
    --label_column_name LABEL \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 64 \
    --learning_rate 5e-5 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.1 \
    --report_to "all" \
    --save_total_limit 2 \
    --num_train_epochs 1 \
    --logging_steps 50 \
    --evaluation_strategy "steps" \
    --eval_steps 200    \
    --save_strategy "steps" \
    --save_steps 200

    #--load_best_model_at_end \
    #--max_grad_norm 1.0 \
    # --bf16 \
    #--gradient_checkpointing \
    #--push_to_hub \
    #--hub_model_id "g8a9/gnt-classifier" \
    #--hub_strategy "end" \
    ##--hub_private_repo

    #--per_device_train_batch_size 2 \
