#!/usr/bin/zsh
#SBATCH --job-name=gnt_cls            # Job name
#SBATCH --output=logs/%A.out             # Output file
#SBATCH --partition=boost_usr_prod                  # Specify the partition name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks (processes) per node
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1                # Number of tasks (processes) per node
#SBATCH --time=00:20:00                   # Walltime limit (hh:mm:ss)

export TOKENIZERS_PARALLELISM=false

#module load cuda
source ~/.zshrc
source $FAST/lm_eval/bin/activate

export WANDB_MODE=offline
export WANDB_PROJECT="gnt-classifier"

# Inspiration from: https://github.com/huggingface/blog/blob/main/Lora-for-sequence-classification-with-Roberta-Llama-Mistral.md
python test.py \
    --model_name_or_path ./checkpoint_2
    #--model_name_or_path ./classifier_v2_ep1
    #--model_name_or_path /leonardo_scratch/fast/IscrC_ItaLLM_0/gnt-classifier-umberto-reproduce-v2
    #--model_name_or_path /leonardo_scratch/fast/IscrC_ItaLLM_0/gnt-classifier-umberto-reproduce-v2
