# Image Captioning with Vision Encoder-Language Decoder

This repository provides the code for training a multi-modal vision-text model for image captioning.
The project aims at replicating the architecture used in [TrOCR](https://arxiv.org/abs/2109.10282).
The code follows HuggingFace's training scripts and uses their [VisionEncoderDecoderModel](https://huggingface.co/transformers/master/model_doc/visionencoderdecoder.html).

Multiple accelerator support is enabled through [accelerate](https://huggingface.co/docs/accelerate/). 

The scripts also supports:
- Comet logging
- Early stop on non improving epochs

## Getting started

Install required libraries from `requirements.txt`. 

## Usage

The following command runs training using Google's Vision Transformer as image encoder and De Mattei's [GePpeTto](https://arxiv.org/abs/2004.14253) as the text generative model. 

```bash

python run_vision_text_captioning.py \
    --output_dir dumps/ViT_geppt \
    --train_file train_dataset.jsonl \
    --validation_file valid_dataset.jsonl \
    --encoder_pretrained_model_name_or_path google/vit-base-patch16-224-in21k \
    --decoder_pretrained_model_name_or_path LorenzoDeMattei/GePpeTto \
    --feature_extractor_name google/vit-base-patch16-224-in21k \
    --tokenizer_name LorenzoDeMattei/GePpeTto \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 64 \
    --learning_rate 2e-6 \
    --num_train_epochs 3 \
    --num_warmup_steps 0.1 \
    --max_seq_length 96 \
    --line_by_line \
    --preprocessing_num_workers 8 \
    --save_last \
    --log_comet

```