#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a Vision Encoder-Image Decoeer model using two HuggingFace model's.
"""

import argparse
import logging
import math
import os
import random
from pathlib import Path
import IPython
import pdb
import pandas as pd
import comet_ml
from comet_ml import Experiment
from PIL import Image
import datasets
from datasets import load_dataset, Dataset
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torchvision.io as io
import numpy as np
import transformers
from accelerate import Accelerator, DistributedType
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoFeatureExtractor,
    AutoTokenizer,
    VisionEncoderDecoderModel,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)
require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a Masked Language Modeling task"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        action="append",
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default=None,
    )
    parser.add_argument(
        "--encoder_pretrained_model_name_or_path", type=str, help="", required=True
    )
    parser.add_argument(
        "--decoder_pretrained_model_name_or_path", type=str, help="", required=True
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--feature_extractor_name",
        type=str,
        default=None,
        help="Pretrained feature extractor name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=float,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=2,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        type=bool,
        default=False,
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_token", type=str, help="The token to use to push to the Model Hub."
    )
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--log_comet", action="store_true")
    parser.add_argument("--comet_tags", nargs="*")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--save_last", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--early_stop_patience", type=int, default=0)
    args = parser.parse_args()

    # Sanity checks
    if (
        args.dataset_name is None
        and args.train_file is None
        and args.validation_file is None
    ):
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
                "jsonl",
                "txt",
            ], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:

            for file in args.validation_file:
                extension = file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "jsonl",
                    "txt",
                ], "`validation_file` should be a csv, json or txt file."

    if args.push_to_hub:
        assert (
            args.output_dir is not None
        ), "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()

    if args.log_comet:
        # Create an experiment with your api key
        experiment = Experiment(
            api_key=os.getenv("COMET_API_KEY"),
            project_name="vision-text-image-captioning",
            workspace="g8a9",
        )
        experiment.log_parameters(args)
        if args.comet_tags:
            experiment.add_tags(args.comet_tags)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(args.output_dir).name, token=args.hub_token
                )
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            validation_datasets = list()
            data_files["validation"] = args.validation_file[0]
            validation_datasets.append("validation")

            if len(args.validation_file) > 1:  # handle additional validation files
                for i, file in enumerate(args.validation_file[1:]):
                    data_files[f"validation_{i+1}"] = file
                validation_datasets.append(f"validation_{i+1}")
        if args.testing_file is not None:
            test_dataset = data_files["test"] = args.testing_file

        extension = args.train_file.split(".")[-1]
        logger.info(f"Extension: {extension}")

        if extension == "txt":
            extension = "text"
        elif extension == "jsonl":
            extension = "json"
        raw_datasets = load_dataset(extension, data_files=data_files)

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # if args.config_name:
    #     config = ViTConfig.from_pretrained(args.config_name)
    # elif args.model_name_or_path:
    #     config = ViTConfig.from_pretrained(args.model_name_or_path)
    # else:
    #     config = CONFIG_MAPPING[args.model_type]()
    #     logger.warning("You are instantiating a new config instance from scratch.")

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.feature_extractor_name
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, use_fast=(not args.use_slow_tokenizer)
    )

    #  Handle cases with pad_token not defined for tokenizer
    if not "pad_token" in tokenizer.vocab:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    assert feature_extractor is not None, "Feature Extractor cannot be None"
    assert tokenizer is not None, "Tokenizer cannot be None"

    def tokenize_text(examples):
        texts = [c[0] for c in examples["captions"]]
        text_inputs = tokenizer(
            texts, padding="max_length", max_length=args.max_seq_length, truncation=True
        )

        return {"img": examples["image_path"], **text_inputs}

    # tokenize text
    proc_datasets = raw_datasets.map(
        tokenize_text,
        batched=True,
        remove_columns=["captions", "image_path"],
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    def transform_data(examples):
        """Preprocess items on the fly at __getitem__ time"""
        images = [
            io.read_image(
                os.path.join(args.data_dir, img_path), mode=io.image.ImageReadMode.RGB
            )
            for img_path in examples["img"]
        ]

        vision_inputs = feature_extractor(
            images,
            return_tensors="pt",
        )

        attention_mask = torch.tensor(examples["attention_mask"])
        labels = torch.LongTensor(examples["input_ids"])
        item = {**vision_inputs, "attention_mask": attention_mask, "labels": labels}
        return item

    proc_datasets.set_transform(transform_data)
    logger.info(f"Found datasets: {proc_datasets}")

    train_dataset = proc_datasets["train"]
    eval_dataset = proc_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.preprocessing_num_workers,
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.preprocessing_num_workers,
        pin_memory=True,
    )

    #  Create dataloaders in case of additional validation datasets
    additional_datasets = [
        n for n in validation_datasets if n != "train" and n != "validation"
    ]
    additional_dataloaders = [
        DataLoader(
            proc_datasets[d],
            # collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size,
            num_workers=args.preprocessing_num_workers,
            pin_memory=True,
        )
        for d in additional_datasets
    ]
    logger.info(f"Found {len(additional_dataloaders)} additional validation datasets")

    # Create model
    if args.model_name_or_path:
        logger.info(f"Loading model from {args.model_name_or_path}")
        model = VisionEncoderDecoderModel.from_pretrained(args.model_name_or_path)
    else:
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path=args.encoder_pretrained_model_name_or_path,
            decoder_pretrained_model_name_or_path=args.decoder_pretrained_model_name_or_path,
        )
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size

    model.decoder.resize_token_embeddings(len(tokenizer))

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [
    #             p
    #             for n, p in model.named_parameters()
    #             if not any(nd in n for nd in no_decay)
    #         ],
    #         "weight_decay": args.weight_decay,
    #     },
    #     {
    #         "params": [
    #             p
    #             for n, p in model.named_parameters()
    #             if any(nd in n for nd in no_decay)
    #         ],
    #         "weight_decay": 0.0,
    #     },
    # ]
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    if additional_dataloaders:
        additional_dataloaders = list(map(accelerator.prepare, additional_dataloaders))

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )

    if args.num_warmup_steps > 0 and args.num_warmup_steps < 1:
        args.num_warmup_steps = int(args.num_warmup_steps * args.max_train_steps)
    else:
        args.num_warmup_steps = int(args.num_warmup_steps)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  LR scheduler = {args.lr_scheduler_type}")
    logger.info(f"  Total warmup steps = {args.num_warmup_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )
    completed_steps = 0

    best_val_loss = float("inf")
    early_stop_countdown = 0

    try:
        for epoch in range(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                output = model(**batch)
                loss = output.loss

                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if (
                    step % args.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                ):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= args.max_train_steps:
                    break

                if args.log_comet and (completed_steps % args.logging_steps == 0):
                    experiment.log_metrics(
                        {"loss": loss.item()},
                        prefix="train",
                        step=completed_steps,
                        epoch=epoch,
                    )

            def run_val_dataloader(model, dataloader):
                model.eval()
                losses = list()

                for batch in tqdm(
                    dataloader, total=len(dataloader), leave=False, desc="Validation"
                ):
                    with torch.no_grad():
                        output = model(**batch)

                    loss = output.loss  # classification loss
                    losses.append(
                        accelerator.gather(loss.repeat(args.per_device_eval_batch_size))
                        .detach()
                        .cpu()
                    )

                losses = torch.cat(losses)
                losses = losses[: len(eval_dataset)]
                try:
                    mean_loss = torch.mean(losses)
                except OverflowError:
                    mean_loss = float("inf")

                return mean_loss

            # Validate at the end of each epoch
            mean_loss = run_val_dataloader(
                model,
                eval_dataloader,
            )
            logger.info(f"epoch: {epoch}, dataset: validation, mean_loss: {mean_loss}")
            if args.log_comet:
                experiment.log_metric(f"val_loss", mean_loss, epoch=epoch)

            if mean_loss < best_val_loss:
                logger.info(f"Improved loss from {best_val_loss} to {mean_loss}!")
                best_val_loss = mean_loss
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.output_dir, save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    feature_extractor.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)
                early_stop_countdown = 0
            else:
                if args.early_stop_patience > 0:
                    early_stop_countdown += 1
                    logger.info(
                        f"Early stop countdown: {early_stop_countdown}/{args.early_stop_patience}"
                    )

            # Run inference on every additional val dataset
            for dataset, dataloader in zip(additional_datasets, additional_dataloaders):
                mean_loss, _ = run_val_dataloader(model, dataloader)
                logger.info(
                    f"epoch: {epoch}, dataset: {dataset}, mean_loss: {mean_loss}"
                )
                if args.log_comet:
                    experiment.log_metric(f"{dataset}_loss", mean_loss, epoch=epoch)

            if (
                args.early_stop_patience > 0
                and early_stop_countdown == args.early_stop_patience
            ):
                logger.info(
                    f"Val loss has not improved in the last {args.early_stop_patience} epochs. Breaking"
                )
                break

            if args.push_to_hub and epoch < args.num_train_epochs - 1:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.output_dir, save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
                    repo.push_to_hub(
                        commit_message=f"Training in progress epoch {epoch}",
                        blocking=False,
                    )

    except KeyboardInterrupt:
        logger.info("Training interrupted by the user.")

    if args.output_dir is not None and args.save_last:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            feature_extractor.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training")


if __name__ == "__main__":
    main()
