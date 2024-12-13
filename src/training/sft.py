"""
Some Training Notes:

local_main_process typically refers to the main process on a specific node/machine in a distributed setup. There can be multiple local_main_processes - one per node.
main_process (or global main process) refers to the single primary process that coordinates across all nodes in the distributed system. There is only one main_process across the entire training setup.
"""

import os
import time
import logging
import datasets
import transformers

from datetime import timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Union
from accelerate.utils import InitProcessGroupKwargs, set_seed, DataLoaderConfiguration
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset

from transformers import AutoConfig

from utils import ArgumentParserPlus, mix_datasets

logger = get_logger(__name__)

@dataclass
class DatasetArguments:
    """
    Arguments for dataset configuration.
    """
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the huggingface dataset to use, the expectation is that this is an existing dataset mixture"}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use"}
    )
    dataset_mixer: Optional[dict] = field(
        default=None, metadata={"help": "A dictionary of datasets (local or HF) to sample from."}
    )
    dataset_mixer_list: Optional[list[str]] = field(
        default=None, metadata={"help": "A list of datasets (local or HF) to sample from."}
    )
    dataset_mix_dir: Optional[str] = field(
        default=None, metadata={"help": "The directory to save the mixed dataset to disk."}
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the training file"}
    )

@dataclass
class ModelArguments:
    """
    Arguments for model configuration.
    """
    model_name_or_path: str = field(
        metadata={"help": "The model checkpoint for weights initialization."}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The version of the model on huggingface to use"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The model configuration to use, it is usually a model name"}
    )
    trust_remote_code: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to trust remote code"}
    )

@dataclass
class ExperimentArguments:
    """
    Arguments for experiment configuration.
    """
    exp_name: str = field(
        metadata={"help": "The name of the experiment"}
    )
    run_name: str = field(
        default=None,
        metadata={"help": "The name of the run"}
    )
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the model to the hub"}
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The output directory to save the model and logs"}
    )
    with_tracking: bool = field(
        default=False,
        metadata={"help": "Whether to enable experiment trackers for logging."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "The number of gradient accumulation steps to use."},
    )
    report_to: Union[str, List[str]] = field(
        default="all",
        metadata={
            "help": "The integration(s) to report results and logs to. "
            "Can be a single string or a list of strings. "
            "Options are 'tensorboard', 'wandb', 'comet_ml', 'clearml', or 'all'. "
            "Specify multiple by listing them: e.g., ['tensorboard', 'wandb']"
        },
    )
    hf_repo_id: Optional[str] = field(
        default=None,
        metadata={"help": "The huggingface repository id to push the model to"}
    )
    hf_entity: Optional[str] = field(
        default=None,
        metadata={"help": "The huggingface entity to push the model to"}
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "The seed to use for the run"}
    )
    reports_to: Optional[List[str]] = field(
        default="wandb",
        metadata={"help": "The service to report to"}
    )
    timeout: Optional[int] = field(
        default=600,
        metadata={"help": "The timeout for the run"}
    )
    

def main(args: ArgumentParserPlus):

    exp_args, model_args, data_args = args[0], args[1], args[2]

    print(exp_args)

    exp_args.run_name = f"{exp_args.exp_name}__{exp_args.seed}__{int(time.time())}"
    if exp_args.push_to_hub:
        exp_args.run_name = f"{exp_args.run_name}__hub"

        if exp_args.hf_entity is None:
            args.hf_entity = "taresco"
        if exp_args.hf_repo_id is None:
            exp_args.hf_repo_id = f"{exp_args.hf_entity}/{exp_args.exp_name}"
        if exp_args.hf_repo_revision is None:
            exp_args.hf_repo_revision = exp_args.run_name

        exp_args.hf_repo_url = f"https://huggingface.co/{exp_args.hf_repo_id}/tree/{exp_args.hf_repo_revision}"

    accelerator_log_kwargs = {}

    if exp_args.with_tracking:
        accelerator_log_kwargs["log_with"] = exp_args.report_to
        accelerator_log_kwargs["project_dir"] = exp_args.output_dir

     # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=exp_args.timeout))
    dataloader_config = DataLoaderConfiguration()
    dataloader_config.use_seedable_sampler = True

    accelerator = Accelerator(
        gradient_accumulation_steps=exp_args.gradient_accumulation_steps,
        dataloader_config=dataloader_config,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
    )

     # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    
    if exp_args.seed:
        set_seed(exp_args.seed)

    # Create output directory in the main process
    if accelerator.is_main_process:
        if exp_args.output_dir is not None:
            os.makedirs(exp_args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    if data_args.dataset_name:
        dataset = datasets.load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
        )
    elif data_args.dataset_mixer:
        dataset = mix_datasets(
            data_args.dataset_mixer,
            configs=data_args.dataset_config_name,
            splits=["train"],
            save_data_dir=data_args.dataset_mix_dir if accelerator.is_main_process else None,
            columns_to_keep=["messages"],
        )
    elif args.dataset_mixer_list:
        dataset = mix_datasets(
            data_args.dataset_mixer_list,
            configs=args.dataset_config_name,
            splits=["train"],
            save_data_dir=data_args.dataset_mix_dir if accelerator.is_main_process else None,
            columns_to_keep=["messages"],
        )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )

    
    # load pretrained model and tokenizer
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )
        


if __name__ == "__main__":
    parser = ArgumentParserPlus((ExperimentArguments, ModelArguments, DatasetArguments))
    args = parser.parse()
    main(args)