from dataclasses import dataclass, field
from typing import Optional, List, Union

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
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "The tokenizer to use"}
    )
    tokenizer_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The version of the tokenizer on huggingface to use"}
    )
    use_slow_tokenizer: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use a slow tokenizer"}
    )
    use_qlora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use the qlora tokenizer"}
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
    use_flash_attention: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use flash attention"}
    )