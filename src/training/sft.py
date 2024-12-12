import time

from datetime import timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Union
from accelerate.utils import InitProcessGroupKwargs, set_seed, DataLoaderConfiguration
from accelerate import Accelerator

from utils import ArgumentParserPlus

@dataclass
class SFTArguments:
    """
    Arguments for SFT training.
    """
    model_name_or_path:str = field(
        metadata={"help": "The model checkpoint for weights initialization."}
    )
    model_revision:str = field(
        metadata={"help": "The version of the model on huggingface to use"}
    )
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
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the huggingface dataset to use, the expectation is that this is an existing dataset mixture"}
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
    args.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.push_to_hub:
        args.run_name = f"{args.run_name}__hub"

        if args.hf_entity is None:
            args.hf_entity = "taresco"
        if args.hf_repo_id is None:
            args.hf_repo_id = f"{args.hf_entity}/{args.exp_name}"
        if args.hf_repo_revision is None:
            args.hf_repo_revision = args.run_name

        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"

    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

     # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))
    dataloader_config = DataLoaderConfiguration()
    dataloader_config.use_seedable_sampler = True

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=dataloader_config,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
    )

    
    print(args)


if __name__ == "__main__":
    parser = ArgumentParserPlus(SFTArguments)
    args = parser.parse()
    main(args)