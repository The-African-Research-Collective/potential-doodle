import os
import sys
import dataclasses

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from dataclasses import dataclass, fields
from typing import List, Optional, Union, Tuple, Any, NewType
from transformers import HfArgumentParser
from datasets.builder import DatasetGenerationError

DataClassType = NewType("DataClassType", Any)


class ArgumentParserPlus(HfArgumentParser):
    def parse_yaml_and_args(self, yaml_arg: str, other_args: Optional[List[str]] = None) -> List[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args}
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # noqa adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys

                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == List[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type == bool:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        else:
                            inputs[arg] = False

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(f"Duplicate argument provided: {arg}, may cause unexpected behavior")

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs

    def parse(self) -> Union[DataClassType, Tuple[DataClassType]]:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file(os.path.abspath(sys.argv[1]))
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_and_args(os.path.abspath(sys.argv[1]), sys.argv[2:])
        # parse command line args only
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output

# functions for handling different formats of messages
def convert_alpaca_gpt4_to_messages(example):
    """
    Convert an instruction in inst-output to a list of messages.
    e.g. vicgalle/alpaca-gpt4"""
    messages = [
        {
            "role": "user",
            "content": (
                "Below is an instruction that describes a task, paired with an input that provides "
                "further context. Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Input:\n{example['input']}\n\n"
                "### Response:"
            ),
        },
        {"role": "assistant", "content": example["output"]},
    ]
    example["messages"] = messages
    return example

def convert_codefeedback_single_turn_to_messages(example):
    """
    Convert a query-answer pair to a list of messages.
    e.g. m-a-p/CodeFeedback-Filtered-Instruction"""
    messages = [
        {"role": "user", "content": example["query"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    example["messages"] = messages
    return example

def convert_metamath_qa_to_messages(example):
    """
    Convert a query-response pair to a list of messages.
    e.g. meta-math/MetaMathQA"""
    messages = [
        {"role": "user", "content": example["query"]},
        {"role": "assistant", "content": example["response"]},
    ]
    example["messages"] = messages
    return example

def convert_code_alpaca_to_messages(example):
    """
    Convert a prompt-completion pair to a list of messages.
    e.g. HuggingFaceH4/CodeAlpaca_20K"""
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    example["messages"] = messages
    return example

def convert_open_orca_to_messages(example):
    """
    Convert a question-response pair to a list of messages.
    e.g. Open-Orca/OpenOrca"""
    messages = [
        {"role": "system", "content": example["system_prompt"]},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["response"]},
    ]
    example["messages"] = messages
    return example


def conversations_to_messages(example):
    """
    Convert from conversations format to messages.

    E.g. change "from": "user" to "role": "user"
        and "value" to "content"
        and "gpt" to "assistant"

    WizardLMTeam/WizardLM_evol_instruct_V2_196k
    """
    name_mapping = {
        "gpt": "assistant",
        "Assistant": "assistant",
        "assistant": "assistant",
        "user": "user",
        "User": "user",
        "human": "user",
    }
    messages = [{"role": name_mapping[conv["from"]], "content": conv["value"]} for conv in example["conversations"]]
    example["messages"] = messages
    return example


def convert_rejection_samples_to_messages(example):
    """
    Convert a rejection sampling dataset to messages.
    """
    example["messages"] = example["chosen"]
    return example

def mix_datasets(
    dataset_mixer: Union[dict, list],
    splits: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle: bool = True,
    save_data_dir: Optional[str] = None,
    need_columns: Optional[List[str]] = None,
    keep_ids: bool = False,
    add_source_col: bool = False,
) -> DatasetDict:
    
    """
    dataset_mixer: Union[dict, list]
        The dataset or datasets to be mixed. If a list, the datasets must be in the same order as the splits.
        e.g. ["dataset_a", 1000, "dataset_b", 500] will mix 1000 samples from dataset_a and 500 samples from dataset_b.
        ["dataset_a", 0.5, "dataset_b", 0.5] will mix 50% of dataset_a and 50% of dataset_b.
        {"dataset_a": 1000, "dataset_b": 500} will mix 1000 samples from dataset_a and 500 samples from dataset_b.
    splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in
            all datasets and have a `train_` or `test_` prefix.
    configs (Optional[List[str]], *optional*, defaults to `None`):
        List of dataset config names. If given must be the same length as 'dataset_mixer' keys.
    columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
        Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
        and for cpt this should be (at least) the text column.
    shuffle (`bool`, *optional*, defaults to `True`):
        Whether to shuffle the training and testing/validation data.
    save_data_dir (Optional[str], *optional*, defaults to `None`):
        Optional directory to save training/test mixes on.
    need_columns (Optional[List[str]], *optional*, defaults to `None`):
        Column names that are required to be in the dataset.
        Quick debugging when mixing heterogeneous datasets.
    keep_ids (`bool`, *optional*, defaults to `False`):
        Whether to keep ids for training that are added during mixing.
        Used primarily in mix_data.py for saving, or the saved dataset has IDs already.
    add_source_col (`bool`, *optional*, defaults to `False`):
        Whether to add a column to the dataset that indicates the source of the data explicitly.
    """
    if isinstance(dataset_mixer, list):
        assert len(dataset_mixer) % 2 == 0, f"Data mixer list length is not even: {dataset_mixer}"
        mixer_dict = {}
        i = 0
        while i < len(dataset_mixer) - 1:
            assert isinstance(dataset_mixer[i], str), f"Invalid type in data mixer: {dataset_mixer}"
            if "." in dataset_mixer[i + 1]:
                value = float(dataset_mixer[i + 1])
            else:
                value = int(dataset_mixer[i + 1])
            mixer_dict[dataset_mixer[i]] = value
            i += 2
        dataset_mixer = mixer_dict
    
    splits = ["train", "test"] if splits is None else splits
    configs = [None] * len(dataset_mixer) if not configs else configs
    columns_to_keep = [] if columns_to_keep is None else columns_to_keep

    if configs is not None and len(configs) != len(dataset_mixer):
        raise ValueError("The number of given dataset config names must be the same as the given number of datasets.")

    # print save location
    if save_data_dir:
        print(f"Saving mixed dataset to {save_data_dir}")

    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    frac_or_sample_list = []
    for (ds, frac_or_samples), ds_config in zip(dataset_mixer.items(), configs):
        frac_or_sample_list.append(frac_or_samples)
        for split in splits:
            # if dataset ends with .json or .jsonl, load from file
            if ds.endswith(".json") or ds.endswith(".jsonl"):
                dataset = load_dataset("json", data_files=ds, split=split)
            else:
                try:
                    # Try first if dataset on a Hub repo
                    dataset = load_dataset(ds, ds_config, split=split)
                except DatasetGenerationError:
                    # If not, check local dataset
                    dataset = load_from_disk(os.path.join(ds, split))

            # shuffle dataset if set
            if shuffle:
                dataset = dataset.shuffle(seed=42)

            # assert that needed columns are present
            if need_columns:
                if not all(col in dataset.column_names for col in need_columns):
                    raise ValueError(f"Needed column {need_columns} not found in dataset {dataset.column_names}.")

            # handle per-case conversions
            # if "instruction" and "output" columns are present and "messages" is not, convert to messages
            if (
                "instruction" in dataset.column_names
                and "output" in dataset.column_names
                and "messages" not in dataset.column_names
            ):
                dataset = dataset.map(convert_alpaca_gpt4_to_messages, num_proc=10)
            elif (
                "prompt" in dataset.column_names
                and "completion" in dataset.column_names
                and "messages" not in dataset.column_names
            ):
                dataset = dataset.map(convert_code_alpaca_to_messages, num_proc=10)
            elif "conversations" in dataset.column_names and "messages" not in dataset.column_names:
                dataset = dataset.map(conversations_to_messages, num_proc=10)
            elif (
                "question" in dataset.column_names
                and "response" in dataset.column_names
                and "messages" not in dataset.column_names
            ):
                dataset = dataset.map(convert_open_orca_to_messages, num_proc=10)
            elif (
                "query" in dataset.column_names
                and "answer" in dataset.column_names
                and "messages" not in dataset.column_names
            ):
                dataset = dataset.map(convert_codefeedback_single_turn_to_messages, num_proc=10)
            elif (
                "query" in dataset.column_names
                and "response" in dataset.column_names
                and "messages" not in dataset.column_names
            ):
                dataset = dataset.map(convert_metamath_qa_to_messages, num_proc=10)
            elif (
                "chosen" in dataset.column_names
                and "rejected" in dataset.column_names
                and "reference_completion" in dataset.column_names
                and "messages" not in dataset.column_names
            ):
                dataset = dataset.map(convert_rejection_samples_to_messages, num_proc=10)

            # if id not in dataset, create it as ds-{index}
            if "id" not in dataset.column_names:
                id_col = [f"{ds}_{i}" for i in range(len(dataset))]
                dataset = dataset.add_column("id", id_col)

            # Remove redundant columns to avoid schema conflicts on load
            dataset = dataset.remove_columns(
                [col for col in dataset.column_names if col not in (columns_to_keep + ["id"])]
            )

            # if add_source_col, add that column
            if add_source_col:
                source_col = [ds] * len(dataset)
                dataset = dataset.add_column("source", source_col)

            # for cols in columns_to_keep, if one is not present, add "None" to the column
            for col in columns_to_keep:
                if col not in dataset.column_names:
                    dataset = dataset.add_column(col, [None] * len(dataset))

            # add tag to the dataset corresponding to where it was sourced from, for
            if "train" in split:
                raw_train_datasets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if len(raw_val_datasets) == 0 and len(raw_train_datasets) == 0:
        raise ValueError("No datasets loaded.")
    elif len(raw_train_datasets) == 0:
        # target features are the features of the first dataset post load
        target_features = raw_val_datasets[0].features
    else:
        # target features are the features of the first dataset post load
        target_features = raw_train_datasets[0].features

    if any(frac_or_samples < 0 for frac_or_samples in frac_or_sample_list):
        raise ValueError("Dataset fractions / lengths cannot be negative.")

    # if any > 1, use count
    if any(frac_or_samples > 1 for frac_or_samples in frac_or_sample_list):
        is_count = True
        # assert that all are integers
        if not all(isinstance(frac_or_samples, int) for frac_or_samples in frac_or_sample_list):
            raise NotImplementedError("Cannot mix fractions and counts, yet.")
    else:
        is_count = False

    if len(raw_train_datasets) > 0:
        train_subsets = []
        # Manage proportions
        for dataset, frac_or_samples in zip(raw_train_datasets, frac_or_sample_list):
            # cast features (TODO, add more feature regularization)
            dataset = dataset.cast(target_features)
            # TODO selection can be randomized.
            if is_count:
                train_subset = dataset.select(range(frac_or_samples))
            else:
                train_subset = dataset.select(range(int(frac_or_samples * len(dataset))))
            train_subsets.append(train_subset)

        raw_datasets["train"] = concatenate_datasets(train_subsets)

    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        for dataset in raw_val_datasets:
            # cast features (TODO, add more feature regularization)
            dataset = dataset.cast(target_features)

        raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with splits {splits}."
            "Check the dataset has been correctly formatted."
        )

    # optional save
    if save_data_dir:
        for split in raw_datasets:
            raw_datasets[split].to_json(save_data_dir + f"mixed_ds_{split}.json")

    if not keep_ids:
        # remove id column
        if len(raw_train_datasets) > 0:
            if "id" in raw_datasets["train"].column_names:
                raw_datasets["train"] = raw_datasets["train"].remove_columns("id")
        if len(raw_val_datasets) > 0:
            if "id" in raw_datasets["test"].column_names:
                raw_datasets["test"] = raw_datasets["test"].remove_columns("id")

    return raw_datasets