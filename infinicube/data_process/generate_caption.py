# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import json
import multiprocessing
import os
import socket
import subprocess
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import torch
import torch.distributed as dist
from qwen_vl_utils import process_vision_info
from termcolor import colored
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

USER_PROMPTS = {
    "weather": """
        You are a video captioning specialist whose goal is to generate high-quality English captions by referring to the user's input videos. You are given a video from a camera on a vehicle. The input video is about a driving scene. Your task is to carefully analyze the weather condition in the video and generate a caption. Strictly adhere to the task requirements provided.
        Task Requirements:
        1. You need to control the caption to no more than 40 words.
        2. You need to describe the weather condition in the video, use keywords like daytime, nighttime, sunny, cloudy, rainy, snowing, etc.
        3. Always output in English. Be concise.
    """,
}


class CaptionGenerationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        video_url_list,
        fps_in_captioning=10,
        caption_keys=["weather"],
        resolution_h_w=(480, 832),
    ):
        self.video_url_list = video_url_list
        self.prompts_dict = USER_PROMPTS
        self.fps = fps_in_captioning  # used in VLM input
        self.caption_keys = caption_keys
        self.resolution_h_w = resolution_h_w  # used in VLM input
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        print(f"{len(self.video_url_list)} videos for all nodes.")

    def __len__(self):
        return len(self.video_url_list)

    def __getitem__(self, idx):
        video_url = self.video_url_list[idx]
        try:
            caption_key_to_input = {}
            for caption_key in self.caption_keys:
                prompt = self.prompts_dict[caption_key]
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": prompt}]},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_url,
                                "fps": self.fps,
                                # "resized_height": self.resolution_h_w[0],
                                # "resized_width": self.resolution_h_w[1],
                            }
                        ],
                    },
                ]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    fps=self.fps,
                    do_resize=False,
                )
                caption_key_to_input[caption_key] = inputs
            return video_url, caption_key_to_input

        except Exception as e:
            print(f"Error processing video {video_url}: {e}")
            return None


class DistributedProcessor(ABC):
    """
    An abstract base class for distributed data processing pipelines.

    This class provides a template for setting up a distributed environment,
    finding items that need to be processed, and running the processing loop
    in a distributed manner. Subclasses must implement the abstract methods
    to provide the specific logic for their task.

    Args:
        input_list_file: The file path to the input list file.
        output_dir: The directory to save the output files.
        split: The split of the input list file. It is a string of the form 'start:interval', where start and interval are integers.
            This is used when you want to split the input list file into multiple parts, and only process a subset of the input list file.
    """

    def __init__(self, input_list_file: str, output_dir: str, split: str = "0:1"):
        self.input_list_file = input_list_file
        self.output_dir = output_dir
        self.output_dir_p = Path(output_dir)
        self.use_distributed = self._setup_distributed()
        self.model = None
        self.this_split, self.total_splits = split.split(":")
        self.this_split = int(self.this_split)
        self.total_splits = int(self.total_splits)

    def _setup_distributed(self):
        """
        Initializes the distributed environment, supporting both Slurm and torchrun.
        """

        # launched by torchrun
        if "LOCAL_RANK" in os.environ:  # Fallback for torchrun
            dist.init_process_group(backend="nccl")
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.global_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            print(
                colored(
                    f"[torchrun] Rank {self.global_rank}/{self.world_size} -> Using GPU: {self.local_rank}",
                    "yellow",
                )
            )
            torch.cuda.set_device(self.local_rank)
            return True

        # launched by Slurm
        elif "SLURM_PROCID" in os.environ:
            # Manually set environment variables for torch.distributed
            # Get master address from Slurm's nodelist
            nodelist = os.environ["SLURM_JOB_NODELIST"]
            try:
                master_addr = (
                    subprocess.check_output(
                        f"scontrol show hostnames {nodelist} | head -n 1", shell=True
                    )
                    .decode()
                    .strip()
                )
                os.environ["MASTER_ADDR"] = master_addr
            except Exception as e:
                print(
                    f"Could not get master address from Slurm, falling back to localhost. Error: {e}"
                )
                os.environ["MASTER_ADDR"] = "127.0.0.1"

            os.environ["MASTER_PORT"] = (
                "29500"  # A static port, usually fine with --exclusive
            )
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

            # PyTorch can now initialize using the standard env:// method
            dist.init_process_group(backend="nccl", init_method="env://")

            self.global_rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["SLURM_LOCALID"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            hostname = socket.gethostname()
            print(
                colored(
                    f"[Slurm] Rank {self.global_rank}/{self.world_size} -> Using GPU: {self.local_rank} on hostname: {hostname}",
                    "yellow",
                )
            )
            torch.cuda.set_device(self.local_rank)
            return True

        else:
            # Not a distributed job
            self.local_rank = 0
            self.global_rank = 0
            self.world_size = 1
            print(
                colored(f"[Single GPU Mode] - Using GPU: {self.local_rank}", "yellow")
            )
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
            return False

    def _is_processed(self, item_and_output_dir):
        """Checks if a single item has already been processed."""
        item, output_dir = item_and_output_dir
        output_file = self.get_output_file_path(item, output_dir)
        return output_file.exists()

    def _find_todo_items(self, all_items):
        """Finds items that have not been processed yet using multiprocessing."""
        tasks = [(item, self.output_dir) for item in all_items]

        with multiprocessing.Pool(processes=64) as pool:
            results = list(
                tqdm(
                    pool.imap(self._is_processed, tasks),
                    total=len(tasks),
                    desc="Checking for processed items",
                )
            )

        return [item for item, processed in zip(all_items, results) if not processed]

    def run(self):
        """The main execution method for the processing pipeline."""
        if isinstance(self.input_list_file, io.StringIO):
            all_items = json.load(self.input_list_file)
        else:
            all_items = json.load(open(self.input_list_file))
        all_items = all_items[self.this_split :: self.total_splits]

        if self.use_distributed:
            objects_to_broadcast = [None]
            if self.global_rank == 0:
                todo_items = self._find_todo_items(all_items)
                colored_text = colored(
                    f"{len(todo_items)} / {len(all_items)} items",
                    "green",
                    attrs=["bold"],
                )
                print(
                    f"Rank {self.global_rank}: Found {colored_text} to process in total."
                )
                objects_to_broadcast = [todo_items]

            device = torch.device(f"cuda:{self.local_rank}")
            dist.broadcast_object_list(objects_to_broadcast, src=0, device=device)
            todo_items = objects_to_broadcast[0]
            if self.global_rank != 0:
                colored_text = colored(
                    f"{len(todo_items)} items", "blue", attrs=["bold"]
                )
                print(
                    f"Rank {self.global_rank}: Received {colored_text} // {self.world_size} to process."
                )
            dist.barrier()
        else:
            todo_items = self._find_todo_items(all_items)
            colored_text = colored(f"{len(todo_items)} items", "green", attrs=["bold"])
            print(f"Found {colored_text} to process in total.")

        if not todo_items:
            if self.global_rank == 0:
                print(colored("No items to process. Exiting.", "red", attrs=["bold"]))
            return

        self.model = self.setup_model()
        self.dataset = self.get_dataset(todo_items)
        collate_fn = self.get_collate_fn()

        sampler = DistributedSampler(self.dataset) if self.use_distributed else None
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            shuffle=False,  # never shuffle
        )

        # Only show the progress bar on the main process to avoid clutter
        iterable = dataloader
        if self.global_rank == 0:
            iterable = tqdm(dataloader, desc="Processing items (Rank 0)")

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for idx, batch in enumerate(iterable):
                if batch is None or not any(item is not None for item in batch):
                    continue

                processed_results = self.process_batch(batch)
                executor.submit(self.save_results, processed_results)

    @abstractmethod
    def get_output_file_path(self, item: str, output_dir: str) -> Path:
        """
        Given an input item (e.g., a URL or file path), return the
        Path object for its corresponding output file.
        """
        pass

    @abstractmethod
    def get_dataset(self, items: list):
        """Return an instance of a torch.utils.data.Dataset."""
        pass

    @abstractmethod
    def get_collate_fn(self):
        """Return the collate function for the DataLoader."""
        pass

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Return the batch size for the DataLoader."""
        pass

    @property
    @abstractmethod
    def num_workers(self) -> int:
        """Return the number of workers for the DataLoader."""
        pass

    @abstractmethod
    def setup_model(self):
        """Load and return the model required for processing."""
        pass

    @abstractmethod
    def process_batch(self, batch):
        """
        Process a single batch of data from the DataLoader and return the results.
        The results should be in a format that save_results can handle.
        """
        pass

    @abstractmethod
    def save_results(self, results):
        """Save the processed results from a batch to disk."""
        pass


def collate_fn_custom(batch):
    # filter out None
    batch = [item for item in batch if item is not None]

    # if the batch is empty after filtering, return None
    if not batch:
        return None

    return torch.utils.data.default_collate(batch)


class CaptionProcessor(DistributedProcessor):
    def __init__(self, video_url_list, output_dir, caption_key, split):
        super().__init__(video_url_list, output_dir, split)
        self.caption_key = caption_key

    def get_output_file_path(self, item: str, output_dir: str) -> Path:
        output_dir_p = Path(output_dir)
        video_name = item.split("/")[-1]
        return output_dir_p / f"{Path(video_name).stem}.json"

    def get_dataset(self, items: list):
        return CaptionGenerationDataset(
            video_url_list=items, fps_in_captioning=10, caption_keys=self.caption_key
        )

    def get_collate_fn(self):
        return collate_fn_custom

    @property
    def batch_size(self) -> int:
        return 1

    @property
    def num_workers(self) -> int:
        return 4

    def setup_model(self):
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            dtype=torch.bfloat16,
            device_map=f"cuda:{self.local_rank}",
            # attn_implementation="flash_attention_2",
        )

    def process_batch(self, batch):
        video_urls, caption_key_to_input = batch
        caption_key_to_output_captions = {}
        dataset_processor = (
            self.dataset.processor
        )  # Get a processor instance from dataset

        for caption_key, input_data in caption_key_to_input.items():
            input_data["second_per_grid_ts"] = (
                input_data["second_per_grid_ts"][0].numpy().tolist()
            )
            input_data["attention_mask"] = input_data["attention_mask"].squeeze(1)
            input_data["input_ids"] = input_data["input_ids"].squeeze(1)
            input_data["video_grid_thw"] = input_data["video_grid_thw"].squeeze(1)
            input_data["pixel_values_videos"] = input_data[
                "pixel_values_videos"
            ].flatten(0, 1)

            input_data = input_data.to(f"cuda:{self.local_rank}")
            generated_ids = self.model.generate(**input_data, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(input_data.input_ids, generated_ids)
            ]
            output_text = dataset_processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            caption_key_to_output_captions[caption_key] = output_text

        return video_urls, caption_key_to_output_captions

    def save_results(self, results):
        video_urls, caption_key_to_output_captions = results
        for idx, video_url in enumerate(video_urls):
            preset_dict = {"weather": ""}
            for caption_key, output_caption in caption_key_to_output_captions.items():
                preset_dict[caption_key] = output_caption[idx]

            output_file = self.get_output_file_path(video_url, self.output_dir)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                json.dump(preset_dict, f, indent=4)


"""
How to run:
# single gpu
python infinicube/data_process/generate_caption.py

# multi gpu
torchrun --nproc_per_node=8 infinicube/data_process/generate_caption.py

# multi node
torchrun --nproc_per_node=8 infinicube/data_process/generate_caption.py -s 0,2
torchrun --nproc_per_node=8 infinicube/data_process/generate_caption.py -s 1,2
"""


@click.command()
@click.option("--data_root", "-d", default="data/", type=str, required=True)
@click.option(
    "--input_attribute", "-i", default="video_480p_front", type=str, required=True
)
@click.option("--output_attribute", "-o", default="caption", type=str, required=True)
@click.option("--caption_key", "-c", type=str, multiple=True, default=["weather"])
@click.option(
    "--split",
    "-s",
    type=str,
    default="0:1",
    help="The split of the input list file. It is a string of the form 'start:interval', where start and interval are integers."
    + "This is used when you want to split the input list file into multiple parts, and only process a subset of the input list file.",
)
def main(data_root, input_attribute, output_attribute, caption_key, split):
    video_url_list = [
        str(path) for path in (Path(data_root) / input_attribute).rglob("*.mp4")
    ]
    video_url_list_json_object = io.StringIO()
    json.dump(video_url_list, video_url_list_json_object)
    video_url_list_json_object.seek(0)

    output_dir = (Path(data_root) / output_attribute).as_posix()
    processor = CaptionProcessor(
        video_url_list_json_object, output_dir, caption_key, split
    )
    processor.run()


if __name__ == "__main__":
    main()
