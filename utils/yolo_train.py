from pathlib import Path
from ultralytics import YOLO, checks
from ultralytics.models.yolo.detect.train import DetectionTrainer
from IPython import display
from typing import Union

import torch
import os
import time
import csv

class YoloTrain(YOLO):
    def __init__(self, model: Union[str, Path] = 'yolov8n.pt', task=None) -> None:
        super().__init__(model, task)

        self.set_gpu()
        self.add_callback("on_train_epoch_end", self.memory_monitor_callback)

    @staticmethod
    def set_gpu() -> None:
        torch.cuda.set_device(0) # Set to your desired GPU number
        os.environ['PYTORCH_CUDA_ALLOC_CONF']='garbage_collection_threshold:0.6,max_split_size_mb:128'

        display.clear_output()
        checks()

    @staticmethod
    def memory_monitor_callback(trainer: DetectionTrainer, results_file: str = 'gpu_results.csv') -> None:
        # GPU memory usage
        allocated_memory = torch.cuda.memory_allocated()
        reserved_memory = torch.cuda.memory_reserved()

        props = torch.cuda.get_device_properties(torch.cuda.current_device())

        with open(f'{trainer.save_dir}/{results_file}', mode='a', newline='') as file:
            writer = csv.writer(file)
            if trainer.epoch == 0:
                writer.writerow(['epoch', 'gpu_allocated', 'gpu_reserved', 'gpu_total', 'ratio_allocated'])
            writer.writerow([trainer.epoch, allocated_memory / (1024 ** 2),
                              reserved_memory / (1024 ** 2), 
                              props.total_memory / (1024 ** 3), 
                              allocated_memory / props.total_memory])
            
    def train_model(self, **kwargs) -> None:
        start = time.process_time()
        self.train(**kwargs)
        timeAmin = (time.process_time() - start)/60
        print(f"##### Time for running model: {timeAmin} min #####")