from .dataset import add_dataset_args
from .model import add_model_args

def add_task_args(parser):
    group = parser.add_argument_group('Breakout')
    add_dataset_args(group)
    add_model_args(group)

