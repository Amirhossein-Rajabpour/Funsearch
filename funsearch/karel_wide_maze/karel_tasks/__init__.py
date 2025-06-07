from __future__ import annotations
# from environment.karel_env.base import BaseTask
from karel_wide_maze.base import BaseTask

from .maze import Maze, MazeSparse

def get_task_cls(task_cls_name: str) -> type[BaseTask]:
    task_cls = globals().get(task_cls_name)
    assert issubclass(task_cls, BaseTask)
    return task_cls