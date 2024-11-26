# name: str: MoonLander
# Parameters:
# max_steps: int: 500
# end
from __future__ import annotations

from typing import Any
import gym
import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.machine_learning.moon_lander.template import template_program, task_description

__all__ = ['MoonLander']


def evaluate(env: gym.Env, action_select: callable) -> float | None:
    try:
        fitness = []
        # parallel evaluation 4 times, core=4
        # fitness = Parallel(n_jobs=4)(delayed(evaluate_single)(env, action_select) for _ in range(5))
        for i in range(5):
            fitness.append(evaluate_single(env, action_select))
        fitness = np.mean(fitness)

        return fitness
    except Exception as e:
        return None

def evaluate_single(env: gym.Env, action_select: callable) -> float:
    """Evaluate heuristic function on moon lander problem."""

    observation, _ = env.reset()  # initialization
    action = 0  # initial action
    reward = 0
    yv = []

    for i in range(env._max_episode_steps + 1):  # protect upper limits
        action = action_select(observation[0], observation[1],
                               observation[2],
                               observation[3],
                               observation[4],
                               observation[5],
                               observation[6],
                               observation[7],
                               action)
        observation, reward, done, truncated, info = env.step(action)
        yv.append(observation[3])

        if done or truncated:
            # self.env.close()
            fitness = abs(observation[0]) + abs(yv[-2]) - ((observation[6] + observation[7]) - 2) + 1
            if reward >= 100:
                return -(i + 1) / env._max_episode_steps
            else:
                return -fitness


class MoonLander(Evaluation):
    """Evaluator for moon lander problem."""

    def __init__(self, max_steps=500, **kwargs):
        """
            Args:
                - 'max_steps' (int): Maximum number of steps allowed per episode in the MountainCar-v0 environment (default is 500).
                - '**kwargs' (dict): Additional keyword arguments passed to the parent class initializer.

            Attributes:
                - 'env' (gym.Env): The MountainCar-v0 environment with a modified maximum episode length.
        """

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=20
        )

        self.env = None
        self.env = gym.make('LunarLander-v2')
        self.env._max_episode_steps = max_steps

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return evaluate(self.env, callable_func)
