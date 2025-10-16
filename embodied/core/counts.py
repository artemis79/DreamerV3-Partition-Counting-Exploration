import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial as bind

import elements
from jax._src.core import axis_frame
import numpy as np

from . import chunk as chunklib
from . import limiters
from . import selectors


class Counts:
    def __init__(self, act_space, stoch_size=1, classes_size=1, beta=1):
        self.num_actions = act_space['action'].high - act_space['action'].low + 1
        self.stoch_size = stoch_size
        self.classes_size = classes_size
        self.beta = beta
        self.counts = np.ones((self.num_actions, self.stoch_size, self.classes_size), dtype=np.int32)

    @elements.timer.section('counts_add')    
    def counts_add(self, step, worker=0):
        step = {k: v for k, v in step.items() if not k.startswith('log/')}
        action_id = step['action']
        stoch_state = step['dyn/stoch']

        self.counts[action_id] = self.counts[action_id] + stoch_state
    
    def get_intrinsic_reward(self, action, stoch_state):
        stoch_state = np.repeat(stoch_state, self.num_actions, axis=0)
        counts = self.counts * stoch_state
        counts = np.min(np.sum(counts, axis=2), axis=1)
        rewards = self.beta * np.sqrt(2 * np.log(np.sum(self.counts)) / counts[action])
        return rewards
