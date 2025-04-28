from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from threading import Lock

from .programs_database import ProgramsDatabase
from ...tools.profiler import ProfilerBase


class FunSearchProfiler(ProfilerBase):
    _prog_db_order = 0

    def __init__(self,
                 log_dir: str | None = None,
                 evaluation_name='Problem',
                 method_name='FunSearch',
                 *,
                 initial_num_samples=0,
                 program_db_register_interval: int = 100,
                 log_style='complex',
                 **kwargs):
        super().__init__(log_dir=log_dir,
                         evaluation_name=evaluation_name,
                         method_name=method_name,
                         initial_num_samples=initial_num_samples,
                         log_style=log_style,
                         **kwargs)
        if log_dir:
            self._prog_db_path = os.path.join(self._log_dir, 'prog_db')
            os.makedirs(self._prog_db_path, exist_ok=True)
        self._intv = program_db_register_interval
        self._db_lock = Lock()

    def register_program_db(self, program_db: ProgramsDatabase):
        """Save ProgramDB to a file.
        [
            [{'score': -300, 'functions': [xxx, xxx, xxx, ...]}, {'score': -200, 'functions': [xxx, xxx, xxx, ...]}, {...}],
            [{...}, {...}],
        ]
        """
        try:
            if (self.__class__._num_samples == 0 or
                    self.__class__._num_samples % self._intv != 0):
                return
            self._db_lock.acquire()
            self.__class__._prog_db_order += 1
            isld_list = []
            for island in program_db.islands:
                clus_list = []
                for k, v in island.clusters.items():
                    funcs = [str(f) for f in v.programs]
                    func_dic = {'score': k, 'functions': funcs}
                    clus_list.append(func_dic)
                isld_list.append(clus_list)

            path = os.path.join(self._prog_db_path, f'db_{self.__class__._prog_db_order}.json')
            with open(path, 'w') as f:
                json.dump(isld_list, f)
        finally:
            if self._db_lock.locked():
                self._db_lock.release()

