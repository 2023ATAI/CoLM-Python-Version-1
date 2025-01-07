# Copyright (c) 2024 ATAI-ccsfu Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml
import codecs
from typing import Dict, Optional

_BASE_KEY = '_base_'

class Config(object):
    """
    Configuration parsing.

    The following hyper-parameters are available in the config file:
        iters: The total training steps.
        train_dataset: A training data config including /home/zjl/data/landmodel.
        val_dataset: A validation data config including /home/zjl/data/landmodel.
        sampler_cfg: SequenceSampler (SequenceSampler,RandomSampler)

    Args:
        path (str) : The path of config file, supports yaml format only.

    """
    def __init__(self, path: str):
        assert os.path.exists(path), \
            'Config path ({}) does not exist'.format(path)
        assert path.endswith('yml') or path.endswith('yaml'), \
            'Config file ({}) should be yaml format'.format(path)
        self.dic = self._parse_from_yaml(path)
        self.dic = self.update_config_dict(self.dic)
    
    @property
    def iters(self) -> int:
        return self.dic.get('iters')
    
    @property
    def train_dataset_cfg(self) -> Dict:
        return self.dic.get('train_dataset', {}).copy()

    @property
    def sampler_cfg(self) -> Dict:
        return self.dic.get('sampler', {}).copy()

    # TODO merge test_config into val_dataset
    @property
    def test_config(self) -> Dict:
        return self.dic.get('test_config', {}).copy()
    
    @classmethod
    def update_config_dict(cls, dic: dict, *args, **kwargs) -> dict:
        return update_config_dict(dic, *args, **kwargs)

    @classmethod
    def _parse_from_yaml(cls, path: str, *args, **kwargs) -> dict:
        return parse_from_yaml(path, *args, **kwargs)
    
def parse_from_yaml(path: str):
    """Parse a yaml file and build config"""
    with codecs.open(path, 'r', 'utf-8') as file:
        dic = yaml.load(file, Loader=yaml.FullLoader)

    if _BASE_KEY in dic:
        base_files = dic.pop(_BASE_KEY)
        if isinstance(base_files, str):
            base_files = [base_files]
        for bf in base_files:
            base_path = os.path.join(os.path.dirname(path), bf)
            base_dic = parse_from_yaml(base_path)
            dic = merge_config_dicts(dic, base_dic)

    return dic


def merge_config_dicts(dic, base_dic):
    """Merge dic to base_dic and return base_dic."""
    base_dic = base_dic.copy()
    dic = dic.copy()

    for key, val in dic.items():
        if isinstance(val, dict) and key in base_dic:
            base_dic[key] = merge_config_dicts(val, base_dic[key])
        else:
            base_dic[key] = val

    return base_dic


def update_config_dict(dic: dict, iters: Optional[int]=None):
    """Update config"""
    # TODO: If the items to update are marked as anchors in the yaml file,
    # we should synchronize the references.
    dic = dic.copy()

    if iters:
        dic['iters'] = iters

    return dic