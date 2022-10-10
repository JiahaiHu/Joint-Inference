# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging

import numpy as np
import mindspore as ms
import yaml

import model.controller as code2space

LOG = logging.getLogger(__name__)
os.environ['BACKEND_TYPE'] = 'MINDSPORE'

class Estimator:

    def __init__(self, **kwargs):
        """
        initialize logging configuration
        """
        self.controller = code2space.Code2SpacePartition()
        self.net = None
        self.config = self.load_config()
        self.config['cpu_freq_ratio'] = 1
        c = self.config
        self.partition_idx, utility = self.controller.get_partition_point(c['network_bw'], c['overall_thr_weight'],
                                                                    c['mobile_thr_weight'], c['cpu_freq_ratio'],
                                                                    c['server_ratio'])

    @staticmethod
    def load_config():
        config = yaml.safe_load(open('./config.yaml', 'r'))
        return config

    def load(self, model_url=""):
        LOG.info("load")
        self.net, self.partition_layer = self.controller.init_model_server(self.partition_idx)

    @staticmethod
    def preprocess(image, input_shape):
        """Preprocess functions in edge model inference"""
        return 0

    @staticmethod
    def postprocess(model_output):
        result_np = model_output.asnumpy()
        return result_np.tolist()

    def predict(self, data, cpu_freq_ratio, **kwargs):
        input_np = np.array(data)
        input_feed = ms.Tensor(input_np, ms.float32)
        result = self.net(input_feed)
        result_np = self.postprocess(result)

        # reload model if partition point is changed
        c = self.load_config()
        c['cpu_freq_ratio'] = cpu_freq_ratio
        if self.config == c:
            self.config = c
            partition_idx, utility = self.controller.get_partition_point(c['network_bw'], c['overall_thr_weight'],
                                                                    c['mobile_thr_weight'], c['cpu_freq_ratio'],
                                                                    c['server_ratio'])
            if self.partition_idx != partition_idx:
                self.partition_idx = partition_idx
                print('reload')
                # self.load()

        # TODO: return partition layer name to edge
        # self.partition_layer

        return result_np
