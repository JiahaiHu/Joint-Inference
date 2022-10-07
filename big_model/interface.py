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

    def load(self, model_url=""):
        LOG.info("load")
        self.net, self.partition_layer = self.controller.init_model_server(500)

    @staticmethod
    def preprocess(image, input_shape):
        """Preprocess functions in edge model inference"""
        return 0
        
    @staticmethod
    def postprocess(model_output):
        result_np = model_output.asnumpy()
        return result_np.tolist()
        
    def predict(self, data, **kwargs):
        input_np = np.array(data)
        input_feed = ms.Tensor(input_np, ms.float32)
        result = self.net(input_feed)
        result_np = self.postprocess(result)
        return result_np
