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
from skimage.transform import resize
#from model import controller as code2space
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
        self.net, self.partition_layer = self.controller.init_model(500)

    @staticmethod
    def preprocess(image, input_shape):
        """Preprocess functions in edge model inference"""
        return 0
        
    @staticmethod
    def postprocess(model_output):
        result_np = model_output.asnumpy()
        return 0
        
    def predict(self, data, **kwargs):
        input_np = resize(data, (224, 224), anti_aliasing=True).transpose((2, 0, 1))
        input_feed = ms.Tensor(np.expand_dims(input_np / np.max(input_np), axis=0), ms.float32)
        result = self.net(input_feed)
        return result
