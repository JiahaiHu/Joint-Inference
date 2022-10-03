import os
import pandas as pd
from model_zoo.load_weight_local import load_checkpoint,load_param_into_net,load_checkpoint_from_torch
from mindvision.dataset.download import DownLoad


class LoadPretrainedModel(DownLoad):
    """Load pretrained model from url."""

    def __init__(self, model, url):
        self.model = model
        self.url = url
        self.path = os.path.join('./', self.__class__.__name__)

    def download_checkpoint_from_url(self):
        """Download the checkpoint if it doesn't exist already."""
        os.makedirs(self.path, exist_ok=True)

        # download files

        self.download_url(self.url, path=self.path)

    def load_checkpoint(self):
        """Load checkpoint."""
        self.param_dict = load_checkpoint(os.path.join(self.path, os.path.basename(self.url)))

    def load_checkpoint_from_torch(self):
        self.param_dict = load_checkpoint_from_torch()

    def load_param_into_net(self):
        load_param_into_net(self.model, self.param_dict)

    def run(self):
        """Download checkpoint file and load it."""
        self.download_checkpoint_from_url()

        self.load_checkpoint_from_torch()
        self.load_param_into_net()