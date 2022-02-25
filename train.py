
from absl import app
from absl import flags

import torch
import torchvision


from imageclassification import models
from imageclassification import dataset
from imageclassification import utils

FLAGS = flags.FLAGS

utils.define_flags()

def train_step():
    pass