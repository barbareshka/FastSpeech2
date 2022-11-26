import os
import json
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import .symbols
from .funcs import get_mask_from_lengths, pad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = "<blank>"
UNK_WORD = "<unk>"
BOS_WORD = "<s>"
EOS_WORD = "</s>"
