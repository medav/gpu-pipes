import os
import time
import torch
from common import utils

cur_path = os.path.dirname(os.path.realpath(__file__))
cuda_ext = utils.make_ext('cuda_ext', [f'{cur_path}/mgn_fullmlp_bs_ext.cu'])
import cuda_ext



