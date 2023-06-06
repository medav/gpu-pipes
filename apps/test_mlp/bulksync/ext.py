import os
import time
import torch
import common.utils as utils

cur_path = os.path.dirname(os.path.realpath(__file__))
testmlp_cuda = utils.make_ext([f'{cur_path}/test_mlp_bs_ext.cu'])
import testmlp_cuda



