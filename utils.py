import torch.backends.cudnn as cudnn
import numpy as np
import random
import math
import torch
import os
import logging
import sys
import pickle
import time
import datetime
from torchvision import utils as vutils
from tqdm import tqdm
# ================== Pytorch Utils =================
def count_parameters_in_M(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def save_ckpt(state_dict, save_path):
    torch.save(state_dict, save_path)

def requires_grad(model, flag=True):
    params = model.parameters()
    for p in params:
        p.requires_grad = flag

def show_single_batch(x, path, nrow):
    vutils.save_image(x, path, normalize=True, nrow=nrow)

# ================== OS Utils =======================
def overwrite_opt(opt, opt_override):
    for (k, v) in opt_override.items():
        setattr(opt, k, v)
    return opt

def save_args(output_dir, args):
    with open(output_dir + 'config.txt', 'w') as fp:
        for key in args:
            fp.write(
                    ('%s : %s\n' % (key, args[key]))
            )

    with open(output_dir + 'config.pkl', 'wb') as fp:
        pickle.dump(args, fp)

def save_file(output_dir, file_name):
    file_in = open('./' + file_name, 'r')
    file_out = open(output_dir + 'code/' + os.path.basename(file_name), 'w')
    for line in file_in:
        file_out.write(line)

def get_output_dir(file, add_datetime=True, added_directory=None):
    output_dir = './{}/'.format(os.path.splitext(os.path.basename(file))[0])

    if added_directory is not None:
        output_dir += added_directory + '/'

    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if add_datetime:
        output_dir += t + '/'

    return output_dir

class Logger(object):
    def __init__(self, save, rank):
        # other libraries may set logging before arriving at this line.
        # by reloading logging, we can get rid of previous configs set by other libraries.
        from importlib import reload
        reload(logging)
        self.rank = rank

        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        self.start_time = time.time()

    def info(self, string, *args):

        elapsed_time = time.time() - self.start_time
        elapsed_time = time.strftime(
            '(Elapsed: %H:%M:%S) ', time.gmtime(elapsed_time))
        if isinstance(string, str):
            # string = elapsed_time + f" job_id: {self.rank} " + string
            string = elapsed_time + f'\033[32;1m job_id: {self.rank} \033[0m' + string

        else:
            logging.info(elapsed_time)
        #
        logging.info(string, *args)