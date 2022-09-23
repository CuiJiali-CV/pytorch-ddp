import os
import logging
import sys
import pickle
import time

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

class Logger(object):
    def __init__(self, save, rank=0):
        # other libraries may set logging before arriving at this line.
        # by reloading logging, we can get rid of previous configs set by other libraries.
        from importlib import reload
        reload(logging)
        self.rank = rank
        if self.rank == 0:
            log_format = '%(asctime)s %(message)s'
            logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                format=log_format, datefmt='%m/%d %I:%M:%S %p')
            fh = logging.FileHandler(os.path.join(save, 'log.txt'))
            fh.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(fh)
            self.start_time = time.time()

    def info(self, string, *args):
        if self.rank == 0:
            elapsed_time = time.time() - self.start_time
            elapsed_time = time.strftime(
                '(Elapsed: %H:%M:%S) ', time.gmtime(elapsed_time))
            if isinstance(string, str):
                string = elapsed_time + string
            else:
                logging.info(elapsed_time)
            logging.info(string, *args)