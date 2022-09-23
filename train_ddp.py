from os_utils import *
from vis_utils import *
from ddp_utils import *
from torch_utils import *
from datasets import get_dataset

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def DDP(address, port, local_rank, nprocs):
    os.environ['MASTER_ADDR'] = address
    os.environ['MASTER_PORT'] = port
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=nprocs, rank=local_rank)

def main(local_rank, nprocs, args):

    args.local_rank = local_rank

    init_seeds(1)
    DDP(address=args.master_address, port=args.master_port, local_rank=local_rank, nprocs=nprocs)

    logging = Logger(local_rank, args['save'])
    logging.info(args)

    train_dl, valid_dl = get_dataset(dataset=args['dataset'], args=args)

    train(VAE, ebm_list, opt_list, train_queue, local_rank, nprocs, logging, args)
    dist.destroy_process_group()