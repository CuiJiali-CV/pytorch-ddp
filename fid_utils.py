from ddp_utils import *
from torch_utils import *
from vis_utils import *

def get_fid(sample_fn, args):
    fid_stat_dir = {
        'cifar10': '/data6/jcui7/nvae/data/fid_stat/cifar10/fid_stats_cifar10_train.npz'

    }[args['dataset']]

    fid_size = 50000
    batch_size = args['batch_size']

    n_batch = int((fid_size // batch_size) // args['nprocs'])

    if args['local_rank'] == 0:  # print rank 0 progress
        count = tqdm(range(n_batch))
    else:
        count = range(n_batch)

    to_range_0_1 = lambda x: (x + 1.) / 2. if args['normalize_data'] else x

    s = []
    for _ in count:
        sample = to_range_0_1(sample_fn()).clamp(min=0., max=1.)
        s.append(sample)

    s = torch.cat(s)

    dist.barrier()

    s_gather = gather(s, args['nprocs'])

    if args['local_rank'] == 0:
        from pytorch_fid_jcui7.fid_score import compute_fid
        s = torch.cat(s_gather, dim=0)
        fid = compute_fid(x_train=None, x_samples=s, path=fid_stat_dir)
        return fid
    else:
        return math.inf
