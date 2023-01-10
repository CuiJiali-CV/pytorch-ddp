from utils import *
from ddp_utils import *
import importlib
from nets import _netG, _netI
from datasets import get_dataset
import argparse
from torch.distributions import Normal
mse = torch.nn.MSELoss(reduction='none')
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def compute_fid_score(netG, args):
    fid_stat_dir = {
        'cifar10': '/data6/jcui7/nvae/data/fid_stat/cifar10/fid_stats_cifar10_train.npz'
    }[args['dataset']]

    fid_size = 50000
    batch_size = args['batch_size']

    n_batch = int((fid_size // batch_size) // args['nprocs'])

    count = tqdm(range(n_batch)) if args['local_rank'] == 0 else range(n_batch)

    to_range_0_1 = lambda x: (x + 1.) / 2. if args['normalize_data'] else x

    def sample_x():
        z1_p = torch.randn((args['batch_size'], args['z1_dim'])).cuda(args['local_rank'])
        syn = netG(z1_p).detach()
        syn = to_range_0_1(syn).clamp(min=0., max=1.)
        return syn

    s = []
    for _ in count:
        syn = sample_x()
        s.append(syn)

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

def train_step(netG, netI, optG, optI, train_dl, logging, args):
    Broken = False
    log_iter = int(len(train_dl) // 4) if len(train_dl) > 1000 else int(len(train_dl) // 2)

    netG.train()
    netI.train()
    requires_grad(netG, True)
    requires_grad(netI, True)

    for b, x in enumerate(train_dl):
        x = x[0] if len(x) > 1 else x
        x = x.cuda(args['local_rank'])
        batch_size = x.shape[0]

        if b % 100 == 0:  # just in case, maybe useless.
            average_params(netG.parameters(), True)
            average_params(netI.parameters(), True)

        optG.zero_grad()
        optI.zero_grad()

        z1_q_mu, z1_q_sig = netI(x)
        q_dist = Normal(z1_q_mu, z1_q_sig)
        z1_q = q_dist.rsample()
        x_rec = netG(z1_q)
        p_dist = Normal(torch.zeros_like(z1_q), torch.ones_like(z1_q))

        kl = (q_dist.log_prob(z1_q) - p_dist.log_prob(z1_q)).sum() / batch_size
        rec = mse(x, x_rec).sum() / batch_size
        loss = rec + kl

        if True in check_nan(loss, args['nprocs']):
            Broken = True
            print(f"local_rank: {args['local_rank']} || loss: {loss:.3f} recon loss : {rec:.3f} || kl : {kl:.3f}")
            return Broken

        loss.backward()
        dist.barrier()

        average_gradients(netG.parameters(), args['distributed'])
        average_gradients(netI.parameters(), args['distributed'])

        optG.step()
        optI.step()

        if b % log_iter == 0:
            info = f"======= batch {b}/{len(train_dl)} ========\n"
            info += f"{'recon loss':<20}: {rec.item():<20.2f} {'kl':<20}: {kl.item():<20.2f}"
            logging.info(info)

    netG.eval(), netI.eval(),
    requires_grad(netG, False)
    requires_grad(netI, False)

    return Broken

def train(netG, netI, optG, optI, train_dl, logging, args):

    global_step = 0
    fid_best, fid_best_ep = math.inf, 0
    img_dir = args['save'] + f'imgs/'
    ckpt_dir = args['save'] + 'ckpt/'
    fix_x = None
    if args['local_rank'] == 0:
        fix_x = next(iter(train_dl))
        fix_x = fix_x[0].cuda(args['local_rank']) if len(fix_x) > 1 else fix_x.cuda(args['local_rank'])

    for ep in range(args['epochs']):

        logging.info("=="*15 + f" epoch {ep}/{args['epochs']} best fid: {fid_best:.4f} at epoch {fid_best_ep}")

        train_dl.sampler.set_epoch(global_step)

        Broken = train_step(netG, netI, optG, optI, train_dl, logging, args)

        if Broken: return

        if args['local_rank'] == 0 and ep % args['vis_iter'] == 0:

            z1_syn = torch.randn((args['batch_size'], args['z1_dim'])).cuda(args['local_rank'])
            syn = netG(z1_syn)
            show_single_batch(syn,img_dir + f'syn-random-{ep:>07d}.png', nrow=10)

            z1_q_mu, z1_q_sig = netI(fix_x)
            q_dist = Normal(z1_q_mu, z1_q_sig)
            z1_q = q_dist.sample()
            x_rec = netG(z1_q)
            show_single_batch(x_rec, img_dir + f'rec-{ep:>07d}.png', nrow=10)

        if ep % args['n_metric'] == 0 and ep >= args['n_start']:
            if args['compute_fid']:
                fid = compute_fid_score(netG, args)

                if args['local_rank'] == 0:
                    logging.info(f"FID : {fid:.5f}")
                    if fid < fid_best:
                        fid_best = fid
                        fid_best_ep = ep
                        state_dict = {
                            'netG': netG.module.state_dict(),
                            'netI': netI.module.state_dict()
                        }
                        save_ckpt(state_dict, ckpt_dir + f'{fid_best:.5f}_{ep}.pth')

                dist.barrier()  # wait rank


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

def init_net_and_opt(args):
    netG = _netG(args['z1_dim']).cuda(args['local_rank'])
    netG = DistributedDataParallel(netG, device_ids=[args['local_rank']])
    average_params(netG.parameters(), is_distributed=True)  # syn params just in case

    netI = _netI(args['z1_dim']).cuda(args['local_rank'])
    netI = DistributedDataParallel(netI, device_ids=[args['local_rank']])
    average_params(netI.parameters(), is_distributed=True)  # syn params

    optG = torch.optim.Adam(netG.parameters(), lr=args['g_lr'])
    optI = torch.optim.Adam(netI.parameters(), lr=args['i_lr'])

    return netG, netI, optG, optI

def DDP(address, port, local_rank, nprocs):
    os.environ['MASTER_ADDR'] = address
    os.environ['MASTER_PORT'] = port
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=nprocs, rank=local_rank)

def main(local_rank, nprocs, args):

    args['local_rank'] = local_rank

    init_seeds(args['seed'])
    DDP(address=args['master_address'], port=args['master_port'], local_rank=local_rank, nprocs=nprocs)

    logging = Logger(args['save'], local_rank)
    logging.info(args)

    # =========== DATA ==========
    logging.info("Setting up dataset")
    train_dl, valid_dl = get_dataset(dataset=args['dataset'], args=args)
    logging.info(f"Training samples {len(train_dl)}, Testing samples {len(valid_dl)}")
    # =========== NET ===========
    logging.info("Building networks")
    netG, netI, optG, optI = init_net_and_opt(args)
    logging.info(f"netG has parameters {count_parameters_in_M(netG)} M")
    logging.info(f"netI has parameters {count_parameters_in_M(netI)} M")

    train(netG, netI, optG, optI, train_dl, logging, args)
    dist.destroy_process_group()

if __name__ == '__main__':
    def check_args(args):
        args['distributed'] = True
        args['save'] = get_output_dir(__file__, add_datetime=True)
        return args

    def copy(args):
        os.makedirs(args['save'], exist_ok=True)
        for f in ['ckpt', 'imgs', 'code']:
            os.makedirs(args['save'] + f'{f}/', exist_ok=True)

        for f in ['nets.py', 'ddp_utils.py', 'datasets.py', os.path.basename(__file__)]:
            save_file(args['save'], f)

        save_args(args['save'], args)


    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument("--dataset", type=str, default='cifar10')
    parser.add_argument('--batch_size', default=100, type=int)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--vis_iter", type=int, default=1)

    parser.add_argument("--compute_fid", type=bool, default=True)
    parser.add_argument("--n_start", type=int, default=0)
    parser.add_argument("--n_metric", type=int, default=5)

    parser.add_argument("--g_lr", type=float, default=3e-4)
    parser.add_argument("--i_lr", type=float, default=3e-4)

    parser.add_argument("--z1_dim", type=int, default=128)

    # ================= DDP ===================
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--master_address', type=str, default='127.0.0.1', help='address for master')
    parser.add_argument('--master_port', type=str, default='6021', help='port for master')
    parser.add_argument('--nprocs', type=int, default=2, help='number of gpus')

    args = parser.parse_args()
    config = importlib.import_module(f'config.{args.dataset}_fixed')
    args = overwrite_opt(args, config)
    args = vars(args)

    args = check_args(args)

    copy(args)

    mp.spawn(main, nprocs=args['nprocs'], args=(args['nprocs'], args))