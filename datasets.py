import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
def get_dataset(dataset, args):
    # data_dir = '/data4/jcui7/images/data/'
    # data_dir = '/home/CAMPUS/jcui7/HugeData/'
    data_dir = '/data4/jcui7/images/data/' if 'CAMPUS' not in __file__ else '/home/CAMPUS/jcui7/HugeData/'

    img_size = args['img_size']
    normalize = args['normalize_data']
    batch_size = args['batch_size']
    if dataset == 'cifar10':
        if normalize:
            transform = transforms.Compose(
                [transforms.Resize(img_size), transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])

        train_ds = torchvision.datasets.CIFAR10(data_dir + 'cifar10', download=True, train=True, transform=transform)
        valid_ds = torchvision.datasets.CIFAR10(data_dir + 'cifar10', download=True, train=False, transform=transform)

    else:
        raise Exception("choose dataset")

    if args['distributed']:
        train_sampler = DistributedSampler(train_ds)
        valid_sampler = DistributedSampler(valid_ds)
        print("using train_sampler")

    else:
        train_sampler, valid_sampler = None, None

    train_dl = DataLoader(train_ds, batch_size=batch_size,
                    shuffle=(train_sampler is None),
                    sampler=train_sampler,
                    pin_memory=True,
                    num_workers=0,
                    drop_last=True)

    valid_dl = DataLoader(valid_ds, batch_size=batch_size,
                    shuffle=(train_sampler is None),
                    sampler=valid_sampler,
                    pin_memory=True,
                    num_workers=0,
                    drop_last=False)

    return train_dl, valid_dl


