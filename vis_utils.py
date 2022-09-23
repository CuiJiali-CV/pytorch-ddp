from torchvision import utils as vutils

def show_single_batch(x, path, nrow):
    vutils.save_image(x, path, normalize=True, nrow=nrow)