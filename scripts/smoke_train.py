import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # disable GPU for speed/stability

import matplotlib
matplotlib.use('Agg')  # headless plotting

from src.data import get_data_loaders
from src.model import MyModel
from src.optimization import get_optimizer, get_loss
from src.train import optimize


def main():
    loaders = get_data_loaders(batch_size=8, limit=64, valid_size=0.25, num_workers=0)
    model = MyModel(num_classes=50)
    optimizer = get_optimizer(model)
    loss = get_loss()
    optimize(loaders, model, optimizer, loss, n_epochs=1, save_path='checkpoints/smoke.pt', interactive_tracking=True)
    print('SMOKE DONE')


if __name__ == '__main__':
    main()
