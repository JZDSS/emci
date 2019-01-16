import argparse

parser = argparse.ArgumentParser(
    description='Landmark Detection Training')

parser.add_argument('-l', '--learning_rate', default=1e-3)
parser.add_argument('-b', '--batch_size', default=16)
parser.add_argument('-c', '--cuda', default=True)
parser.add_argument('-n', '--n_gpu', default=1)

args = parser.parse_args()

if __name__ == '__main__':
    pass
