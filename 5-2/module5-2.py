import argparse
from mnist_data import MnistData

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MnistData class")
    parser.add_argument('dataset_type', type=str, help='train or test')
    parser.add_argument('index', type=int, help='Index of the image to display')
    
    args = parser.parse_args()

    mnist_data = MnistData()
    (train_images, train_labels), (test_images, test_labels) = mnist_data.load()

    if args.dataset_type == 'train':
        print(f"Label: {train_labels[args.index]}")
        mnist_data.show_image('train', args.index)
    elif args.dataset_type == 'test':
        print(f"Label: {test_labels[args.index]}")
        mnist_data.show_image('test', args.index)
