import numpy as np
import pickle as pkl
from utils import batch_generator
from tensorflow.examples.tutorials.mnist import input_data


class DataGen(object):
    gen_source_batch = None
    gen_target_batch = None
    domain_labels = None
    mnist_test = None
    mnistm_test = None
    test_labels = None
    combined_test_labels = None
    test_imgs_m = None
    test_imgs_mm = None
    combined_test_domain = None
    mnist_mean = None
    mnistm_mean = None

    def __init__(self, config):
        print("Initializing data model...")
        self.config = config
        self.batch_size = config.train.batch_size
        self._load_data()

    def _load_data(self):
        print("Loading data...")
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        # Process MNIST
        mnist_train = (mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.uint8) * 255
        mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
        mnist_test = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
        self.mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)

        self.mnist_mean = mnist_train.mean((0, 1, 2))
        # Load MNIST-M
        mnistm = pkl.load(open('mnistm_data.pkl', 'rb'))
        mnistm_train = mnistm['train']
        self.mnistm_test = mnistm['test']

        # for normalization
        self.mnistm_mean = mnistm_train.mean((0, 1, 2))

        # Create a mixed dataset for TSNE visualization
        num_test = 500
        self.test_imgs_m = self.mnist_test[:num_test]
        self.test_imgs_mm = self.mnistm_test[:num_test]
        self.test_labels = mnist.test.labels 
        self.combined_test_labels = np.vstack([mnist.test.labels, mnist.test.labels])
        self.combined_test_domain = np.vstack([np.tile([1., 0.], [num_test, 1]),
                                          np.tile([0., 1.], [num_test, 1])])

        # Batch generators
        self.gen_source_batch = batch_generator(
            [mnist_train, mnist.train.labels], self.batch_size // 2)
        self.gen_target_batch = batch_generator(
            [mnistm_train, mnist.train.labels], self.batch_size // 2)
        self.domain_labels = np.vstack([np.tile([1., 0.], [self.batch_size // 2, 1]),
                                   np.tile([0., 1.], [self.batch_size // 2, 1])])

    def get_data_stats(self):
        print("Getting data stats...")
        return [self.mnist_mean, self.mnistm_mean]
