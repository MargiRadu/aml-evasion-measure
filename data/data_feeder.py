import numpy as np
import os
import gzip
import urllib.request


def get_random_indices(n, lo, hi):
    if n > (hi - lo):
        raise ValueError("The specified range is smaller than the number of DIFFERENT random indices you are requesting.")

    nums = []
    while len(nums) < n:
        r = int((np.random.random(1) * (hi - lo)) + lo)
        if r not in nums:
            nums.append(r)

    if n == 1:
        return nums[0]
    else:
        return nums


def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images * 28 * 28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255)
        data = data.reshape(num_images, 28, 28, 1)
        return data


def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)


class AbstractFeeder:
    def __init__(self):
        self.craft_count = 0

    def _extract_data_by_class(self, source, data_class, total_count='max'):
        if source == 'craft':
            data = self.craft_data
            labels = self.craft_labels
        elif source == 'train':
            data = self.train_data
            labels = self.train_labels
        elif source == 'test':
            data = self.test_data
            labels = self.test_labels
        else:
            raise ValueError("Invalid data source. Valid options are {craft, train, test}.")

        # Extract the indices of all source_class test instances.
        ohe_target = [0.0 for _ in range(10)]
        ohe_target[data_class] = 1.0
        indices = np.where((labels == ohe_target).all(axis=1))[0]

        if total_count == 'max':
            total_count = len(indices)
        elif isinstance(total_count, int):
            if total_count > len(indices):
                total_count = len(indices)
                print("The requested craft data amount ({}) is too large. Using max available data.".format(total_count))
        else:
            total_count = len(indices)
            print("The requested craft data amount ({}) is invalid. Using max available data.".format(total_count))

        indices = indices[:total_count]

        data = data[indices]
        labels = labels[indices]

        return data, labels

    def get_craft_data(self, data_class, total_count='max', metric=None):
        craft_data = self._extract_data_by_class('craft', data_class, 'max')[0]
        if total_count == 'max':
            total_count = len(craft_data)
        if metric is None:
            return craft_data[:total_count]
        else:
            return np.array(sorted(craft_data, key=metric, reverse=True))[:total_count]

    def get_examination_data(self, data_class, total_count='max'):
        return self._extract_data_by_class('train', data_class, total_count)[0]

    def get_target_image(self, data_class, index):
        data, label = self._extract_data_by_class('test', data_class)
        return data[index], label[index]

    def get_random_target_indices(self, data_class, total_count):
        possible_images_count = len(self._extract_data_by_class('test', data_class)[0])
        return get_random_indices(total_count, 0, possible_images_count)

    def get_labels(self, data_class, n):
        return np.repeat([np.eye(self.nb_classes)[data_class]], repeats=n, axis=0)

    def compile_poison_batch(self, poison_data, craft_class, batch_size, shuffle=False):
        clean_data_amount = batch_size - len(poison_data)

        new_count = self.craft_count + clean_data_amount
        adv_labels = np.repeat([np.eye(10)[craft_class]], repeats=len(poison_data), axis=0)

        batch_data = np.append(self.craft_data[self.craft_count:new_count], poison_data, axis=0)
        batch_labels = np.append(self.craft_labels[self.craft_count:new_count], adv_labels, axis=0)

        self.craft_count = new_count

        # Shuffle the batches
        if shuffle:
            p = np.random.permutation(len(batch_data))
            return batch_data[p], batch_labels[p]
        else:
            return batch_data, batch_labels


class MNISTFeederEvasion(AbstractFeeder):
    def __init__(self, path_to_mnist):
        super().__init__()

        self.nb_classes = 10

        self.train_data = extract_data(os.path.join(path_to_mnist, 'train-images-idx3-ubyte.gz'), 60000)
        self.train_labels = extract_labels(os.path.join(path_to_mnist, 'train-labels-idx1-ubyte.gz'), 60000)
        test_data = extract_data(os.path.join(path_to_mnist, 't10k-images-idx3-ubyte.gz'), 10000)
        test_labels = extract_labels(os.path.join(path_to_mnist, 't10k-labels-idx1-ubyte.gz'), 10000)

        # Setup training data split.
        self.splitter = 6000

        self.test_data = test_data[:self.splitter, :, :, :]
        self.test_labels = test_labels[:self.splitter]

        self.craft_data = test_data[self.splitter:, :, :, :]
        self.craft_labels = test_labels[self.splitter:]

        self.test_count = 0

    def get_evasion_craft_data(self, source_class, total_count):
        possible_images = self._extract_data_by_class('craft', source_class)[0]
        possible_images_count = len(possible_images)
        indices = get_random_indices(total_count, 0, possible_images_count)
        return possible_images[indices]


class MNISTFeederPoison(AbstractFeeder):
    def __init__(self, path_to_mnist):
        super().__init__()

        self.nb_classes = 10

        train_data = extract_data(os.path.join(path_to_mnist, 'train-images-idx3-ubyte.gz'), 60000)
        train_labels = extract_labels(os.path.join(path_to_mnist, 'train-labels-idx1-ubyte.gz'), 60000)
        self.test_data = extract_data(os.path.join(path_to_mnist, 't10k-images-idx3-ubyte.gz'), 10000)
        self.test_labels = extract_labels(os.path.join(path_to_mnist, 't10k-labels-idx1-ubyte.gz'), 10000)

        # Setup training data split.
        self.splitter = 50000

        self.train_data = train_data[:self.splitter, :, :, :]
        self.train_labels = train_labels[:self.splitter]

        self.craft_data = train_data[self.splitter:, :, :, :]
        self.craft_labels = train_labels[self.splitter:]

        self.test_count = 0


class CIFARFeeder(AbstractFeeder):
    def __init__(self, path_to_cifar):
        super(CIFARFeeder, self).__init__()
        train_batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        test_batches = ['test_batch']
        self.nb_classes = 10

        def one_hot_encode(label):
            return np.eye(self.nb_classes)[label]

        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            data = map(lambda x: np.transpose(x.reshape(3,32,32), axes=(1, 2, 0)) / 255, dict[b'data'])
            labels = map(one_hot_encode, dict[b'labels'])
            return data, labels

        train_data = []
        train_labels = []
        for file_name in train_batches:
            d, l = unpickle(os.path.join(path_to_cifar, file_name))
            train_data.extend(d)
            train_labels.extend(l)
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)

        self.test_data = []
        self.test_labels = []
        for file_name in test_batches:
            data, labels = unpickle(os.path.join(path_to_cifar, file_name))
            self.test_data.extend(data)
            self.test_labels.extend(labels)
        self.test_data = np.array(self.test_data)
        self.test_labels = np.array(self.test_labels)

        # Setup training data split.
        total_train = len(train_data)
        self.splitter = total_train - 2000

        self.train_data = train_data[:self.splitter, :, :, :]
        self.train_labels = train_labels[:self.splitter]

        self.craft_data = train_data[self.splitter:, :, :, :]
        self.craft_labels = train_labels[self.splitter:]


class PepperCIFARFeeder(CIFARFeeder):
    def __init__(self, path_to_cifar, noise_pct):
        super().__init__(path_to_cifar)

        def pepperize(nd_data, pct):
            mask = (np.random.rand(*nd_data.shape)) > pct
            return nd_data - mask

        self.train_data = pepperize(self.train_data, noise_pct)
        self.test_data = pepperize(self.test_data, noise_pct)
        self.craft_data = pepperize(self.craft_data, noise_pct)


class SharedCIFARFeeder(CIFARFeeder):
    def __init__(self, path_to_cifar, train_seize_pct, test_seize_pct):
        super().__init__(path_to_cifar)

        def split_data(data, labels, split_pct):
            split_index = int(split_pct * len(data))
            return (data[:split_index], labels[:split_index]), (data[split_index:], labels[split_index:])

        # Seize train data
        (big_data, big_labels), (small_data, small_labels) = split_data(self.train_data,
                                                                        self.train_labels,
                                                                        train_seize_pct)

        self.train_data = big_data
        self.train_labels = big_labels

        # Seize test data
        (big_data, big_labels), (small_data, small_labels) = split_data(self.test_data,
                                                                        self.test_labels,
                                                                        test_seize_pct)

        self.train_data = np.append(self.train_data, big_data, axis=0)
        self.train_labels = np.append(self.train_labels, big_labels, axis=0)
        self.test_data = small_data
        self.test_labels = small_labels


class SqueezedCIFARFeeder(CIFARFeeder):
    def __init__(self, path_to_cifar, new_bit_depth):
        super().__init__(path_to_cifar)

        def squeeze(nd_data, bit_depth):
            n_bit_depth = 2**bit_depth
            return (nd_data * n_bit_depth + 0.5).astype(np.int16).astype(np.float64) / n_bit_depth

        self.train_data = squeeze(self.train_data, new_bit_depth)
        self.test_data = squeeze(self.test_data, new_bit_depth)
        self.craft_data = squeeze(self.craft_data, new_bit_depth)
