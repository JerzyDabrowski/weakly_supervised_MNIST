import copy
import math
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import typing as t
import torchvision.transforms as T
import scipy.stats
from torchvision import datasets


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32


class MNISTDatasetWeak(Dataset):
    def __init__(self, org_data: torch.Tensor, org_labels: torch.Tensor, dataset_size: int,
                 transform: t.Optional[bool] = None):
        self.org_data = org_data
        self.org_labels = org_labels
        self.transform = transform
        self.dataset_size = dataset_size
        self.final_data, self.final_labels = self.prepare_final_data()

    # https://stackoverflow.com/questions/50626710/generating-random-numbers-with-predefined-mean-std-min-and-max
    def create_distribution(self, min_val: float, max_val: float, mean: float, std: float):
        scale = max_val - min_val
        location = min_val
        # Mean and standard deviation of the unscaled beta distribution
        unscaled_mean = (mean - min_val) / scale
        unscaled_var = (std / scale) ** 2
        # Computation of alpha and beta can be derived from mean and variance formulas
        t = unscaled_mean / (1 - unscaled_mean)
        beta = ((t / unscaled_var) - (t * t) - (2 * t) - 1) / ((t * t * t) + (3 * t * t) + (3 * t) + 1)
        alpha = beta * t
        # Not all parameters may produce a valid distribution
        if alpha <= 0 or beta <= 0:
            raise ValueError('Cannot create distribution for the given parameters.')
        # Make scaled beta distribution with computed parameters
        return scipy.stats.beta(alpha, beta, scale=scale, loc=location)

    # If number 4 is in the grid, return 1
    def _get_label_for_grid_image(self, labels: t.List) -> int:
        return 1 if 4 in labels else 0

    def prepare_final_data(self) -> t.Tuple:
        final_list_of_labels = []
        final_list_of_images = []
        distribution = self.create_distribution(3, 30, 10, 3)
        list_of_images_in_the_grid = distribution.rvs(size=self.dataset_size).astype('int')
        data_and_labels = list(zip(self.org_data, self.org_labels))
        copy_of_data = copy.copy(data_and_labels)
        start = 0
        stop = 0
        for idx, n_img in enumerate(list_of_images_in_the_grid):
            stop += n_img
            # start new loop when stop is greater than loop range
            if stop >= len(list_of_images_in_the_grid):
                stop = n_img
                start = 0
            temporary_grid_of_data_and_labels = copy_of_data[start:stop]
            res = [ele[0] for ele in temporary_grid_of_data_and_labels]
            labels = [ele[1] for ele in temporary_grid_of_data_and_labels]
            grid_label = self._get_label_for_grid_image(labels)
            frame = torch.stack(res)
            # set output image size 224x224 for resnet
            target_len = 224
            # Create tensor of zeros for 3 channels image (Resnet input size)
            target_pic = torch.zeros((3, target_len, target_len))
            natural_size_grid_image = torchvision.utils.make_grid(frame.view(frame.size(0), 1, 28, 28), padding=0,
                                                                  nrow=math.ceil(math.sqrt(frame.shape[0])))
            # Determine the padding size
            pad_size_vert = int((target_len - natural_size_grid_image.size(1)) / 2)
            pad_size_horiz = int((target_len - natural_size_grid_image.size(2)) / 2)
            # Normalization of RGB images
            natural_size_grid_image = natural_size_grid_image/255
            target_pic[:, pad_size_vert:target_len - pad_size_vert, pad_size_horiz:target_len - pad_size_horiz] = natural_size_grid_image
            # possibility to change the image size for other models
            transform = T.Resize(int(target_len / 1))
            target_pic = transform(target_pic)
            start = stop
            final_list_of_images.append(target_pic)
            final_list_of_labels.append(grid_label)

        return final_list_of_images, final_list_of_labels

    def __getitem__(self, idx: int):
        return self.final_data[idx], self.final_labels[idx]

    def __len__(self):
        return len(self.final_data)


train_data = datasets.MNIST(
    root='data',
    train=True,
    transform=T.ToTensor(),
    download=True,
)
test_data = datasets.MNIST(
    root='data',
    train=False,
    transform=T.ToTensor()
)

X_train = train_data.data
y_train = train_data.targets

X_test = test_data.data
y_test = test_data.targets


train_dataset = MNISTDatasetWeak(X_train, y_train, 30000)
test_dataset = MNISTDatasetWeak(X_test, y_test, 1000)


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
