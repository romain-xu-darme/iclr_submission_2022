import torch
from torchvision.datasets.celeba import CelebA
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, Any, List


class CelebAByAttribute(Dataset):
    """ Subset of CelebA dataset selected by attribute value

    :param root: Root directory where images are downloaded to.
    :param split: One of {'train', 'valid', 'test', 'all'}.
    :param attr_index: Attribute selection index
    :param attr_value: Attribute selection value
    :param transform: A function/transform that takes in a PIL image and returns a transformed version.
    :param target_transform: A function/transform that takes in the target and transforms it.
    :param download: If true, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again.
    """
    def __init__(
            self,
            root: str,
            split: str,
            attr_index: int,
            attr_value: int,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        if split == 'trainval':
            split = 'train'
        if not split:
            split = 'all'
        self.dataset = CelebA(
            root=root,
            split=split,
            target_type=['attr', 'identity'],
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        selection = list(torch.nonzero(self.dataset.attr[:, attr_index] == attr_value).detach().numpy()[:, 0])

        def subset(source: List, index: List) -> List:
            return [source[i] for i in index]

        self.dataset.filename = subset(self.dataset.filename, selection)
        self.dataset.attr = torch.stack(subset(self.dataset.attr, selection))
        self.dataset.identity = torch.stack(subset(self.dataset.identity, selection))
        self.dataset.landmarks_align = torch.stack(subset(self.dataset.landmarks_align, selection))
        self.dataset.bbox = torch.stack(subset(self.dataset.bbox, selection))

    def __getitem__(self, index) -> Tuple[Any, Any]:
        x, label = self.dataset[index]
        # Remove attributes
        return x, label[-1]

    def __len__(self) -> int:
        return len(self.dataset)


def CelebABlurry(
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
) -> Dataset:
    """ Wrapper for the CelebA blurry dataset

    :param root: Root directory where images are downloaded to.
    :param split: One of {'train', 'valid', 'test', 'all'}.
    :param transform: A function/transform that takes in a PIL image and returns a transformed version.
    :param target_transform: A function/transform that takes in the target and transforms it.
    :param download: If true, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again.
    """
    return CelebAByAttribute(root, split, 10, 1, transform, target_transform, download)


def CelebANonBlurry(
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
) -> Dataset:
    """ Wrapper for the CelebA non blurry dataset

    :param root: Root directory where images are downloaded to.
    :param split: One of {'train', 'valid', 'test', 'all'}.
    :param transform: A function/transform that takes in a PIL image and returns a transformed version.
    :param target_transform: A function/transform that takes in the target and transforms it.
    :param download: If true, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again.
    """
    return CelebAByAttribute(root, split, 10, 0, transform, target_transform, download)
