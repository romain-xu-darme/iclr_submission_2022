from torch.utils.data import Dataset, Subset
from torchvision.datasets.folder import (
    has_file_allowed_extension,
    IMG_EXTENSIONS,
)
from PIL import Image
from pathlib import Path
import os
import tarfile
import numpy as np
from typing import Any, Callable, List, Optional, Tuple, BinaryIO
from bisect import bisect_left
from io import BytesIO
import copy


def _pil_binary_loader(buff: BinaryIO) -> Image.Image:
    """ Load PIL image for buffer

    :param buff: Input buffer
    :return: PIL image
    """
    img = Image.open(buff)
    return img.convert('RGB')


class EntryInfo:
    """  Class for storing all information regarding an entry in a TAR archive.
    Some fields are directly extracted from a list of TarInfo objects while with
    some additional information is related to database management (image index)

    :param  path: Entry directory path
    :param tar_infos: List of images TAR infos
    :param index: Entry index
    """

    def __init__(self,
                 path: Optional[str] = "",
                 tar_infos: Optional[List[tarfile.TarInfo]] = None,
                 index: Optional[int] = 0,
                 ) -> None:
        if tar_infos is None:
            tar_infos = []
        self.path = path
        self.nviews = len(tar_infos)
        self.offsets = [info.offset_data for info in tar_infos]
        self.sizes = [info.size for info in tar_infos]
        self.index = index

    def serialize(self) -> List:
        """ Serialize object

        :return:
            Concatenation of object attributes
        """
        return [self.index, self.nviews] + self.offsets + self.sizes

    def deserialize(self, data: np.array) -> np.array:
        """ Deserialize entry and return unused values

        :param data: Serialized entry
        :return:
            Unused values in data
        """
        self.index = int(data[0])
        self.nviews = int(data[1])
        offset = 2
        self.offsets = [int(o) for o in data[offset:offset + self.nviews]]
        offset += self.nviews
        self.sizes = [int(s) for s in data[offset:offset + self.nviews]]
        offset += self.nviews
        return data[offset:]


def _create_entries(
        tar_infos: List[tarfile.TarInfo],
        order: List[Tuple[int, str]],
) -> List[EntryInfo]:
    """ Create entry infos with respect to a given order
    :param tar_infos: List of TarInfo for all images
    :param order: List of pairs (index,path)
    :return: List of entries
    """
    # Sort tarinfos wrt to file path
    tar_infos.sort(key=lambda t: t.name, reverse=False)
    # Extract entry names for binary search
    tar_names = [t.name for t in tar_infos]
    entry_infos = []
    for index, path in order:
        location = bisect_left(tar_names, path)
        # List all images related to path
        infos = []
        while location != len(tar_names) \
                and (tar_names[location].startswith(path)):
            infos.append(tar_infos[location])
            tar_names.pop(location)
            tar_infos.pop(location)
        if not infos:
            raise KeyError('Could not find index for image ' + path)
        entry_infos.append(EntryInfo(path, infos, index))
    return entry_infos


def _sort_img_auto(
        tar_infos: List[tarfile.TarInfo],
) -> List[EntryInfo]:
    """ Sort images with respect to their respective directory

    :param tar_infos: List of TarInfo for all images
    :return: Sorted list
    """
    # Sort tarinfos wrt to file path (speeds up finding directory names
    tar_infos.sort(key=lambda t: t.name, reverse=False)
    # Recover name of directories
    dirnames = list(set([os.path.dirname(t.name) for t in tar_infos]))
    dirnames.sort()
    return _create_entries(tar_infos, enumerate(dirnames))


def _sort_img_with_file(
        tar_infos: List[tarfile.TarInfo],
        fp: BinaryIO,
        decode: Optional[bool] = False
) -> List[EntryInfo]:
    """ Sort images according to index file

    :param tar_infos: List of TarInfo for all images
    :param fp: File pointer to index file
    :param decode: Decode index file content
    :return: List of entries
    """
    lines = fp.read().splitlines()
    if decode:
        # Decode index file
        lines = [line.decode('utf-8') for line in lines]

    # Sort lines wrt index
    lines.sort(key=lambda l: int(l.split()[0]), reverse=False)
    order = [(int(line.split()[0]), line.split()[1]) for line in lines]
    return _create_entries(tar_infos, order)


def _get_entries_from_tar(
        path: str,
        root: Optional[str] = '',
        index_file: Optional[str] = '',
        is_valid_file: Optional[Callable[[str], bool]] = None,
        mode: Optional[str] = 'single',
) -> List[EntryInfo]:
    """
    Open TAR file and returns a list of EntryInfo corresponding to the target images

    :param path: Path to TAR archive
    :param root: Root directory path for images inside archive
    :param index_file: Path to file associating each image with an index
    :param is_valid_file: A function that takes path of a file \
        and check if the file is a valid file (used to check of corrupt files)
    :return:
        List of entries
    """
    def is_valid_file_func(x):
        return has_file_allowed_extension(x, IMG_EXTENSIONS)

    if is_valid_file is None:
        is_valid_file = is_valid_file_func
    # Open TAR file (uncompressed only)
    tar = tarfile.open(path, mode='r:')
    members = copy.deepcopy(tar.getmembers())

    # Select files in root directory
    if root:
        members = [m for m in members
                   if os.path.dirname(m.name).startswith(root)]
        if len(members) == 0:
            raise ValueError(f'Could not find files in archive starting with {root}. '
                             f'Examples of file paths {[m.name for m in tar.getmembers()[:5]]}')
        # Remove root from members' names
        for m in members:
            m.name = str(Path(m.name).relative_to(root))

    # Check file extensions
    members = [m for m in members if is_valid_file(m.name)]
    # Use index file to sort images
    if index_file:
        # Open index file
        if index_file.startswith(path):
            # Index file is inside the TAR archive and given in the form
            # path/to/archive.tar/path/to/index_file
            index_fp = tar.extractfile(index_file[len(path) + 1:])
            # Parse index file and sort images
            entry_infos = _sort_img_with_file(members, index_fp, decode=True)
        else:
            # Index is a simple file outside the TAR archive
            index_fp = open(index_file, 'r')
            # Parse index file and sort images
            entry_infos = _sort_img_with_file(members, index_fp, decode=False)
    else:
        if mode == 'single':
            entry_infos = [EntryInfo(m.name, [m], i) for i, m in enumerate(members)]
        else:
            entry_infos = _sort_img_auto(members)

    tar.close()
    return entry_infos


def _find_target_from_file(
        apath: str,
        entry_infos: List[EntryInfo],
        lpath: str,
        ipath: Optional[str] = '',
) -> np.array:
    """
    Find targets from file assuming the following content:

        image_index_0 list_of_target information

        image_index_1 list_of_target information

        ...

    Note: Images may have multiple targets described on several lines, e.g.

        25 first target_information

        25 second target_information

        26 first target_information

        26 second target_information

    Then, for each entry of the dataset, returns:

        - the offset of each image in the TAR archive
        - the size of each image
        - its list of targets


    :param apath: Path to TAR archive
    :param entry_infos: List of EntryInfo corresponding to the target images
    :param lpath: Path to file containing image targets
    :param ipath: Path to output file for index database
    :return: Numpy array
    """
    # Open the target file and read content
    if lpath.startswith(apath):
        # Target file is inside the TAR archive and given in the form
        # path/to/archive.tar/path/to/target_file
        tar = tarfile.open(apath, mode='r:')
        target_fp = tar.extractfile(str(Path(lpath).relative_to(apath)))
        lines = target_fp.read().splitlines()
        # Decode binary strings
        lines = [line.decode('utf-8') for line in lines]
    else:
        # Index is a simple file outside the TAR archive
        target_fp = open(lpath, 'r')
        lines = target_fp.read().splitlines()

    # Keep track of entry indices
    entry_index = [einfo.index for einfo in entry_infos]

    # Init table
    data = []
    for einfo in entry_infos:
        np_data = einfo.serialize()
        data.append(np_data)

    # Read all targets
    for line in lines:
        eidx = int(line.split()[0])
        target = [np.float64(m) for m in line.split()[1:]]
        # Find correct data index
        didx = bisect_left(entry_index, eidx)
        data[didx] += target

    # Convert to numpy array
    data = np.array(data)
    if ipath:
        np.save(ipath, data, allow_pickle=True)
    return data


def _find_classes_from_directories(
        entry_infos: List[EntryInfo],
        ipath: Optional[str] = '',
) -> np.array:
    """
    Find classes assuming the following directory tree:

        class0/xxx.png

        class0/yyy.png

        class0/subdir/zzz.png

        ...

        classN/ttt.png

    Note: all images images included in subdirectories of a class directory
    are considered part of the class.

    Then, for each image of the dataset, returns:

        - its offset in the TAR archive
        - its size
        - its class index

    :param entry_infos: List of EntryInfo corresponding to the target images
    :param ipath: Path to output file for index database
    :return: Numpy array
    """

    def get_class_name(path: str) -> str:
        """ Return base directory name """
        return Path(path).parts[0]

    # Find classes from directories
    classes = list(set([get_class_name(e.path) for e in entry_infos]))
    classes.sort()
    classes_dict = {name: i for i, name in enumerate(classes)}

    data = []
    for i, e in enumerate(entry_infos):
        class_idx = [classes_dict[get_class_name(e.path)]]
        data.append(e.serialize() + class_idx)
    data = np.array(data)
    if ipath:
        np.save(ipath, data, allow_pickle=True)
    return data


class MultiImageArchive(Dataset):
    """ Dataloader operating on the content of a TAR archive.
    Each entry is either:

    - a single image (``single`` mode). Images are arranged in this way by default: ::

        root/
        ├── class_x
        │   ├── zzz.ext
        │   ├── aaa.ext
        │   ├── ...
        │   └── ddd.ext
        ├── ...
        └── class_y
            ├── uuu.ext
            ├── ...
            └── vvv.ext

    - a set of n images (``multi`` mode) corresponding to different viewpoints of the \
    same object. Images are arranged in this way by default: ::

        root/
        ├── class_x
        │   ├── xxx
        │   │   ├── P1.ext
        │   │   ├── P2.ext
        │   │   ├── ...
        │   │   └── Pn.ext
        │   ├── ...
        │   │   └── ...
        │   └── xxy
        │       ├── P1.ext
        │       ├── ...
        │       └── Pn.ext
        ├── ...
        └── class_y
            └── xxx
                ├── P1.ext
                ├── ...
                └── Pn.ext

    :param apath: Path to TAR archive
    :param mode: ``single`` or ``multi``
    :param root: Root path inside TAR archive
    :param transform: A functionthat takes in an PIL image and returns a transformed \
            version.
    :param target_transform: A function that takes in the target and transforms it.
    :param loader: A function to load an image given its path.
    :param is_valid_file: A function that takes path of an Image file and check if the  \
            file is a valid file (used to check of corrupt files)
    :param ipath: Path to index file
    :param index_file: Path to file containing images indices
    :param target_file: Path to file containing images targets
    :param index_transform: A function that takes in an PIL image and the image index \
            and returns a transformed version (used to apply index based transforms such \
            as bounding box or segmentation)
    :param data_in_memory: Load content of TAR archive in memory
    :param open_after_fork: When used in a multiprocess context, this option indicates \
            that the TAR file should not be opened yet but rather after processes are \
            spawned (using worker_open_archive method)
    :param overwrite_index: Overwrite index file if present
    """

    def __init__(
            self,
            apath: str,
            mode: Optional[str] = 'single',
            root: Optional[str] = '',
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[BinaryIO], Any] = _pil_binary_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            ipath: Optional[str] = '',
            index_file: Optional[str] = '',
            target_file: Optional[str] = '',
            index_transform: Optional[Callable] = None,
            data_in_memory: Optional[bool] = False,
            open_after_fork: Optional[bool] = False,
            overwrite_index: Optional[bool] = False,
    ):
        if mode not in ['single', 'multi']:
            raise ValueError(
                f"Unsupported mode {mode}. Should be 'single' or 'multi'.")

        # Using option target_file requires to specify image indices
        if target_file and not index_file:
            raise ValueError(
                'Specifying target file requires to also specify index file')
        self.transform = transform
        self.index_transform = index_transform
        self.target_transform = target_transform
        self.loader = loader
        ############################

        if data_in_memory and open_after_fork:
            print(f'[{self.__class__.__name__}] Ignoring open_after_fork option \
                    (archive loaded in memory)')
            open_after_fork = False

        self.data_in_memory = data_in_memory
        self.open_after_fork = open_after_fork

        if data_in_memory:
            # Load entire archive into memory
            self.data = Path(apath).read_bytes()
        elif not open_after_fork:
            # Open archive. This is safe only in a single process context
            self.afile = open(apath, 'rb')
        else:
            # Store only path to file
            self.afile = None
            self.apath = apath

        ############################

        if not ipath or not (Path(ipath).exists()) or overwrite_index:
            # Get list of TAR infos corresponding to all images of the dataset
            members = _get_entries_from_tar(apath, root, index_file, is_valid_file, mode)

            if target_file:
                self.index_table = _find_target_from_file(apath, members, target_file, ipath)
            else:
                self.index_table = _find_classes_from_directories(members, ipath)

        else:
            self.index_table = np.load(ipath, allow_pickle=True)

    def __len__(self) -> int:
        """
        :return: Number of images/batches in the database
        """
        return self.index_table.shape[0]

    def __getitem__(self, index: int) -> Tuple[List[Any], Any]:
        """
        :param index: Entry index
        :return: Tuple ([images],target)
        """
        entry = EntryInfo()
        target = entry.deserialize(self.index_table[index])
        if self.data_in_memory:
            items = [BytesIO(self.data[offset:offset + size])
                     for offset, size in zip(entry.offsets, entry.sizes)]
        else:
            if self.afile is None:
                raise LookupError(f'[{self.__class__.__name__}] '
                                  f'Archive file {self.afile} not opened yet.'
                                  'This may happen when using open_after_fork option but forgetting '
                                  'to call worker_open_archive in each worker in a multiprocess '
                                  'context.')
            items = []
            for offset, size in zip(entry.offsets, entry.sizes):
                # Move file pointer to offset
                self.afile.seek(offset)
                # Read item
                items.append(BytesIO(self.afile.read(size)))
                self.afile.seek(0)
        items = [self.loader(item) for item in items]
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.index_transform is not None:
            items = [self.index_transform(item, entry.index) for item in items]
        if self.transform is not None:
            items = [self.transform(item) for item in items]
        # Handle "single" mode
        if len(items) == 1:
            items = items[0]
        # Handle single target
        if isinstance(target, list) and len(target) == 1:
            target = target[0]
        return items, target

    def entry_size(self) -> int:
        """
        Return number of views per entry
        """
        entry = EntryInfo()
        entry.deserialize(self.index_table[0])
        return entry.nviews

    def worker_open_archive(self, wid: Optional[int] = 0) -> None:
        """
        Explicitely open archive file (used in multiprocess context)

        :param wid: Unused
        """
        if not self.open_after_fork or self.data_in_memory:
            return
        if not self.afile:
            self.afile = open(self.apath, 'rb')


def _parse_file(
        filename: str,
        process: Optional[Callable[[str], str]] = None
) -> List[str]:
    """ Given a file and a processing function, return a list of processed lines

    :param filename: Path to file
    :param process: Processing function
    :return: List of processed lines
    """
    with open(filename, 'r') as fin:
        lines = fin.read().splitlines()
        if process is not None:
            lines = [process(line) for line in lines]
        return lines


def MultiImageArchive_build_from_config(
        config_file: str,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
) -> Dataset:
    """ Build dataset out of a configuration file.

    The configuration file is a plaintext file with following mandatory fields:

    - ``apath``: <path_to_tar_image>
    - ``mode`` : ``single``/``multi``

    List of optional fields:

    - ``root``: <root_directory_inside_archive>
    - ``ipath``: <path_to_tar_index_file>
    - ``index_file``: <path_to_list_of_entries>
    - ``target_file``: <path_to_target_file>
    - ``train``: <path_to_train_split>
    - ``test``: <path_to_test_split>
    - ``val``: <path_to_val_split>

    :param config_file: Path to configuration file
    :param split: Dataset split
    :param transform: A function that takes in an PIL image and returns \
            a transformed version.
    :param target_transform: A function that takes in the target and transforms it.
    :return: MultiImageArchive dataset
    """
    root = os.path.dirname(config_file)
    cpath = os.path.basename(config_file)

    def check_file(name) -> str:
        """ Check file existence, then return full path """
        fpath = os.path.join(root, name)
        if not os.path.isfile(fpath):
            raise ValueError(f'Could not find {name} in directory {root}')
        return fpath

    # Build configuration
    mandatory = ['apath', 'mode']
    if split:
        mandatory.append(split)
    non_mandatory = ['root', 'ipath', 'index_file', 'target_file']
    config = {}
    for n in non_mandatory:
        config[n] = None
    infos = _parse_file(check_file(cpath), lambda l: l.split(':'))
    for key, value in infos:
        config[key.strip()] = value.strip()
    for n in ['ipath', 'index_file', 'target_file']:
        if config[n] is not None:
            config[n] = os.path.join(root, config[n])
    # Check mandatory fields
    for name in mandatory:
        if name not in config.keys():
            raise ValueError(f'{name} missing from config file {cpath}')

    dataset = MultiImageArchive(
        apath=os.path.join(root, config['apath']),
        mode=config['mode'],
        root=config['root'],
        transform=transform,
        target_transform=target_transform,
        ipath=config['ipath'],
        index_file=config['index_file'],
        target_file=config['target_file'],
        data_in_memory=True,
    )
    if split:
        # Split file contains lines in the form
        # <index> <keep>
        spath = os.path.join(root, config[split])
        split_mask = np.array(_parse_file(spath, lambda l: int(l.split()[1])))
        dataset = Subset(dataset, np.where(split_mask == 1)[0])
    return dataset
