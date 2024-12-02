import logging
from zipfile import ZipFile
import numpy as np
import contextlib

from ...exceptions import DoesNotExistException
from ...transforms import StaticTransform
from ...utils import convert_static_h5_dataset_to_folder, zip_dir
from ..base import FileTreeDatasetBase

logger = logging.getLogger(__name__)


class FileTreeDataset(FileTreeDatasetBase):
    _transform_types = (StaticTransform,)

    @staticmethod
    def initialize_from(filename, outpath=None, overwrite=False):
        """
        Convenience function. See `convert_static_h5_dataset_to_folder` in `.utils`
        """
        convert_static_h5_dataset_to_folder(filename, outpath=outpath, overwrite=overwrite)

    @property
    def img_shape(self):
        return (1,) + self[0].images.shape

    @property
    def n_neurons(self):
        target_group = "responses" if "responses" in self.data_keys else "targets"
        val = self[0]
        if hasattr(val, target_group):
            val = getattr(val, target_group)
        else:
            val = val[target_group]
        return len(val)

    def change_log(self):
        if (self.basepath / "change.log").exists():
            with open(self.basepath / "change.log", "r") as fid:
                logger.info("".join(fid.readlines()))

    def zip(self, filename=None):
        """
        Zips current dataset.
        Args:
            filename:  Filename for the zip. Directory name + zip by default.
        """

        if filename is None:
            filename = str(self.basepath) + ".zip"
        zip_dir(filename, self.basepath)

    def unzip(self, filename, path):
        logger.info(f"Unzipping {filename} into {path}")
        with ZipFile(filename, "r") as zip_obj:
            zip_obj.extractall(path)

    def add_link(self, attr, new_name):
        """
        Add a new dataset that links to an existing dataset.
        For instance `targets` that links to `responses`
        Args:
            attr:       existing attribute such as `responses`
            new_name:   name of the new attribute reference.
        """
        if not (self.basepath / "data/{}".format(attr)).exists():
            raise DoesNotExistException("Link target does not exist")

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class ShuffledFileTreeDataset(FileTreeDataset):
    def __init__(self, path = None, shuffle_dimensions = None, *args, **kwargs):
        super().__init__(path, *args, **kwargs)
        # check shuffle_dims for which dimension needs to be shuffled
        # crete a permutation of the indices for these keys and store
        self.__shuffle_idx = {}
        self.shuffle_dimensions = shuffle_dimensions
        
        if self.shuffle_dimensions:
            for data_key, seed in self.shuffle_dimensions.items():
                with temp_seed(seed):
                    self.__shuffle_idx[data_key] = np.random.permutation(len(self))
                    print(self.__shuffle_idx)

    def __getitem__(self, item):
        # load data from cache or disk
        ret = []
        for data_key in self.data_keys:
            if self.use_cache and item in self._cache[data_key]:
                ret.append(self._cache[data_key][item])
            else:
                # for the data_key to be shuffled, change the item to permuted item
                tmp_item = self.__shuffle_idx[data_key][item] if data_key in self.__shuffle_idx else item
                if data_key in self.trial_info.keys():
                    val = self.trial_info[data_key][tmp_item : tmp_item + 1] 
                else:
                    datapath = self.resolve_data_path(data_key)
                    val = np.load(datapath / "{}.npy".format(tmp_item))
                if self.use_cache:
                    self._cache[data_key][item] = val
                ret.append(val)    
        
        # create data point and transform
        x = self.data_point(*ret)

        for tr in self.transforms:
            # ensure only specified types of transforms are used
            assert isinstance(tr, self._transform_types)
            x = tr(x)

        # apply output rename if necessary
        if self.rename_output:
            x = self._output_point(*x)

        if self.output_dict:
            x = x._asdict()

        return x
