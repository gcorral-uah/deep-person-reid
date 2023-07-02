import copy
import os.path as osp
import random
from typing import Optional, Self

from torchreid.data import ImageDataset
from gtxmlparser import parse_all_xml, generate_frames, generate_all_xml_of_dataset


class UAHDataset(ImageDataset):
    dataset_dir = "gba_dataset/"
    dataset_url = None

    # NOTE: I am basing this loader in ilids.py
    def __init__(self, root: str = "", training_test_split: float = 0.5, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)
        self.prepare_dataset(self.dataset_dir)
        self.training_test_split = training_test_split

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # See https://github.com/KaiyangZhou/deep-person-reid/issues/442, for
        # an explanation of the train, query and gallery list purposes.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).

        # Important note:
        # The evaluation expects different cam ids in the gallery results than
        # that one used in the query. My workaround for single cam use is to set
        # all query cam IDs to 0 and all gallery IDs to 1.

        # Call our debugging copies _train, etc as train, etc are overwritten
        # by parent classes constructors.
        self._train, self._query, self._gallery = self.create_splits()
        super(UAHDataset, self).__init__(
            self._train, self._query, self._gallery, **kwargs
        )

    def create_splits(self):
        """Create training, query and gallery splits"""

        # The basic idea here is to take divide the dataset into training and
        # testing based on the apparence of people in images. So for example
        # the subjects with id 1, 3 and go into the training data, and the ones
        # with id 2 and 5 go into testing or validation folds.

        _, pid_dict = parse_all_xml(self.dataset_dir)
        pids = list(pid_dict.keys())
        num_pids = len(pids)
        num_train_pids = int(num_pids * self.training_test_split)
        pids_copy = copy.deepcopy(pids)
        random.shuffle(pids_copy)
        train_pids = pids_copy[:num_train_pids]
        test_pids = pids_copy[num_train_pids:]

        train = []
        query = []
        gallery = []
        camera_train = 0
        camera_query = 0
        camera_gallery = 1

        # for train IDs, all images are used in the train set.
        for pid in train_pids:
            for image_path in pid_dict[pid]:
                image_tuple = (image_path, pid, camera_train)
                train.append(image_tuple)

        # for each test ID, randomly choose two images, one for
        # query and the other one for gallery, until we have only zero or one
        # left.

        for pid in test_pids:
            img_names = set(copy.deepcopy(pid_dict[pid]))
            while (len(img_names)) >= 2:
                query_img = img_names.pop()
                image_query_tuple = (query_img, pid, camera_query)
                query.append(image_query_tuple)

                gallery_img = img_names.pop()
                image_gallery_tuple = (gallery_img, pid, camera_gallery)
                gallery.append(image_gallery_tuple)

        # TODO: Maybe we want to shuffle them?
        return train, query, gallery

    def prepare_dataset(self, path: str):
        generate_frames(path)
        generate_all_xml_of_dataset(path)
