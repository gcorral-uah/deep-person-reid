import sys
import os
import copy
import os.path as osp
import random
import collections
import glob

from torchreid.data import ImageDataset


class UAHDataset(ImageDataset):
    dataset_dir = "uah_dataset"
    dataset_url = None

    # NOTE: I am basing this loader in ilids.py
    def __init__(self, root="", **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        self.split_path = osp.join(self.dataset_dir, "splits.json")

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

        train = self.gen_train_imgs()
        query = self.gen_query_imgs()
        gallery = self.gen_gallery_imgs()

        super(UAHDataset, self).__init__(train, query, gallery, **kwargs)

    def _create_splits(self, force=False):
        """
        Summary or Description of the Function

        Parameters:
        force (bool): Force the regeneration of the splits.json file
        """

        # If the split already exists, and we don't want to force the
        # regeneration, do nothing
        if osp.exists(self.split_path) and not force:
            return

        paths = glob.glob(osp.join(self.dataset_dir, "*.jpg"))
        img_names = [osp.basename(path) for path in paths]

        # Dict, where the defult value (if the key doesn't exists is a [], and
        # it doesn't throw a KeyNotFound exception)
        pid_dict = collections.defaultdict(list)

        for img_name in img_names:
            pid = int(img_name[:4])
            pid_dict[pid].append(img_name)

        pids = list(pid_dict.keys())
        num_pids = len(pids)
        num_train_pids = int(num_pids * 0.5)
        pids_copy = copy.deepcopy(pids)
        random.shuffle(pids_copy)
        train_pids = pids_copy[:num_train_pids]
        test_pids = pids_copy[num_train_pids:]

        train = []
        query = []
        gallery = []

        # for train IDs, all images are used in the train set.
        for pid in train_pids:
            img_names = pid_dict[pid]
            train.extend(img_names)

        # for each test ID, randomly choose two images, one for
        # query and the other one for gallery.
        for pid in test_pids:
            img_names = pid_dict[pid]
            samples = random.sample(img_names, 2)
            query.append(samples[0])
            gallery.append(samples[1])

        split = {"train": train, "query": query, "gallery": gallery}

    def gen_train_imgs(self):
        return [("", 0, 0)]

    def gen_query_imgs(self):
        return [("", 0, 0)]

    def gen_gallery_imgs(self):
        return [("", 0, 1)]
