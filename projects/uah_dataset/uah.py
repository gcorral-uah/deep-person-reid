from torchreid.data import ImageDataset

import copy
import random
import os.path as osp
from typing import Optional
from gtxmlparser import (
    YOLO_IOU_THRESHOLD,
    YOLO_DETECTON_THRESHOLD,
    parse_all_xml,
    generate_frames,
    generate_all_xml_of_dataset,
)


class UAHDataset(ImageDataset):
    dataset_dir = "gba_dataset/"
    dataset_url = None

    # NOTE: This is based after the loader in ilids.py
    def __init__(
        self,
        root: str = "",
        crop_images: bool = True,
        use_yolo_for_testing: bool = True,
        yolo_threshold: Optional[int] = None,
        iou_threshold: Optional[int] = None,
        training_test_split: float = 0.5,
        shuffle_train_test_pids: bool = False,
        **kwargs,
    ):
        self.old_new_label_dict: Optional[dict[int, int]] = None
        self.pid_dict_before_yolo: Optional[dict[int, list[str]]] = None

        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)
        self.prepare_dataset(self.dataset_dir)
        self.training_test_split = training_test_split
        self.shuffle_train_test_pids = shuffle_train_test_pids
        self.crop_images = crop_images
        self.use_yolo_for_testing = use_yolo_for_testing
        self.yolo_threshold = (
            yolo_threshold if yolo_threshold is not None else YOLO_DETECTON_THRESHOLD
        )
        self.yolo_iou_threshold = (
            iou_threshold if iou_threshold is not None else YOLO_IOU_THRESHOLD
        )

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

        # The training-test split is a float between 0 and 1.
        assert self.training_test_split > 0.0
        assert self.training_test_split < 1.0

        # The basic idea here is to take divide the dataset into training and
        # testing based on the apparence of people in images. So for example
        # the subjects with id 1, 3 and go into the training data, and the ones
        # with id 2 and 5 go into testing or validation folds.

        # If we use YOLO we have to do 2 passes of parsing the xml. The first
        # one is to separate in training and test PIDS. If we don't use YOLO
        # this is the only pass neccesary.
        _, pid_dict, _, _, _ = parse_all_xml(
            self.dataset_dir, crop_images=self.crop_images, use_yolo=False
        )
        pids = list(pid_dict.keys())
        num_pids = len(pids)
        num_train_pids = int(num_pids * self.training_test_split)
        pids_copy = copy.deepcopy(pids)
        if self.shuffle_train_test_pids:
            random.shuffle(pids_copy)

        # The training labels should start from zero and increment by one for
        # one-shot-encoding. We realize that the trains_pids are the
        # N first ones and the test pids are the len - N last one's
        # https://github.com/KaiyangZhou/deep-person-reid/issues/190#issuecomment-502843010
        label_dict = {label: index for index, label in enumerate(pids_copy)}
        self.old_new_label_dict = label_dict
        print(f"The dict with the translation of the labels is: {label_dict=}")

        train_pids = pids_copy[:num_train_pids]
        test_pids = pids_copy[num_train_pids:]

        train: list[tuple[str, int, int]] = []
        query: list[tuple[str, int, int]] = []
        gallery: list[tuple[str, int, int]] = []
        camera_train = 0
        camera_query = 0
        camera_gallery = 1

        # for train IDs, all images are used in the train set.
        for pid in train_pids:
            for image_path in pid_dict[pid]:
                new_label = label_dict[pid]
                image_tuple = (image_path, new_label, camera_train)
                train.append(image_tuple)

        if self.use_yolo_for_testing:
            # Parse all the xml again, and in consecuence do the cropping with
            # YOLO, with the predeterminded train/test PID split.
            self.pid_dict_before_yolo = pid_dict.copy()
            (
                _,
                pid_dict,
                num_xml_ids,
                num_yolo_ids,
                num_correct_yolo_ids,
            ) = parse_all_xml(
                self.dataset_dir,
                crop_images=self.crop_images,
                use_yolo=True,
                yolo_ids=test_pids,
                yolo_threshold=self.yolo_threshold,
                iou_threshold=self.yolo_iou_threshold,
            )

            print(
                "IMPORTANT: "
                + f"The stats for YOLO are xml_identified: {num_xml_ids=}, "
                + f"yolo_identified: {num_yolo_ids=}, "
                + f"correct_yolo_identified: {num_correct_yolo_ids=}"
            )
        # for each test ID, randomly choose two images, one for query and the
        # other one for gallery, until we have only zero or one left.
        for pid in test_pids:
            img_names = set(copy.deepcopy(pid_dict[pid]))
            while (len(img_names)) >= 2:
                new_label = label_dict[pid]

                query_img = img_names.pop()
                image_query_tuple = (query_img, new_label, camera_query)
                query.append(image_query_tuple)

                gallery_img = img_names.pop()
                image_gallery_tuple = (gallery_img, new_label, camera_gallery)
                gallery.append(image_gallery_tuple)

            print(
                "IMPORTANT: "
                + f"The training data len is: {len(train)=}, "
                + f"the testing query len is : {len(query)=}, "
                + f"and the len of the testing gallery is: {len(gallery)=}"
            )

        return train, query, gallery

    def prepare_dataset(self, path: str):
        generate_frames(path)
        generate_all_xml_of_dataset(path)
