from collections import defaultdict
from typing import Dict, Tuple
import xml.dom.minidom
import glob
import subprocess
import os
import re
import shutil
from PIL import Image, ImageDraw
from typing import Optional, Literal
from ultralytics import YOLO
import functools
import torchvision
import torch
import math

YOLO_DETECTON_THRESHOLD: float = 0.6
YOLO_IOU_THRESHOLD: float = 0.5
YOLO_CLASSES_MAP: dict[str, int] = {
    "human": 0,
}


def generate_frames(
    folder: str, recreate_frames=False, delete_xml=False, delete_frames=False
) -> None:
    # Expand the ~
    folder = os.path.expanduser(folder)
    command = ["bash", "generate_frames.sh"]
    dir = folder

    if delete_xml:
        assert recreate_frames is False and delete_frames is False
        command.append("-x")
    elif recreate_frames:
        assert delete_frames is False
        command.append("-r")
    elif delete_frames:
        command.append("-d")

    print(f"Starting to run {dir}generate_frames.sh")
    subprocess.run(command, cwd=dir)
    print(f"Finished running {dir}generate_frames.sh")


def generate_xml_using_octave(folder: str, remove_xml: bool = False) -> None:
    # Expand the ~
    folder = os.path.expanduser(folder)

    # Example of calling the function on octave
    # gt2xml('~/gba/2016_video003/video3.gt',
    # '~/gba_/2016_video003/FRAMES',
    # '~/gba/2016_video003/xml')
    gt_file_name = glob.glob("*.gt", root_dir=folder)
    assert len(gt_file_name) == 1

    dir = folder + "/" if folder[-1] != "/" else folder
    gt_file = dir + gt_file_name[0]
    frames_dir = dir + "FRAMES"
    xml_dir = dir + "xml"

    if not remove_xml and os.path.exists(xml_dir):
        # If we are not recreating the files and the folder exists, we can skip it.
        return

    if remove_xml:
        print(f"Removing dir {xml_dir}")
        shutil.rmtree(xml_dir, ignore_errors=True)

    command = ["octave", "--eval", f"gt2xml('{gt_file}', '{frames_dir}', '{xml_dir}')"]
    print(f"Running octave command: {command=}")
    subprocess.run(command)


def generate_all_xml_of_dataset(folder: str) -> None:
    # Expand the ~
    folder = os.path.expanduser(folder)
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]

    for dir in subfolders:
        # The other dir contain files that are not video, so ignore it.
        if re.match(".*other.*", dir):
            continue
        print(f"Entering {dir=} to generate_xml_of_dataset")

        generate_xml_using_octave(dir)


# TODO: Refactor, this function is becoming a monstrosity.
def parse_gt_xml_file_and_maybe_crop(
    file: str,
    crop_images: bool = False,
    use_yolo: bool = False,
    yolo_ids: list[int] = [],
    yolo_threshold: float = YOLO_DETECTON_THRESHOLD,
    iou_threshold: float = YOLO_IOU_THRESHOLD,
) -> tuple[list[tuple[str, list[int]]], int, int, int]:
    # Expand the ~
    file = os.path.expanduser(file)
    identities: list[int] = []

    width_parsed, height_parsed = 0, 0
    path = ""
    # The coordinates tuples must be (xmin, xmax, ymin, ymax)
    coords_list: list[tuple[int, int, int, int]] = []
    size_list: list[tuple[int, int]] = []

    anon_coords_list: list[tuple[int, int, int, int]] = []

    xml_doc = xml.dom.minidom.parse(file)
    # There is only one annotation on every document.
    annotation = xml_doc.getElementsByTagName("annotation")[0]
    path_tag = annotation.getElementsByTagName("path")
    path = path_tag[0].childNodes[0].data

    objs = annotation.getElementsByTagName("object")
    for obj in objs:
        id_ = obj.getElementsByTagName("id")
        parsed_id = id_[0].childNodes[0].data
        try:
            number_id = int(parsed_id)
        except ValueError:
            # There are some id numbers that are floats, which is an error so
            # we don't use them.
            continue

        # If the number is greater than 100 there are anonymous, so we don't
        # want them.
        if number_id >= 100:
            bb_wrapper = obj.getElementsByTagName("bndbox")
            for bb_box in bb_wrapper:
                xmin_tag = bb_box.getElementsByTagName("xmin")
                xmin = xmin_tag[0].childNodes[0].data
                xmax_tag = bb_box.getElementsByTagName("xmax")
                xmax = xmax_tag[0].childNodes[0].data
                ymin_tag = bb_box.getElementsByTagName("ymin")
                ymin = ymin_tag[0].childNodes[0].data
                ymax_tag = bb_box.getElementsByTagName("ymax")
                ymax = ymax_tag[0].childNodes[0].data
                try:
                    xmin_parsed = int(xmin)
                    xmax_parsed = int(xmax)
                    ymin_parsed = int(ymin)
                    ymax_parsed = int(ymax)
                    # The coordinates tuples must be (xmin, xmax, ymin, ymax)
                    anon_coords_list.append(
                        (xmin_parsed, xmax_parsed, ymin_parsed, ymax_parsed)
                    )
                except (ValueError, AssertionError):
                    # Ignore the error, this is only for reporting stats for yolo
                    pass

            continue

        identities.append(number_id)

        bb_wrapper = obj.getElementsByTagName("bndbox")
        if len(bb_wrapper) > 1:
            raise ValueError(f"The xml {file=}  has more than bndbox in an object")

        # FIXME: This seems to work, but it shouldn't be a for. There should
        # only be one bndbox tag in per object tag.
        for bb_box in bb_wrapper:
            xmin_tag = bb_box.getElementsByTagName("xmin")
            xmin = xmin_tag[0].childNodes[0].data

            xmax_tag = bb_box.getElementsByTagName("xmax")
            xmax = xmax_tag[0].childNodes[0].data

            ymin_tag = bb_box.getElementsByTagName("ymin")
            ymin = ymin_tag[0].childNodes[0].data

            ymax_tag = bb_box.getElementsByTagName("ymax")
            ymax = ymax_tag[0].childNodes[0].data

            try:
                xmin_parsed = int(xmin)
                xmax_parsed = int(xmax)
                ymin_parsed = int(ymin)
                ymax_parsed = int(ymax)

                assert xmax_parsed > xmin_parsed
                assert ymax_parsed > ymin_parsed

                # The coordinates tuples must be (xmin, xmax, ymin, ymax)
                coords_list.append((xmin_parsed, xmax_parsed, ymin_parsed, ymax_parsed))
            except (ValueError, AssertionError):
                # This is something going very wrong if we have to throw here,
                # but I want to be able to print a little more info, so print and raise.
                print(
                    f"Error parsing bounding box in file {file=} .\n"
                    + f"{xmin=}, {xmax=}, {ymin=}, {ymax=}"
                )
                raise

    size_wrapper = annotation.getElementsByTagName("size")
    if len(size_wrapper) > 1:
        raise ValueError(f"The xml document {file=} has more than one dim tag")

    # Only parse the size of the image if we have any person of interest on in
    # (don't do it if all identities > 100)
    if len(identities) != 0:
        # FIXME: This seems to work, but it shouldn't be a for. There should only
        # be one dim tag in the xml document.
        for dim in size_wrapper:
            width_tag = dim.getElementsByTagName("width")
            width = width_tag[0].childNodes[0].data
            height_tag = dim.getElementsByTagName("height")
            height = height_tag[0].childNodes[0].data
            try:
                width_parsed = int(width)
                height_parsed = int(height)
                size_list.append((width_parsed, height_parsed))
            except ValueError:
                # This is something going very wrong if we have to throw here,
                # but I want to be able to print a little more info, so print and raise.
                print(
                    f"Error parsing dimensions of image in file {file=} .\n"
                    + f"{width=}, {height=}"
                )
                raise

    if use_yolo and len(yolo_ids) == 0:
        # If we haven't specified any ids to crop with YOLO, use them all.
        yolo_ids = identities

    if crop_images:
        cropped_data: list[Tuple[str, list[int]]] = []

        num_identities_in_xml = 0
        num_identified_by_yolo = 0
        num_correct_identified_by_yolo = 0
        num_identities_in_xml = len(coords_list) + len(anon_coords_list)

        if use_yolo:
            valid_yolo_model_results, valid_iou_results = False, False
            yolo_results, yolo_iou_results = None, None
            print(f"Cropping {path=} with YOLO")
            yolo_results, valid_yolo_model_results = calculate_yolo(
                path, confidence_threshold=yolo_threshold
            )
            num_identified_by_yolo = len(yolo_results)

            if valid_yolo_model_results:
                (
                    yolo_iou_results,
                    valid_iou_results,
                    num_correct_identified_by_yolo,
                ) = calculate_best_fit_yolo_greedy(
                    identities,
                    coords_list,
                    yolo_ids,
                    yolo_results,
                    iou_threshold=iou_threshold,
                )

                # If we have any of the ignored results (id > 100) we need to
                # add to the correct identified by YOLO count if any of the
                # additional bboxes by YOLO is one the ignored ids.
                # In the case of the xml identities it's already calculated,
                # so we don't touch it here
                if len(anon_coords_list) > 0 and use_yolo:
                    # yolo_iou_results is list of (_id, coordinates) and we
                    # only want the coordinates
                    yolo_taken_results: list[tuple[int, int, int, int]] = []
                    for _, yolo_taken_res in yolo_iou_results:
                        yolo_taken_results.append(yolo_taken_res)

                    additional_correct_identified_by_yolo = (
                        calculate_extra_anon_id_yolo(
                            anon_coords_list, yolo_taken_results, yolo_results
                        )
                    )

                    num_correct_identified_by_yolo = (
                        num_correct_identified_by_yolo
                        + additional_correct_identified_by_yolo
                    )

            if valid_yolo_model_results and valid_iou_results:
                assert yolo_iou_results is not None
                for yolo_result in yolo_iou_results:
                    id_, coords = yolo_result
                    # The coordinates tuples must be (xmin, xmax, ymin, ymax)
                    new_path = crop_image(
                        path,
                        id_,
                        coords[0],
                        coords[1],
                        coords[2],
                        coords[3],
                        use_yolo=True,
                    )
                    cropped_data.append((new_path, [id_]))

        else:
            print(f"Cropping {path=}")
            for id_, coords in zip(identities, coords_list):
                # The coordinates tuples must be (xmin, xmax, ymin, ymax)
                new_path = crop_image(
                    path,
                    id_,
                    coords[0],
                    coords[1],
                    coords[2],
                    coords[3],
                    use_yolo=False,
                )
                cropped_data.append((new_path, [id_]))

        return (
            cropped_data,
            num_identities_in_xml,
            num_identified_by_yolo,
            num_correct_identified_by_yolo,
        )
    else:
        print(f"Parsing an image:{path=}, {identities=}")
        return [(path, identities)], -1, -1, -1


def build_rev_dict_gt_xml(map: Dict[str, list[int]]) -> Dict[int, list[str]]:
    reverse_map: Dict[int, list[str]] = defaultdict(list)

    # NOTE: Now we are considering all the persons on the image as valid for
    # reidentification. We may want to change it to only consider one person
    # (only use the first value of v to append)
    for k, v in map.items():
        # The value is a list of str, so we need to iterate over it.
        for s in v:
            reverse_map[s].append(k)

    return reverse_map


def parse_gt_xml_video(
    files: list[str],
    crop_images: bool = False,
    use_yolo: bool = False,
    yolo_ids: list[int] = [],
    yolo_threshold: float = YOLO_DETECTON_THRESHOLD,
    iou_threshold: float = YOLO_IOU_THRESHOLD,
) -> tuple[dict[str, list[int]], list[int], list[int], list[int]]:
    dict_file_people: dict[str, list[int]] = {}
    num_xml_identities_list: list[int] = []
    num_identified_by_yolo_list: list[int] = []
    num_correct_identified_by_yolo_list: list[int] = []

    for file in files:
        file = os.path.expanduser(file)
        (
            xml_data_list,
            num_xml_identities,
            num_identified_by_yolo,
            num_correct_identified_by_yolo,
        ) = parse_gt_xml_file_and_maybe_crop(
            file,
            use_yolo=use_yolo,
            yolo_ids=yolo_ids,
            crop_images=crop_images,
            yolo_threshold=yolo_threshold,
            iou_threshold=iou_threshold,
        )

        num_xml_identities_list.append(num_xml_identities)
        num_identified_by_yolo_list.append(num_identified_by_yolo)
        num_correct_identified_by_yolo_list.append(num_correct_identified_by_yolo)

        for elem in xml_data_list:
            image_path, people = elem
            dict_file_people[image_path] = people

    return (
        dict_file_people,
        num_xml_identities_list,
        num_identified_by_yolo_list,
        num_correct_identified_by_yolo_list,
    )


def parse_gt_xml_dir(
    path: str,
    crop_images: bool = False,
    use_yolo: bool = False,
    yolo_ids: list[int] = [],
    yolo_threshold: float = YOLO_DETECTON_THRESHOLD,
    iou_threshold: float = YOLO_IOU_THRESHOLD,
) -> tuple[dict[str, list[int]], list[int], list[int], list[int]]:
    # Expand the ~
    path = os.path.expanduser(path)

    num_xml_identities_list_return: list[int] = []
    num_identified_by_yolo_list_return: list[int] = []
    num_correct_identified_by_yolo_list_return: list[int] = []

    files = glob.glob("**.xml", root_dir=path)
    p = path + "/" if path[-1] != "/" else path
    files_loc = [p + f for f in files]

    (
        dict_file_people,
        num_xml_identities_list,
        num_identified_by_yolo_list,
        num_correct_identified_by_yolo_list,
    ) = parse_gt_xml_video(
        files_loc,
        use_yolo=use_yolo,
        yolo_ids=yolo_ids,
        crop_images=crop_images,
        yolo_threshold=yolo_threshold,
        iou_threshold=iou_threshold,
    )

    num_xml_identities_list_return.extend(num_xml_identities_list)
    num_identified_by_yolo_list_return.extend(num_identified_by_yolo_list)
    num_correct_identified_by_yolo_list_return.extend(
        num_correct_identified_by_yolo_list
    )

    return (
        dict_file_people,
        num_xml_identities_list_return,
        num_identified_by_yolo_list_return,
        num_correct_identified_by_yolo_list_return,
    )


def parse_all_xml(
    folder: str,
    crop_images: bool = False,
    use_yolo: bool = False,
    yolo_ids: list[int] = [],
    yolo_threshold: float = YOLO_DETECTON_THRESHOLD,
    iou_threshold: float = YOLO_IOU_THRESHOLD,
) -> tuple[
    dict[str, list[int]],
    dict[int, list[str]],
    Optional[int],
    Optional[int],
    Optional[int],
]:
    # Expand the ~
    folder = os.path.expanduser(folder)
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    dict_files_people_global: Dict[str, list[int]] = {}

    num_xml_identities_list_final: list[int] = []
    num_identified_by_yolo_list_final: list[int] = []
    num_correct_identified_by_yolo_list_final: list[int] = []

    for dir in subfolders:
        # The other dir contain files that are not video, so ignore it.
        if re.match(".*other.*", dir):
            continue

        print(f"Entering {dir=} to parse xml")

        real_dir = dir + "/" if dir[-1] != "/" else dir
        xml_dir = real_dir + "xml"

        (
            dict_people_files_directory,
            num_xml_identities_list,
            num_identified_by_yolo_list,
            num_correct_identified_by_yolo_list,
        ) = parse_gt_xml_dir(
            xml_dir,
            use_yolo=use_yolo,
            yolo_ids=yolo_ids,
            crop_images=crop_images,
            yolo_threshold=yolo_threshold,
            iou_threshold=iou_threshold,
        )
        # Merge the two dicts.
        dict_files_people_global = (
            dict_files_people_global | dict_people_files_directory
        )
        num_xml_identities_list_final.extend(num_xml_identities_list)
        num_identified_by_yolo_list_final.extend(num_identified_by_yolo_list)
        num_correct_identified_by_yolo_list_final.extend(
            num_correct_identified_by_yolo_list
        )

    dict_people_files_global = build_rev_dict_gt_xml(dict_files_people_global)

    if use_yolo:
        xml_identities = sum(num_xml_identities_list_final)
        identified_by_yolo = sum(num_identified_by_yolo_list_final)
        correct_identified_by_yolo = sum(num_correct_identified_by_yolo_list_final)
    else:
        xml_identities = None
        identified_by_yolo = None
        correct_identified_by_yolo = None

    return (
        dict_files_people_global,
        dict_people_files_global,
        xml_identities,
        identified_by_yolo,
        correct_identified_by_yolo,
    )


def calculate_yolo(
    path: str,
    classes: list[int] = [YOLO_CLASSES_MAP["human"]],
    confidence_threshold: float = YOLO_DETECTON_THRESHOLD,
    yolo_level: Literal["small", "large"] = "large",
) -> tuple[list[tuple[int, int, int, int]], bool]:
    if yolo_level == "small":
        model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model (small and fast)
        print("Using small YOLO model")
    elif yolo_level == "large":
        model = YOLO("yolov8x.pt")  # pretrained YOLOv8x model (big and slow)
        print("Using big YOLO model")
    else:
        model = None
        assert (
            model is not None
        ), f"yolo_level, can only be 'small' or 'large', you have {yolo_level=}"

    valid_yolo_results = False
    results = model([path], classes=classes, conf=confidence_threshold)

    boxes: list[list[int]] = []
    for idx, result in enumerate(results):
        assert idx == 0  # Confirmation that we only have one result.

        # Location of human bounding boxes.
        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(int).tolist()

    print(f"The YOLO boxes for {path=} are {boxes=} in xyxy")
    if len(boxes) > 0:
        valid_yolo_results = True

    # The coordinates tuples must be  (xmin, xmax, ymin, ymax)
    # The tuples returned by yolo are (xmin, ymin, xmax, ymax)
    tuple_boxes: list[tuple[int, int, int, int]] = []
    for array in boxes:
        x_min = array[0]
        y_min = array[1]
        x_max = array[2]
        y_max = array[3]
        crop_tuple = (x_min, x_max, y_min, y_max)
        tuple_boxes.append(crop_tuple)

    return tuple_boxes, valid_yolo_results


def iou_pytorch(
    boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]
) -> float:
    # Solution from https://stackoverflow.com/a/65988061

    # The coordinates tuples are (xmin, xmax, ymin, ymax)
    x_min_a = float(boxA[0])
    x_max_a = float(boxA[1])
    y_min_a = float(boxA[2])
    y_max_a = float(boxA[3])

    x_min_b = float(boxB[0])
    x_max_b = float(boxB[1])
    y_min_b = float(boxB[2])
    y_max_b = float(boxB[3])

    # In torch the boxes are (x1, y1, x2, y2) with 0 <= x1 < x2 and 0 <= y1 < y2.
    tensorA = torch.tensor([[x_min_a, y_min_a, x_max_a, y_max_a]], dtype=torch.float)
    tensorB = torch.tensor([[x_min_b, y_min_b, x_max_b, y_max_b]], dtype=torch.float)
    tensorResult = torchvision.ops.box_iou(tensorA, tensorB)
    resList = tensorResult.tolist()
    assert len(resList) == 1 and len(resList[0]) == 1
    res = resList[0][0]

    # If the condition are not matched the function returns NaN. If that is the case
    # the boxes don't overlap, so return 0. It can also return -0.0f that is
    # equivalent to 0.0f, so we don't need to handle the case.
    if math.isnan(res):
        return 0
    else:
        return res


def iou(boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]) -> float:
    # The coordinates tuples must be (xmin, xmax, ymin, ymax)
    x_min_a = boxA[0]
    x_max_a = boxA[1]
    y_min_a = boxA[2]
    y_max_a = boxA[3]

    x_min_b = boxB[0]
    x_max_b = boxB[1]
    y_min_b = boxB[2]
    y_max_b = boxB[3]

    # determine the (x, y)-coordinates of the intersection rectangle
    xmax = min(x_max_a, x_max_b)
    ymax = min(y_max_a, y_max_b)
    xmin = max(x_min_a, x_min_b)
    ymin = max(y_min_a, y_min_b)

    # compute the area of intersection rectangle
    interArea = max(0, xmax - xmin + 1) * max(0, ymax - ymin + 1)
    if interArea == 0:
        return 0

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((x_max_a - x_min_a + 1) * (y_max_a - y_min_a + 1))
    boxBArea = abs((x_max_b - x_min_b + 1) * (y_max_b - y_min_b + 1))

    # compute the intersection over union by taking the intersection area and
    # dividing it by the sum of prediction + ground-truth areas - the
    # interesection area (The intersecton area is sumed twice, so we have to
    # substract it once, to obtain the correct resutl)
    union_area = boxAArea + boxBArea - interArea
    if union_area == 0:
        return 0

    iou = float(interArea) / float(union_area)
    return iou


def calculate_best_fit_yolo(
    identities: list[int],
    xml_coords_list: list[tuple[int, int, int, int]],
    yolo_ids: list[int],
    yolo_coords_results: list[tuple[int, int, int, int]],
    iou_threshold: float = YOLO_IOU_THRESHOLD,
) -> tuple[list[tuple[int, tuple[int, int, int, int]]], bool, int]:
    results: list[tuple[int, tuple[int, int, int, int]]] = []
    valid_iou_results = False

    num_correct_identified_by_yolo = 0

    # This may not find the best IOU the first time.
    # If you have more than one result use the greedy version.
    print(f"Calculate best fit yolo {xml_coords_list=} {yolo_coords_results=}")
    annotations_seen_by_yolo_list: list[int] = []
    yolo_coordinates_matched: list[tuple[int, int, int, int]] = []
    for annotation_id, coords in zip(identities, xml_coords_list):
        for yolo_coords in yolo_coords_results:
            iou_result = iou(coords, yolo_coords)
            if iou_result >= iou_threshold:
                if annotation_id not in annotations_seen_by_yolo_list:
                    num_correct_identified_by_yolo = num_correct_identified_by_yolo + 1
                    annotations_seen_by_yolo_list.append(annotation_id)

                # The problem may be that there are two coordinates items with
                # the same annotation_id value. The break takes into account
                # the unique matching of coords with all the yolo_coords, but
                # another set of coords may match an already taken yolo_coords,
                # so use `yolo_coordinates_matched` to avoid it.
                if (
                    annotation_id in yolo_ids
                    and yolo_coords not in yolo_coordinates_matched
                ):
                    results.append((annotation_id, yolo_coords))
                    yolo_coordinates_matched.append(yolo_coords)
                    break

    if len(results) > 0:
        valid_iou_results = True

    print(f"Calculate best fit yolo {results=}")
    return (
        results,
        valid_iou_results,
        num_correct_identified_by_yolo,
    )


def crop_image(
    path: str,
    idx: Optional[int],
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    use_yolo: bool = False,
) -> str:
    """
    This function crops an image to (x_min-x_max, y_min-y_max) and saves it in
    the original path with _ + str(idx) as suffix. If no sufix is given it uses
    0 by default.
    It returns the path of the cropeed image.
    """
    idx = idx if idx is not None else 0

    # This is a hack to separate the absolute path and the extension, but as we
    # have a user with a dot in it's name, we can't relly in the dot to act as
    # a separator between the filename and the extension and extract it with
    # regex. Also the iteration should always be O(c*1), with a very small c.
    dot_char_idx = -1
    for char in reversed(path):
        if char == ".":
            break
        dot_char_idx -= 1

    # We use dot_char_idx + 1 to exclude the point of the extension.
    filename_without_extension, extension = (
        path[:dot_char_idx],
        path[dot_char_idx + 1 :],
    )

    yolo_msg = "_yolo" if use_yolo is True else ""
    new_path = filename_without_extension + yolo_msg + "_" + str(idx) + "." + extension

    # To avoid work, if the new file already exists, we can skip the work.
    if os.path.isfile(new_path):
        print(f"Skipping the cropping of {new_path=}, because it already exists")
        return new_path

    print(
        f"Cropping {path=} and {idx=} to form {new_path},"
        + f"with {x_min=}, {x_max=}, {y_min=}, {y_max=} with {use_yolo=}"
    )

    img = Image.open(path)
    new_img = img.crop((x_min, y_min, x_max, y_max))
    new_img.save(new_path)

    return new_path


def calculate_best_fit_yolo_greedy(
    identities: list[int],
    xml_coords_list: list[tuple[int, int, int, int]],
    yolo_ids: list[int],
    yolo_coords_results: list[tuple[int, int, int, int]],
    iou_threshold: float = YOLO_IOU_THRESHOLD,
) -> tuple[list[tuple[int, tuple[int, int, int, int]]], bool, int]:
    results: list[tuple[int, tuple[int, int, int, int]]] = []
    valid_iou_results = False

    assert len(identities) == len(xml_coords_list)

    def _sort_best_iou(
        x: tuple[tuple[int, int], float], y: tuple[tuple[int, int], float]
    ) -> int:
        iou_x: float = x[1]
        iou_y: float = y[1]
        if iou_x > iou_y:
            return 1
        elif iou_x < iou_y:
            return -1
        else:
            return 0

    if len(xml_coords_list) == 1 and len(yolo_coords_results) == 1:
        print(
            "We only have one identity, using the normal function to calc the best fit."
        )
        return calculate_best_fit_yolo(
            identities=identities,
            xml_coords_list=xml_coords_list,
            yolo_ids=yolo_ids,
            yolo_coords_results=yolo_coords_results,
            iou_threshold=iou_threshold,
        )
    elif len(xml_coords_list) == 0 or len(yolo_coords_results) == 0:
        print("In iou calc one of the coordinates list is empty.")
        dummy_coords = (-1, -1, -1, -1)
        dummy_id = -1
        num_correct_identified_by_yolo_error_case = 0
        return (
            [(dummy_id, dummy_coords)],
            False,
            num_correct_identified_by_yolo_error_case,
        )

    num_xml_identities = len(xml_coords_list)
    num_identified_by_yolo = len(yolo_coords_results)
    num_correct_identified_by_yolo = 0

    # Be greedy when we calculate the best fit. Try to match all the
    # coordinates from the coordinates list with all the yolo coordinates and
    # select the best fit for each.
    print(f"Calculate best fit yolo greedy {xml_coords_list=} {yolo_coords_results=}")
    possible_num_correct_identified_by_yolo = 0
    best_fits: list[tuple[tuple[int, int], float]] = []
    for i_idx, xml_data in enumerate(zip(identities, xml_coords_list)):
        annotation_id, coords = xml_data
        for j_idx, yolo_coords in enumerate(yolo_coords_results):
            iou_result = iou(coords, yolo_coords)
            if iou_result >= iou_threshold:
                # First check for correct detections by YOLO, and after that
                # check if it's an interesting one.
                possible_num_correct_identified_by_yolo = (
                    possible_num_correct_identified_by_yolo + 1
                )
                if annotation_id in yolo_ids:
                    idxs = (i_idx, j_idx)
                    best_fits.append((idxs, iou_result))

    # By default sorts from lower to higher and here we want the best scores first
    best_fits.sort(key=functools.cmp_to_key(_sort_best_iou), reverse=True)

    print(f" The IOU best fits are {best_fits=}")
    avalible_xml_idx = [i for i in range(len(xml_coords_list))]
    avalible_yolo_idx = [i for i in range(len(yolo_coords_results))]
    while (
        len(avalible_xml_idx) > 0 and len(avalible_yolo_idx) > 0 and len(best_fits) > 0
    ):
        best_result = best_fits[0]
        idxs = best_result[0]
        xml_idx = idxs[0]
        yolo_idx = idxs[1]
        if xml_idx in avalible_xml_idx and yolo_idx in avalible_yolo_idx:
            identifier = identities[xml_idx]
            yolo_best_coords = yolo_coords_results[yolo_idx]
            results.append((identifier, yolo_best_coords))

            avalible_yolo_idx.remove(yolo_idx)
            avalible_xml_idx.remove(xml_idx)

        # Remove this inconditionaly, as the result is already appended or not
        # valid.
        best_fits.remove(best_result)

    if len(results) > 0:
        valid_iou_results = True

    # HACK: For now this seems good enough
    max_possible_num_correct_identified_by_yolo = min(
        num_identified_by_yolo, num_xml_identities
    )
    if (
        possible_num_correct_identified_by_yolo
        > max_possible_num_correct_identified_by_yolo
    ):
        num_correct_identified_by_yolo = max_possible_num_correct_identified_by_yolo
    else:
        num_correct_identified_by_yolo = possible_num_correct_identified_by_yolo

    print(f"Calculated best fit yolo greedily {results=}")
    return (
        results,
        valid_iou_results,
        num_correct_identified_by_yolo,
    )


def calculate_extra_anon_id_yolo(
    anon_coords: list[tuple[int, int, int, int]],
    taken_yolo_coordinates: list[tuple[int, int, int, int]],
    yolo_coords: list[tuple[int, int, int, int]],
    iou_threshold: float = YOLO_IOU_THRESHOLD,
) -> int:
    # HACK: This is a naive version of the coordinate matching, because it's
    # only for statistics, and it should have the mayority of people already
    # identified.

    # Don't modify an array we are operating in.
    remaining_yolo_coordinates = yolo_coords.copy()
    for y_coords in yolo_coords:
        for taken_y_coords in taken_yolo_coordinates:
            if taken_y_coords in remaining_yolo_coordinates:
                remaining_yolo_coordinates.remove(y_coords)

    num_matched_coords = 0
    anon_matched_coords: list[tuple[int, int, int, int]] = []

    for a_coords in anon_coords:
        for rem_yolo_coords in remaining_yolo_coordinates:
            calculated_iou = iou(a_coords, rem_yolo_coords)
            if calculated_iou > iou_threshold and a_coords not in anon_matched_coords:
                num_matched_coords = num_matched_coords + 1
                anon_matched_coords.append(a_coords)

    return num_matched_coords


def calculate_coords_in_bounding_box(
    list_coords: list[tuple[int, int, int, int]], box: tuple[int, int, int, int]
) -> int:
    # The coordinates tuples must be (xmin, xmax, ymin, ymax)
    return_value = 0
    for coords in list_coords:
        if check_bbox_intersection_iou(coords, box):
            return_value += 1

    return return_value


def check_bbox_intersection(
    boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]
) -> bool:
    x_min_a = boxA[0]
    x_max_a = boxA[1]
    y_min_a = boxA[2]
    y_max_a = boxA[3]

    x_min_b = boxB[0]
    x_max_b = boxB[1]
    y_min_b = boxB[2]
    y_max_b = boxB[3]

    # Code inspired by https://stackoverflow.com/a/55088324
    # In this rec1 is boxA and rec2 is boxB.
    # Also x1 == x_min, x2 == xmax, y1 == ymin and y2 == ymax.
    first_term_x = (x_max_b >= x_min_a) and (x_max_b <= x_max_a)
    second_term_x = (x_min_b >= x_min_a) and (x_min_b <= x_max_a)

    first_term_y = (y_max_b >= y_min_a) and (y_max_b <= y_max_a)
    second_term_y = (y_min_b >= y_min_a) and (y_min_b <= y_max_a)

    x_match = first_term_x or second_term_x
    y_match = first_term_y or second_term_y

    return x_match and y_match


def check_bbox_intersection_iou(
    boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]
) -> bool:
    return iou(boxA, boxB) > 0.0


def draw_region_in_image(
    path: str,
    bbox: tuple[int, int, int, int],
    new_path: Optional[str],
    show: bool = False,
):
    # The coordinates tuples must be (xmin, xmax, ymin, ymax)

    if new_path is None:
        # This is a hack to separate the absolute path and the extension, but as we
        # have a user with a dot in it's name, we can't relly in the dot to act as
        # a separator between the filename and the extension and extract it with
        # regex. Also the iteration should always be O(c*1), with a very small c.
        dot_char_idx = -1
        for char in reversed(path):
            if char == ".":
                break
            dot_char_idx -= 1

        # We use dot_char_idx + 1 to exclude the point of the extension.
        filename_without_extension, extension = (
            path[:dot_char_idx],
            path[dot_char_idx + 1 :],
        )

        new_path = filename_without_extension + "_with_region_painted" + extension

    orig_img = Image.open(path)
    img = orig_img.copy()

    x_min = bbox[0]
    x_max = bbox[1]
    y_min = bbox[2]
    y_max = bbox[3]

    rectangle = ImageDraw.Draw(img)
    rectangle.rectangle((x_min, y_min, x_max, y_max), fill=None, outline="green")

    img.save(new_path)
    if show:
        img.show()


def xxyy_to_xyxy(coords: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (coords[0], coords[2], coords[1], coords[3])


def xyxy_to_xxyy(coords: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (coords[0], coords[2], coords[1], coords[3])


if __name__ == "__main__":
    folder = os.path.expanduser("~/Documents/gba_dataset")
    generate_frames(folder)
    generate_all_xml_of_dataset(folder)
    if os.path.exists(folder):
        (
            d,
            rd,
            xml_identities,
            identified_by_yolo,
            correct_identified_by_yolo,
        ) = parse_all_xml(
            folder, crop_images=True, use_yolo=True, yolo_ids=[5, 7, 9, 1]
        )
        for k, v in d.items():
            print(f"[{k}]= [")
            for item in v:
                print(item)
            print("]")

        for k, v in rd.items():
            print(f"[{k}]= [")
            for item in v:
                print(item)
            print("]")

        print(
            f"{xml_identities=}, {identified_by_yolo=}, {correct_identified_by_yolo=}"
        )

        len_training_data = 0
        for k, v in rd.items():
            len_training_data += len(v[k])
