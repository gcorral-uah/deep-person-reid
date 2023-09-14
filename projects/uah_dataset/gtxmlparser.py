from collections import defaultdict
from typing import Dict, Tuple
import xml.dom.minidom
import glob
import subprocess
import os
import re
import shutil
from PIL import Image
from typing import Optional
from ultralytics import YOLO

YOLO_DETECTON_THRESHOLD: float = 0.8
YOLO_IOU_THRESHOLD: float = 0.8
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


# TODO: Refactor, this function is becomming a monstruosity.
def parse_gt_xml_file_and_maybe_crop(
    file: str,
    crop_images=False,
    use_yolo: bool = False,
    yolo_ids: list[int] = [],
    yolo_threshold=YOLO_IOU_THRESHOLD,
) -> list[Tuple[str, list[int]]]:
    # Expand the ~
    file = os.path.expanduser(file)
    identities: list[int] = []

    width_parsed, height_parsed = 0, 0
    path = ""
    coords_list: list[tuple[int, int, int, int]] = []
    size_list: list[tuple[int, int]] = []

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

    if crop_images:
        print(f"Cropping {path=}")
        cropped_data: list[Tuple[str, list[int]]] = []
        for id_, elem in zip(identities, coords_list):
            coords = elem
            new_path = crop_image(path, id_, coords[0], coords[1], coords[2], coords[3])
            cropped_data.append((new_path, [id_]))
        return cropped_data
    else:
        print(f"Parsing an image:{path=}, {identities=}")
        return [(path, identities)]


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


def parse_gt_xml_video(files: list[str], crop_images=False) -> Dict[str, list[int]]:
    dict_file_people: Dict[str, list[int]] = {}

    for file in files:
        file = os.path.expanduser(file)
        xml_data_list = parse_gt_xml_file_and_maybe_crop(file, crop_images=crop_images)
        for elem in xml_data_list:
            image_path, people = elem
            dict_file_people[image_path] = people

    return dict_file_people


def parse_gt_xml_dir(path: str, crop_images=False) -> Dict[str, list[int]]:
    # Expand the ~
    path = os.path.expanduser(path)

    files = glob.glob("**.xml", root_dir=path)
    p = path + "/" if path[-1] != "/" else path
    files_loc = [p + f for f in files]

    dict_file_people = parse_gt_xml_video(files_loc, crop_images=crop_images)
    return dict_file_people


def parse_all_xml(
    folder: str, crop_images=False
) -> Tuple[Dict[str, list[int]], Dict[int, list[str]]]:
    # Expand the ~
    folder = os.path.expanduser(folder)
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    dict_files_people_global: Dict[str, list[int]] = {}

    for dir in subfolders:
        # The other dir contain files that are not video, so ignore it.
        if re.match(".*other.*", dir):
            continue

        print(f"Entering {dir=} to parse xml")

        real_dir = dir + "/" if dir[-1] != "/" else dir
        xml_dir = real_dir + "xml"

        dict_people_files_directory = parse_gt_xml_dir(xml_dir, crop_images=crop_images)
        # Merge the two dicts.
        dict_files_people_global = (
            dict_files_people_global | dict_people_files_directory
        )

    dict_people_files_global = build_rev_dict_gt_xml(dict_files_people_global)

    return dict_files_people_global, dict_people_files_global


def calculate_yolo(
    path: str,
    classes: list[int] = [YOLO_CLASSES_MAP["human"]],
    confidence_threshold: float = YOLO_IOU_THRESHOLD,
) -> list[tuple[int, int, int, int]]:
    model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

    results = model([path], classes=classes, conf=confidence_threshold)

    boxes: list[list[int]] = []
    for idx, result in enumerate(results):
        assert idx == 0  # Confirmation that we only have one result.

        # Location of human bounding boxes.
        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(int).tolist()

    print(f"The YOLO bboxes are {boxes=}")

    tuple_boxes: list[tuple[int, int, int, int]] = []
    for array in boxes:
        x_min = array[0]
        x_max = array[1]
        y_min = array[2]
        y_max = array[3]
        crop_tuple = (x_min, x_max, y_min, y_max)
        tuple_boxes.append(crop_tuple)

    return tuple_boxes


def iou(objA: tuple[int, int, int, int], objB: tuple[int, int, int, int]) -> float:
    x_min_a = objA[0]
    x_max_a = objA[1]
    y_min_a = objA[2]
    y_max_a = objA[3]

    x_min_b = objB[0]
    x_max_b = objB[1]
    y_min_b = objB[2]
    y_max_b = objB[3]

    # determine the (x, y)-coordinates of the intersection rectangle
    xmax = max(x_max_a, x_max_b)
    ymax = max(y_max_a, y_max_b)
    xmin = min(x_min_a, x_min_b)
    ymin = min(y_min_a, y_min_b)

    # compute the area of intersection rectangle
    interArea = max(0, xmax - xmin + 1) * max(0, ymax - ymin + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (x_max_a - x_min_a + 1) * (y_max_a - y_min_a + 1)
    boxBArea = (x_max_b - x_min_b + 1) * (y_max_b - y_min_b + 1)

    # compute the intersection over union by taking the intersection area and
    # dividing it by the sum of prediction + ground-truth areas - the
    # interesection area (The intersecton area is sumed twice, so we have to
    # substract it once, to obtain the correct resutl)
    iou = float(interArea) / float(boxAArea + boxBArea - interArea)

    return iou


def calculate_best_fit_yolo(
    identities: list[int],
    coords_list: list[tuple[int, int, int, int]],
    yolo_ids: list[int],
    yolo_results: list[tuple[int, int, int, int]],
    iou_threshold: float = YOLO_IOU_THRESHOLD,
) -> list[tuple[int, tuple[int, int, int, int]]]:
    results: list[tuple[int, tuple[int, int, int, int]]] = []

    for annotation_id, coords in zip(identities, coords_list):
        for yolo_coords in yolo_results:
            iou_result = iou(coords, yolo_coords)
            if iou_result >= iou_threshold and annotation_id in yolo_ids:
                results.append((annotation_id, yolo_coords))

    return results


def crop_image(
    path: str, idx: Optional[int], x_min: int, x_max: int, y_min: int, y_max: int, use_yolo: bool = False
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

    yolo_msg = "_yolo_" if use_yolo is True else ""
    new_path = filename_without_extension + yolo_msg + "_" + str(idx) + "." + extension

    # To avoid work, if the new file already exists, we can skip the work.
    if os.path.isfile(new_path):
        print(f"Skipping the cropping of {new_path=}, because it already exists")
        return new_path

    img = Image.open(path)
    new_img = img.crop((x_min, y_min, x_max, y_max))
    new_img.save(new_path)

    print(
        f"Cropped {path=} and {idx=} to form {new_path},"
        + f"with {x_min=}, {x_max=}, {y_min=}, {y_max=}"
    )

    return new_path


if __name__ == "__main__":
    folder = os.path.expanduser("~/Documents/gba_dataset")
    generate_frames(folder)
    generate_all_xml_of_dataset(folder)
    if os.path.exists(folder):
        d, rd = parse_all_xml(folder, crop_images=False)
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
