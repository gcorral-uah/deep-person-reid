from collections import defaultdict
from typing import Dict, Tuple
import xml.dom.minidom
import glob
import subprocess
import os
import re
import shutil


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


def parse_gt_xml_file(file: str) -> Tuple[str, list[int]]:
    # Expand the ~
    file = os.path.expanduser(file)
    identities: list[int] = []

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


    print(f"Parsing an image:{path=}, {identities=}")
    return path, identities


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


def parse_gt_xml_video(files: list[str]) -> Dict[str, list[int]]:
    dict_file_people: Dict[str, list[int]] = {}

    for file in files:
        file = os.path.expanduser(file)
        image_path, people = parse_gt_xml_file(file)
        dict_file_people[image_path] = people

    return dict_file_people


def parse_gt_xml_dir(path: str) -> Dict[str, list[int]]:
    # Expand the ~
    path = os.path.expanduser(path)

    files = glob.glob("**.xml", root_dir=path)
    p = path + "/" if path[-1] != "/" else path
    files_loc = [p + f for f in files]

    dict_file_people = parse_gt_xml_video(files_loc)
    return dict_file_people


def parse_all_xml(folder: str) -> Tuple[Dict[str, list[int]], Dict[int, list[str]]]:
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

        dict_people_files_directory = parse_gt_xml_dir(xml_dir)
        # Merge the two dicts.
        dict_files_people_global = (
            dict_files_people_global | dict_people_files_directory
        )

    dict_people_files_global = build_rev_dict_gt_xml(dict_files_people_global)

    return dict_files_people_global, dict_people_files_global


if __name__ == "__main__":
    folder = os.path.expanduser("~/Documents/gba_dataset")
    generate_frames(folder)
    generate_all_xml_of_dataset(folder)
    if os.path.exists(folder):
        d, rd = parse_all_xml(folder)
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
