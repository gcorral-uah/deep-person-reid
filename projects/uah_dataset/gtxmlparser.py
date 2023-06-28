from collections import defaultdict
from os.path import exists
from typing import Dict
import xml.dom.minidom
import glob
import subprocess
import os
import glob
import re
import shutil

def generate_frames(folder: str) -> None:
    # Expand the ~
    folder = os.path.expanduser(folder)
    command = ["bash", "generate_frames.sh"]
    dir = folder
    subprocess.run(command, cwd=dir)

def generate_xml_using_octave(folder: str, remove_xml: bool = False) -> None:
    # Expand the ~
    folder = os.path.expanduser(folder)

    print(f"Entering to process and generate xml on folder {folder=}")

    # Example of calling the function on octave
    # gt2xml('~/gba/2016_video003/video3.gt', '~/gba_/2016_video003/FRAMES', '~/gba/2016_video003/xml')
    gt_file_name = glob.glob("*.gt",root_dir=folder)
    assert len(gt_file_name) == 1

    dir = folder + '/' if folder[-1] != '/' else folder
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
    subprocess.run(command)

def generate_all_xml_of_dataset(folder: str) -> None:
    # Expand the ~
    folder = os.path.expanduser(folder)
    subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]

    for dir in subfolders:
        print(f"Entering {dir=}")
        # The other dir contain files that are not video, so ignore it.
        if re.match(".*other.*", dir):
            continue

        generate_xml_using_octave(dir)


def parse_gt_xml_file(file: str) -> list[int]:
    # Expand the ~
    file = os.path.expanduser(file)
    identities: list[int] = []

    xml_doc = xml.dom.minidom.parse(file)
    # There is only one annotation on every document.
    annotation = xml_doc.getElementsByTagName("annotation")[0]
    objs = annotation.getElementsByTagName("object")
    for obj in objs:
        id = obj.getElementsByTagName("id")
        parsed_id = id[0].childNodes[0].data
        number_id = int(parsed_id)
        # If the number is greater than 100 there are anonymous, so we don't
        # want them.
        if number_id >= 100:
            continue

        identities.append(number_id)

    return identities


def build_rev_dict_gt_xml(map: Dict[str, list[int]]) -> Dict[int, list[str]]:
    reverse_map: Dict[int, list[str]] = {}

    for k, v in map.items():
        # The value is a list of str, so we need to iterate over it.
        for s in v:
            # Now the string is the key of the map, but we need to initialize
            # the array of int if it doesn't exist before appending.
            if s in reverse_map:
                # The key exist, we can simply append.
                reverse_map[s].append(k)
            else:
                # Create a new key
                reverse_map[s] = [k]

    return reverse_map


def parse_gt_xml_video(files: list[str]) -> Dict[int, list[str]]:
    dict_file_people: Dict[str, list[int]] = {}

    for file in files:
        file = os.path.expanduser(file)
        dict_file_people[file] = parse_gt_xml_file(file)

    dict_people_file = build_rev_dict_gt_xml(dict_file_people)

    return dict_people_file

def parse_gt_xml_dir(path: str) -> Dict[int, list[str]]:
    # Expand the ~
    path = os.path.expanduser(path)

    files = glob.glob(path + "*.xml")
    return parse_gt_xml_video(files)

def parse_all_xml(folder: str) -> Dict[int, list[str]]:
    # Expand the ~
    folder = os.path.expanduser(folder)
    subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
    key_val_store = defaultdict(list)

    for dir in subfolders:
        print(f"Entering {dir=}")
        # The other dir contain files that are not video, so ignore it.
        if re.match(".*other.*", dir):
            continue

        real_dir = dir + '/' if dir[-1] != '/' else dir
        xml_dir = real_dir + "xml"
        local_dict = parse_gt_xml_dir(xml_dir)
        for k,v in local_dict.items():
            key_val_store[k].append(v)

    return key_val_store
