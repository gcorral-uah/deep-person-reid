from collections import defaultdict
from typing import Dict
import xml.dom.minidom
import glob
import os
import re


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
