import os
import json
from pathlib import Path
import shutil
import datetime


def remove_file(file_path):
    if file_path is None:
        return
    path = os.path.abspath(file_path)
    if path is not None:
        if os.path.exists(path):
            os.remove(path)


def remove_directory_tree(dir_path):
    path = os.path.abspath(dir_path)
    if os.path.exists(path):
        shutil.rmtree(path)


def get_filename_based_on_time():
    s = str(datetime.datetime.now())
    fname = s.replace(':', '-')
    return fname


def get_file_modification_datetime(filename_path):
    filepath = os.path.abspath(filename_path)
    t = os.path.getmtime(filepath)
    return datetime.datetime.fromtimestamp(t)


def move_and_override_file(src_file_path, dst_dir_path, src_file_name):
    src_file_path = os.path.abspath(src_file_path)
    dst_dir_path = os.path.abspath(dst_dir_path)
    if not os.path.exists(dst_dir_path):
        os.makedirs(dst_dir_path)

    dst_file_path = os.path.join(dst_dir_path, src_file_name)
    if os.path.exists(dst_file_path):
        os.remove(dst_file_path)

    # dst_file_name = src_file_path.rsplit('/', 1)[1]
    dst_file_path = dst_dir_path + src_file_name
    shutil.move(src_file_path, dst_file_path)


def get_json_data(json_path):
    dict = {}
    file_path = os.path.abspath(json_path)
    with open(file_path, 'r') as json_data:
        dict = json.load(json_data)
    return dict


def get_lines_from_file(rel_file_path):
    file_path = os.path.abspath(rel_file_path)
    lines = None
    result = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.rstrip()
            result.append(l)
    return result
