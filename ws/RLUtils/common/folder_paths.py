import os



# def fn_separate_folderpath_and_filename(filepath):
#     filepathname_parts = filepath.rsplit('/', 1)
#     cwd = filepathname_parts[0]
#     #
#     filename = filepathname_parts[1]
#     filename_parts = filename.rsplit('_', 1)
#     current_path = filename_parts[0]
#     return cwd, current_path
from copy import copy


def fn_get_rel_dot_folder_path(current_path, base_path):

    common_path = os.path.commonprefix([current_path, base_path])
    base_dot_path = common_path.rsplit('/', 2)[1]
    current_dir_path = copy(current_path)
    if current_dir_path.endswith('.py'): # its a file, so it has to be processed
        current_dir_path = os.path.dirname(current_dir_path)

    if common_path in current_dir_path:
        current_path = current_dir_path.replace(common_path, '')
    folder_rel_dot_path = current_path.replace('/', '.')
    if folder_rel_dot_path.startswith('.'):
        folder_rel_dot_path = folder_rel_dot_path[1:]
    folder_rel_path = f'{base_dot_path}.{folder_rel_dot_path}'
    return base_dot_path, folder_rel_path

