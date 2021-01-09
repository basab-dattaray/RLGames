def fn_get_rel_dot_folder_path(folder_path, start_pattern):
    index = folder_path.find(start_pattern)
    relative_demo_path = folder_path[index:]
    demo_dot_path = relative_demo_path.replace('/', '.')[1:]
    return demo_dot_path

def fn_separate_folderpath_and_filename(filepath):
    filepathname_parts = filepath.rsplit('/', 1)
    cwd = filepathname_parts[0]
    #
    filename = filepathname_parts[1]
    filename_parts = filename.rsplit('_', 1)
    folder_path = filename_parts[0]
    return cwd, folder_path