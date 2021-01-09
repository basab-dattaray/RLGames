def fn_get_rel_dot_folder_path(folder_path, start_pattern):
    index = folder_path.find(start_pattern)
    relative_demo_path = folder_path[index:]
    demo_dot_path = relative_demo_path.replace('/', '.')[1:]
    return demo_dot_path