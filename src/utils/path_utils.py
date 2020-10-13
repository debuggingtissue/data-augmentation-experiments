import os


def remove_file(path):
    os.remove(path)

def get_all_files_in_directory(path):
    return [x for x in os.listdir(path)]
