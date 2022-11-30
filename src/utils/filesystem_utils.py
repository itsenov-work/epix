import glob


def search_path_recursive(directory, pattern):
    return glob.glob(f'{directory}/**/{pattern}', recursive=True)
