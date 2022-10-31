import os
from utils.logger import LoggerMixin


class Dir:
    '''
    Utility methods to change the root directory to the main project dir
    Use every time you want to run tests on any file in the directory

    Files in 'src' will automatically use the root project directory as project dir
    and have access to the 'resources' folder there, for example

    Files in 'tests' will set the root project directory as 'project dir'/'tests'
    that way tests will not affect the main resources
    '''

    @staticmethod
    def get_project_dir():
        cur_path = os.path.abspath('.')
        dirs = cur_path.split(os.path.sep)
        project_idx = len(dirs)
        for idx, dir in enumerate(dirs):
            if dir == 'src':
                project_idx = idx
            elif dir == 'tests':
                project_idx = idx + 1

        project_dir = os.path.sep.join(dirs[:project_idx])
        return project_dir

    @staticmethod
    def set_project_dir():
        project_dir = Dir.get_project_dir()
        os.chdir(project_dir)

    @staticmethod
    def get_resources_dir():
        return os.path.join(Dir.get_project_dir(), 'resources')

    @staticmethod
    def get_data_dir():
        return os.path.join(Dir.get_resources_dir(), 'data')

    @staticmethod
    def get_download_dir():
        return os.path.join(Dir.get_resources_dir(), 'downloads')
