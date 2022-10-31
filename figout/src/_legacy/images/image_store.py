import os
import pickle
import datetime
import shutil
import warnings
import natsort

import numpy as np
from skimage import io
from skimage import transform
from skimage import img_as_ubyte
from utils.logger import LoggerMixin
from utils.resize import Resize


class ImageStore(LoggerMixin):
    img_folder = os.path.join("resources", "images")
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    def __init__(self, logging_frequency=200):
        """
        logging frequency: how often the imagestore informs you about its progress during image i/o
        """
        super(ImageStore, self).__init__()
        self.logging_frequency = logging_frequency
        
    def pickle(self, folder, resize: Resize = Resize(100), num_images=0, delete_older=False):
        """
        folder: the subfolder of images
        image_dimensions: a tuple (x, y, num_channels) you want to transform the images to
        delete_older: boolean to clear all former pickles of the same name and size, so as not to clutter space
        """

        image_paths = self._get_all_image_paths(folder)
        if num_images == 0:
            num_images = len(image_paths)
        else:
            image_paths = image_paths[:num_images]
        self.log.start("Pickling %d images in folder %s at size %s." % (num_images, folder, str(resize)))

        if delete_older:
            self.log.i("Deleting older pickles...")
            self.clear_pickles(folder, resize)

        self.log.i("Dumping images...")

        image_dimensions = self._get_image_dimensions(image_paths[0], resize)
        image_array = self._get_empty_np_array(num_images, image_dimensions)

        for i, path in enumerate(image_paths):
            self._read_and_dump_image(image_array, i, path)

            if (i + 1) % self.logging_frequency == 0:
                self.log.i("%d images dumped..." % (i + 1))

        self.log.ok("All %d images dumped" % num_images)
        self.log.i("Attempting to pickle...")

        pickle_filename = self._do_pickle(folder, image_array, resize)
        self.log.s("Pickled successfully! Creating index file...")
        self._create_pickle_indices(folder, resize, pickle_filename)
        self.log.s("Index file created. Enjoy your pickle %s!" % pickle_filename)
        self.log.end()

    def unpickle(self, folder_name, resize: Resize = Resize(100),
                 timestamp=None, store_as_files=False, get_image_names=False):
        '''
        folder_name: the subfolder of images
        timestamp: (Optional) which pickle you want to unpickle. If not provided, automatically unpickles latest pickle
        '''

        self.log.start("Unpickling in folder %s at size %s" % (folder_name, str(resize)))

        pickle_folder = self._get_pickled_folder(folder_name, resize)
        pickled_filename = self._get_pickled_filename(folder_name, resize, timestamp)

        file_path = os.path.join(pickle_folder, pickled_filename)
        self.log.i("Unpickling %s..." % pickled_filename)
        self.log.i("Loading images...")
        with open(file_path, "rb") as f:
            images = pickle.load(f)
            num_images = images.shape[0]
        self.log.ok("%d images loaded in memory!" % num_images)

        if store_as_files:
            self.log.i("Storing images:")
            image_names = self._get_pickle_indices(folder_name, resize, pickled_filename)
            self._store_unpickled_images_as_files(images, folder_name, resize, image_names)
            self.log.i("Stored images!")

        self.log.s("Unpickled %d images successfully!" % num_images)
        self.log.end()

        if get_image_names:
            image_names = self._get_pickle_indices(folder_name, resize, pickled_filename)
            return images, image_names
        else:
            return images

    def provide(self, folder_name, resize: Resize = Resize(100), num_images=0, get_image_names=False):
        try:
            return self.unpickle(folder_name=folder_name, resize=resize, get_image_names=get_image_names)
        except FileNotFoundError as e:
            self.pickle(folder=folder_name, resize=resize, num_images=num_images)
            return self.unpickle(folder_name=folder_name, resize=resize, get_image_names=get_image_names)

    def clear_pickles(self, folder_name, resize: Resize = Resize(100)):
        '''
        Deletes all current pickle files.
        '''
        pickled_folder = self._get_pickled_folder(folder_name, resize)
        shutil.rmtree(pickled_folder)

    def store_images_as_files_in_path(self, images, folder_path, image_names=None):
        self.log.i("Storing images:")
        self._do_store(images, folder_path, image_names)
        self.log.i("Stored images!")
        self.log.end()

    def store_images_as_files(self, images, folder_name, image_names=None):
        folder_path = self.get_folder_path(folder_name)
        self.store_images_as_files_in_path(images, folder_path, image_names)

    def _do_store(self, images, folder_path, image_names=None):
        if image_names is None:
            image_names = [str(i) + ".png" for i in range(len(images))]

        for idx, image in enumerate(images):
            image_path = os.path.join(folder_path, image_names[idx])
            io.imsave(image_path, img_as_ubyte(image), check_contrast=False)
            if (idx + 1) % self.logging_frequency == 0:
                self.log.i("%d images saved..." % (idx + 1))

    def _store_unpickled_images_as_files(self, images, folder_name, resize, image_names=None):
        folder_path = self._get_unpickled_folder(folder_name, resize)
        os.makedirs(folder_path, exist_ok=True)
        self._do_store(images, folder_path, image_names)

    def get_folder_path(self, folder_name):
        """
        if folder_name is not a valid path, we expect it to be a path within resources/images
        """
        if os.path.exists(folder_name):
            return folder_name
        return os.path.join(self.img_folder, folder_name)

    def _get_all_image_names(self, folder_name):
        folder_path = self.get_folder_path(folder_name)
        image_names = [img for img in os.listdir(folder_path) if img.endswith(".png")]
        return natsort.natsorted(image_names)


    def _get_all_image_paths(self, folder_name):
        folder_path = self.get_folder_path(folder_name)
        return [os.path.join(folder_path, img) for img in self._get_all_image_names(folder_name)]

    def _do_pickle(self, folder_name, image_array, resize: Resize = Resize(100)):
        timestamp = int(datetime.datetime.now().timestamp())
        filename = self._get_pickled_filename(folder_name, resize, timestamp)

        pickled_folder = self._get_pickled_folder(folder_name, resize)
        file_path = os.path.join(pickled_folder, filename)

        os.makedirs(pickled_folder, exist_ok=True)
        with open(file_path, 'wb+') as fp:
            pickle.dump(image_array.astype('float32'), fp, protocol=4)

        return filename

    def _get_pickled_folder(self, folder_name, resize: Resize = Resize(100)):
        folder_path = self.get_folder_path(folder_name)
        return os.path.join(folder_path, "pickledx%s" % str(resize).replace(os.path.sep, "_"))

    def _get_unpickled_folder(self, folder_name, resize: Resize):
        folder_path = self.get_folder_path(folder_name)
        return os.path.join(folder_path, "unpickledx%s" % str(resize).replace(os.path.sep, "_"))

    def _get_pickled_filename(self, folder_name, resize: Resize = Resize(100), timestamp=None):
        if timestamp is None:
            return self._get_latest_pickle(folder_name, resize)
        if os.path.exists(folder_name):
            folder_name = os.path.basename(folder_name)
        return folder_name + "_pickled##" + str(timestamp) + ".pickle"

    def _get_index_filename(self, pickled_filename):
        return pickled_filename + ".index"

    def _get_latest_pickle(self, folder_name, resize: Resize = Resize(100)):
        pickled_folder = self._get_pickled_folder(folder_name, resize)
        filenames = [f for f in os.listdir(pickled_folder) if f.endswith(".pickle")]
        filenames.sort(reverse=True)
        return filenames[0]

    def _get_empty_np_array(self, num_images, image_dimensions):
        return np.zeros((num_images, *image_dimensions), dtype="float32")

    def _read_and_dump_image(self, image_array, i, path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = io.imread(path, as_gray=False)
        resized = transform.resize(image, image_array[0].shape)
        if len(image_array.shape) == 4:
            image_array[i, :, :, :] = resized
        else:
            image_array[i, :, :] = resized

    @staticmethod
    def _get_image_dimensions(image_path, resize: Resize = Resize(100)):
        image = io.imread(image_path, as_gray=False)
        w, h = resize.resize((image.shape[0], image.shape[1]))
        w, h = int(w), int(h)
        if len(image.shape) == 3:
            return w, h, image.shape[2]
        else:
            return w, h

    def _create_pickle_indices(self, folder_name, resize, pickle_filename):
        index_filename = self._get_index_filename(pickle_filename)
        pickle_folder = self._get_pickled_folder(folder_name, resize)
        file_path = os.path.join(pickle_folder, index_filename)

        image_names = self._get_all_image_names(folder_name)

        with open(file_path, 'w+') as f:
            for i, name in enumerate(image_names):
                f.write("%d=%s," %(i, name))

    def _get_pickle_indices(self, folder_name, resize, pickle_filename):
        index_filename = self._get_index_filename(pickle_filename)
        pickle_folder = self._get_pickled_folder(folder_name, resize)
        file_path = os.path.join(pickle_folder, index_filename)

        indices = list()
        if not os.path.exists(file_path):
            self.log.w("Pickle indices not found!")
            return None

        with open(file_path, 'r') as f:
            s = f.read()
            for t in s.split(","):
                if t != "":
                    indices.append(t.split("=")[1])
        return indices


imgstore = ImageStore()


