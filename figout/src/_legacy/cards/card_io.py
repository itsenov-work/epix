import numpy as np
from tensorflow.keras import backend as K
from _legacy.images.image_store import imgstore
from utils.logger import LoggerMixin


class CardIO(LoggerMixin):
    """
    A class for I/O of cards
    """
    folder_name = "cards"

    def __init__(self, size_percent=100):
        """
        size_percent: how much an image is compressed during pickling. 100% is full-size image.
        """
        super(CardIO, self).__init__()
        self.size_percent = size_percent

    def provide_cards(self, get_names):
        try:
            return self.get_cards(get_names=get_names)
        except FileNotFoundError as e:
            self.make_cards()
            return self.get_cards(get_names=get_names)

    def get_cards(self, store_as_files=False, get_names=False):
        """
        Unpickles cards and returns them in memory
        """
        self.log.i("Getting cards...")
        try:
            raw_cards = imgstore.unpickle(self.folder_name, self.size_percent,
                                          store_as_files=store_as_files, get_image_names=get_names)
        except FileNotFoundError as e:
            self.log.e("ERROR! Your cards are probably not pickled in this format.")
            raise e

        names = None
        if get_names:
            raw_cards, names = raw_cards

        num_cards = raw_cards.shape[0]
        self.log.i("Got %d cards! Processing them in memory..." % num_cards)
        cards = self._process_cards(raw_cards)
        self.log.i("Processed cards!")
        self.log.end()

        if get_names:
            return cards, names
        return cards

    def make_cards(self, num_cards=0, delete_older=False):
        '''
        Pickles cards in folder "resources/images/cards"
        num_cards: how many cards to pickle. If 0, pickles all of them.
        '''
        self.log.i("Making %d cards:" % num_cards)
        imgstore.pickle("cards", self.size_percent, num_images=num_cards, delete_older=delete_older)
        self.log.i("Successfully pickled %d cards!" % num_cards)
        self.log.end()

    def _process_cards(self, raw_cards):
        '''
        Normalizes and reshapes cards for learning
        '''
        dims = (raw_cards.shape[1], raw_cards.shape[2])
        normalized_cards = self._normalize_cards(raw_cards)
        cards = self._backend_reshape(normalized_cards, *dims)
        return cards

    def _backend_reshape(self, x_train, img_rows, img_cols):
        """Such errors are routinely produced due to the different image format used by the Theano & TensorFlow backends
        for Keras. In your case, the images are obviously in channels_first format (Theano), while most probably you use
         a TensorFlow backend which needs them in channels_last format.
         """
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
            input_shape = (3, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
            input_shape = (img_rows, img_cols, 3)
        return x_train

    def _normalize_cards(self, cards):
        return cards/np.max(cards)


