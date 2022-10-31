import os

from utils.google_storage import GoogleStorage
from utils.logger import LoggerMixin
from _legacy.images.image_store import imgstore
from _legacy.cards import CardIO


class CardDownloader(LoggerMixin):
    def provide_cards(self):

        def end_procedure():
            self.log.i("Cards provided!")
            self.log.end()

        self.log.i("Providing cards...")
        cards_folder = imgstore.get_folder_path("cards")
        if not os.path.exists(imgstore.get_folder_path("cards")):
            self.log.i("The \"cards\" folder does not exist!")
            self.log.i("Creating folder and downloading cards.")
            os.makedirs(cards_folder)
            self.download_cards()
            end_procedure()
            return

        self.log.i("Cards folder found. Checking if all cards are there...")
        # If all cards are available, we good
        if self._check_for_raw_cards():
            self.log.i("All cards are already available.")
            self.log.end()
            return

        self.log.i("There are cards missing. Trying to find them in a pickled file of size 100%.")


        try:
            # Case where we have pickled cards in 100% size
            CardIO(100).get_cards(store_as_files=True)
            self.log.i("Unpickled cards successfully! Now moving them to main cards folder.")
            import shutil

            source_dir = os.path.join(cards_folder, "unpickledx100")
            target_dir = cards_folder

            file_names = os.listdir(source_dir)

            for file_name in file_names:
                shutil.move(os.path.join(source_dir, file_name), target_dir)

            shutil.rmtree(source_dir)
            if not self._check_for_raw_cards():
                self.log.e("ERROR! Something went wrong! Cards are not all available after unpickling.")
            end_procedure()

        except FileNotFoundError as e:
            self.log.i("No pickle of size 100% found. Trying to find zipped cards.")
            if os.path.exists(GoogleStorage().get_destination_path("cards.tar.gz")):
                self.extract_cards()
                end_procedure()
                return

            self.log.i("No zipped cards found. Will download.")
            self.download_cards()
            if not self._check_for_raw_cards():
                self.log.e("ERROR! Something went wrong! Cards are not all available after downloading.")
                return
            end_procedure()

    def download_cards(self):
        file_path = GoogleStorage().download("cards.tar.gz")
        self.log.i("Cards downloaded in path {}".format(file_path))
        self.extract_cards()

    def extract_cards(self):
        file_path = GoogleStorage().get_destination_path("cards.tar.gz")
        cards_folder = imgstore.img_folder

        self.log.i("Extracting cards to images folder.")
        import tarfile

        tar = tarfile.open(file_path, "r:gz")
        tar.extractall(path=cards_folder)
        tar.close()
        self.log.i("Cards extracted successfully.")

        self.log.i("Deleting non-cards in folder.")
        filenames = self._get_all_cards_filenames()
        cards_folder = imgstore.get_folder_path("cards")
        dir_filenames = os.listdir(cards_folder)

        cards_folder_abs = os.path.abspath(cards_folder)

        [os.remove(os.path.join(cards_folder_abs, file)) for file in dir_filenames if file not in filenames]

    def _check_for_raw_cards(self):
        filenames = self._get_all_cards_filenames()
        cards_folder = imgstore.get_folder_path("cards")
        dir_filenames = os.listdir(cards_folder)

        return all(card in dir_filenames for card in filenames)

    def _get_all_cards_filenames(self):
        filenames_path = os.path.join("resources", "card_utils", "all_cards_filenames")
        with open(filenames_path, 'r') as f:
            filenames = f.read().split(",")
            return filenames
