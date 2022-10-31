from _legacy.cards import CardReader, CardTypes, CardRaces
from _legacy.images.image_store import imgstore
from _legacy.images.transformers.image_transformer import ImageTransformer, TransformerModes
import shutil
import os


class CardSplitter(ImageTransformer):
    def __init__(self, card_type):
        super(CardSplitter, self).__init__()
        self.card_type = card_type
        if isinstance(card_type, CardTypes):
            self.key = 'type'
        elif isinstance(card_type, CardRaces):
            self.key = 'race'
        else:
            raise ValueError("card_type must be one of the CardData enum types!")

    def transform_files(self, folder_name, files):
        info = CardReader()
        cards = info.get_all_flag(self.key, self.card_type)
        card_ids = [str(card.getID()) for card in cards]
        cards_folder = imgstore.get_folder_path(folder_name)
        transformed_folder = self.get_folder_path(folder_name)
        for file in files:
            if file.split(".png")[0] in card_ids:
                from_path = os.path.join(cards_folder, file)
                to_path = os.path.join(transformed_folder, file)
                shutil.copy(from_path, to_path)

    def suffix(self):
        return self.card_type.name.lower() + "s"

    def get_transformer_mode(self):
        return TransformerModes.FILE_MODE
