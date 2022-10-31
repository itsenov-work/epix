import cv2
import numpy as np
from images.transformers.chain_transformer import ChainTransformer
from _legacy.cards import CardTypes
from _legacy.cards import CardDownloader
from _legacy.cards.card_categorizer import CardSplitter
import os


class CardLayout(object):
    def __init__(self):
        self.card_layout = self.retrieve_layout()
        self.path = 'D:\\yugigan project\\yugiGAN\\resources\\images\\layout'

    def retrieve_layout(self):
        pass

    def create_layout(self):
        reader = CardReader()
        CardDownloader().provide_cards()
        transformer = ChainTransformer([
            CardSplitter(CardTypes.MONSTER)
        ])
        cards, labels = transformer.provide("cards", size=100, get_names=True)
        list_of_types = [16,        # Normal
                         32,        # Effect
                         64,        # Fusion
                         128,       # Ritual
                         8192,      # Synchro
                         8388608,   # XYZ
                         16777216,  # Pendulum
                         67108864   # Link
                         ]
        list_of_attributes = [1,    # Earth
                              2,    # Water
                              4,    # Fire
                              8,    # Wind
                              16,   # Light
                              32    # Dark
                              ]

        card_background = dict()
        card_attribute = dict()
        for card, card_id in zip(cards, labels):
            if not card_background and not card_attribute:
                break
            else:
                reader.setID(card_id)
                ctype = int(reader.getType())
                cattr = int(reader.getAttribute())
                if ctype in list_of_types:
                    self.save_image(self._keep_background(card), str(ctype))
                    list_of_types.remove(ctype)
                if cattr in list_of_attributes:
                    self.save_image(self._keep_attr(card), str(cattr))
                    list_of_attributes.remove(cattr)

    def save_image(self, image, name):
        cv2.imwrite(os.path.join(self.path, name + '.png'), image)


    @staticmethod
    def _keep_background(card):
        # Assume standard size (322, 433)
        card[37:284, 79:306, ] = np.zeros_like(card[37:284, 79:306, ])  # Remove art
        card[19:301, 19:46, ] = np.ones_like(card[19:301, 19:46, ])*card[180, 40, ]  # Remove name
        card[23:296, 325:390, ] = np.ones_like(card[23:296, 325:390, ]) * card[220, 330, ]  # Remove text
        card[23:296, 395:403, ] = np.ones_like(card[23:296, 395:403, ]) * card[220, 330, ]  # Remove ATK/DEF

        return card

    @staticmethod
    def _keep_stars(card):
        return card[30:290, 50:72, ]

    @staticmethod
    def _insert_stars(card, stars):
        card[30:290, 50:72, ] = stars
        return stars

    @staticmethod
    def _keep_attr(card):
        return card[263:296, 19:48, ]

    @staticmethod
    def _insert_attr(card, attr):
        card[263:296, 19:48, ] = attr
        return card


if __name__ == '__main__':
    cw = CardLayout()
    cw.create_layout()




