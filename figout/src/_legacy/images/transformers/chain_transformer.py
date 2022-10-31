from _legacy.images.transformers.image_transformer import ImageTransformer


class ChainTransformer(ImageTransformer):
    def __init__(self, transformers):
        super(ChainTransformer, self).__init__()
        self.transformers = transformers

    def make(self, folder_name):
        for transformer in self.transformers:
            transformer.provide(folder_name)
            folder_name = transformer.get_folder_name(folder_name)

    def suffix(self):
        t = self.transformers
        sfx = t[0].suffix()
        for t in self.transformers[1:]:
            sfx = t.get_folder_name(sfx)
        return sfx

    def get_transformer_mode(self):
        return 0
