if __name__ == '__main__':
    path_to_data = '/Users/igeorgievtse/Downloads/metal_pole_classificator/train'
    path_to_tfrecords = 'resources/tfrecords/metal_poles'
    import os
    os.makedirs(path_to_tfrecords, exist_ok=True)
    from data.labelled_images_manual import serialize_images_to_tfrecord
    serialize_images_to_tfrecord(
        'metal_poles',
        path_to_data,
        path_to_tfrecords,
        (250, 250),
    )