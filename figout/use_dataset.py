if __name__ == '__main__':
    import tensorflow as tf
    import os
    import os.path as osp
    root = 'resources/tfrecords/metal_poles'
    files = [osp.join(root, file) for file in os.listdir(root) if file.endswith(".tfrecord")]
    ds = tf.data.TFRecordDataset(files)

    i = 0
    for d in ds:
        feature_description = {
            'image': tf.io.FixedLenFeature((), tf.string),
            'label': tf.io.FixedLenFeature((), tf.int64),
            'height': tf.io.FixedLenFeature((), tf.int64),
            'width': tf.io.FixedLenFeature((), tf.int64),
            'depth': tf.io.FixedLenFeature((), tf.int64),
            'image/filepath': tf.io.VarLenFeature(tf.string),
        }
        d = tf.io.parse_single_example(d, feature_description)
        image = tf.io.parse_tensor(d['image'], out_type=float)
        image_shape = [d['height'], d['width'], d['depth']]
        image = tf.reshape(image, image_shape)
        print(i)
        i+=1