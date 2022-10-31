import os
import os.path as osp

import pandas as pd
import tensorflow as tf



def get_basic_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def remove_nan_rows(df):
    isNaN = df.isnull()
    df = df[isNaN.sum(axis=1) < 40]
    return df


def remove_nan_cols(df):
    return df.dropna(axis=1)


def remove_nans(filepath):
    names = pd.read_csv('/Users/igeorgievtse/Downloads/csv/depression_complete.csv', nrows=0)
    with open('resources/depression_cpg.txt', 'w+') as f:
        f.write(names.columns)
        print("Saved column names")
    cg_100 = names.columns[:100]
    df = pd.read_csv(filepath, usecols=cg_100)
    cg_cols = [col for col in df.columns if col.startswith('cg')]
    df = df[[*cg_cols, 'is_case']]
    df = remove_nan_rows(df)
    df = remove_nan_cols(df)
    return df


if __name__ == '__main__':
    file = '/Users/igeorgievtse/Downloads/csv/depression_complete.csv'
    df = remove_nans(file)
    cg_cols = [col for col in df.columns if col.startswith('cg')]
    target = df['is_case'].astype(float)
    num_features = df[cg_cols]

    tf_features = tf.convert_to_tensor(num_features)
    tf_target = tf.convert_to_tensor(target)

    model = get_basic_model()
    model.fit(tf_features, tf_target, epochs=1000, batch_size=32, validation_split=0.2)