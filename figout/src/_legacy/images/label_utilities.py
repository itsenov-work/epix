from _legacy.cards import CardReader
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer


def mlbLabeller(labels):
    if labels.ndim == 1:
        mlb = LabelBinarizer()
    else:
        mlb = MultiLabelBinarizer()
    mlb.fit(labels)
    return mlb


def labelGenerator(n_labels):
    """Seems like every label is associated with a power of 2"""
    return 2 ** np.arange(n_labels)


def labelTransformer(labels):
    return np.log(labels)


def id_standardizer(name_list):
    ids = np.zeros(len(name_list))
    for i, name in enumerate(name_list):
        if name.endswith('.png'):
            ids[i] = int(name[:-4])
        else:
            ids[i] = int(name)
    return ids


def id_to_monster_type_and_attr(id_list):
    reader = CardReader()
    labels_race = np.zeros(len(id_list))
    labels_attr = np.zeros(len(id_list))
    for i, entry in enumerate(id_list):
        if entry.endswith('.png'):
            entry = int(entry[:-4])
        else:
            entry = int(entry)
        reader.setID(ID=entry)
        labels_race[i] = reader.getRace()
        labels_attr[i] = reader.getAttribute()

    return np.c_[labels_race, labels_attr]


def id_to_attr(id_list, transform=False):
    reader = CardReader()
    ids = id_standardizer(id_list)
    labels = np.zeros_like(ids)
    for i in range(len(labels)):
        reader.setID(ID=ids[i])
        labels[i] = reader.getAttribute()
        if transform:
            labels = labelTransformer(labels)
    return labels


def id_to_type(id_list, transform=False):
    reader = CardReader()
    ids = id_standardizer(id_list)
    labels = np.zeros_like(ids)
    for i in range(len(labels)):
        reader.setID(ID=ids[i])
        labels[i] = reader.getRace()
        if transform:
            labels = labelTransformer(labels)
    return labels


def id_to_description(id_list):
    reader = CardReader()
    ids = id_standardizer(id_list)
    descriptions = []
    for i in range(len(ids)):
        reader.setID(ID=ids[i])
        descriptions.append(reader.getDescription())
    return descriptions


def id_to_name(id_list):
    reader = CardReader()
    ids = id_standardizer(id_list)
    names = []
    for i in range(len(ids)):
        reader.setID(ID=ids[i])
        names.append(reader.getName())
    return names

