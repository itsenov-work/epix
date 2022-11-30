import os
import shutil
import csv
import json
from ewas.ewas_dh_utils import EWASDataHubDiseases
from utils.logging import Logger, my_tqdm
from utils.csv_utils import read_csv_line

logger = Logger('Samplifier')


def get_disease_file(disease):
    return f'{disease}.csv'


def get_disease_filepath(local_path, disease):
    return os.path.join(local_path, get_disease_file(disease))


def get_save_folder(local_path, disease):
    return os.path.join(local_path, 'Samplified Diseases', disease)


def clear_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)




def save_sample(sample, save_folder):
    csv_filename = sample[0] + '.csv'
    csv_filepath = os.path.join(save_folder, csv_filename)
    with open(csv_filepath, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(sample)


def save_metadata(metadata, save_folder):
    metadata_filename = 'metadata.json'
    with open(os.path.join(save_folder, metadata_filename), 'w+') as f:
        json.dump(metadata, f)


def save_cg_list(cg_names, save_folder):
    cg_list = 'cg_list.csv'
    with open(os.path.join(save_folder, cg_list), 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(cg_names)


def parse_disease_csv(filepath):
    metadata = dict()
    samples = []
    cg_names = []

    with open(filepath, 'r') as f:
        l = f.readline()
        l = f.readline()

        sample_ids = read_csv_line(l)[1:]

        for _id in sample_ids:
            samples.append([_id])
            metadata[_id] = {}
        logger.i("Reading disease file lines:")
        for i, line in my_tqdm(enumerate(f)):
            if line.startswith('ch'):
                continue
            is_cg = line.startswith('cg')
            line = read_csv_line(line)
            values = line[1:]
            name = line[0]
            if is_cg:
                cg_names.append(name)
                for i, v in enumerate(values):
                    samples[i].append(v)
            else:
                for i, v in enumerate(values):
                    metadata[sample_ids[i]][name] = v
    return metadata, samples, cg_names


def samplify_ewas_datahub_disease(local_path, disease, override=True):
    disease = str(disease)
    logger.start(f'Splitting disease {disease} into samples.')
    disease_filepath = get_disease_filepath(local_path, disease)
    save_folder = get_save_folder(local_path, disease)
    logger.i(f"Files will be saved in {save_folder}.")
    if override:
        logger.i("Override is on. Clearing save folder...")
        clear_folder(save_folder)
        logger.ok("Folder cleared")
    else:
        if os.path.exists(save_folder):
            logger.end("Save folder exists and override is not on. Skipping procedure.")

    logger.i("Parsing disease CSV...")
    try:
        metadata, samples, cg_names = parse_disease_csv(disease_filepath)
    except FileNotFoundError as e:
        logger.error(f'Disease csv not found at path: {disease_filepath}')
        raise
    except Exception as e:
        logger.e('There was an error parsing the diseases csv file. Likely a formatting error in the file itself.')
        raise

    logger.i(f"Found {len(samples)} samples in CSV file.")
    logger.i(f"Each sample contains {len(cg_names)} cpgs.")
    logger.s("Parsed disease CSV successfully!")

    logger.i("Saving samples...")
    for sample in my_tqdm(samples):
        save_sample(sample, save_folder)
    logger.ok("Samples saved.")
    logger.i("Saving metadata...")
    save_metadata(metadata, save_folder)
    logger.ok("Metadata saved.")
    logger.i("Saving cpg names...")
    save_cg_list(cg_names, save_folder)
    logger.ok("Cpg names saved")
    logger.s(f"Samplification of disease {disease} was successful.")
    logger.end()


if __name__ == '__main__':
    local_path = r'C:\Users\itsen\workspaces\epix\epigenetic data\diseases\Clean Data'
    for disease in EWASDataHubDiseases:
        samplify_ewas_datahub_disease(local_path=local_path, disease=disease)
