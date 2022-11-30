import json
import time

import requests


def _post_response(url, body):
    r = requests.post(url, body)
    r.raise_for_status()
    return r.json()


def _get_response(url):
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


def _iterate_response(urls):
    i = 0
    while i < len(urls):
        url = urls[i]
        try:
            result = _get_response(url)
            time.sleep(0.5)
            i += 1
            yield result

        except Exception as e:
            print(f"Retrying url {url} in 5 seconds...")
            time.sleep(5)


def save_json(folder, filename, obj):
    import os
    import os.path as osp
    folder_path = osp.join('../..', 'resources', folder)

    os.makedirs(folder_path, exist_ok=True)
    filepath = osp.join(folder_path, filename + ".json")
    with open(filepath, 'w+') as f:
        json.dump(obj, f, sort_keys=True, indent=4)


def get_json(folder, filename):
    import os
    import os.path as osp
    folder_path = osp.join('../..', 'resources', folder)

    os.makedirs(folder_path, exist_ok=True)
    filepath = osp.join(folder_path, filename + ".json")
    with open(filepath, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    samples = get_json('ewas_datahub_all', 'ewas_datahub_all_metadata')
    samples = [s for s in samples if 'tissue' in s if 'blood' in s['tissue']]
    samples = [s for s in samples if 'disease' in s if 'type 2 diabetes' in s['disease']]
    print(len(samples))
    projects = set([p['project id'] for p in samples])
    print(projects)
    print(len([p for p in projects if p.startswith('GSE')]))
    print(len(projects))