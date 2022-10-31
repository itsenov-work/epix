from struct import unpack
import os

from tqdm import tqdm

marker_mapping = {
    0xffd8: "Start of Image",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
}


class JPEG:
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()

    def decode(self):
        data = self.img_data
        while (True):
            marker, = unpack(">H", data[0:2])
            # print(marker_mapping.get(marker))
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            elif marker == 0xffda:
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2 + lenchunk:]
            if len(data) == 0:
                break





def main():
    bads = []
    from data.google_databox import GoogleDataBox
    dataset_box = GoogleDataBox('drones_poles_metal')
    dataset_box.prepare()
    dataset_files = dataset_box.local_path
    for root, dirs, files in os.walk(dataset_files):
        for file in files:
            filepath = os.path.join(root, file)
            image = JPEG(filepath)
            try:
                image.decode()
            except:
                print(f"Bad JPEG: ${filepath}")
                bads.append(filepath)

    for path in bads:
        os.remove(path)

if __name__ == '__main__':
    main()