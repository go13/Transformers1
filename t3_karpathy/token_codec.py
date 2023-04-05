import string
import os
import requests

import torch

from t3_karpathy.commons.commons import AbstractCodec


def download(url: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)

    if os.path.exists(dest_folder + filename):
        print("File already downloaded")
        return

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))


class TokenCodec(AbstractCodec):

    def __init__(self):
        input_path = "input.txt"
        dest_folder = "../"
        src_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

        download(src_url, dest_folder=dest_folder)

        # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
        with open(dest_folder + input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        # text = ""
        # here are all the unique characters that occur in this text
        self.vocab = ''.join(set(text + string.ascii_letters + string.digits))
        #self.vocab = ''.join(set((string.ascii_letters + string.digits).upper()))
        self.chars = sorted(list(self.vocab))
        self.vocab_size = len(self.chars)
        # create a mapping from characters to integers
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        # self.encode = lambda s: [self.stoi[c] for c in s]  # encoder: take a string, output a list of integers
        # self.decode = lambda l: ''.join([self.itos[i] for i in l])  # decoder: take a list of integers, output a string

        # Train and test splits
        self.data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9 * len(self.data))  # first 90% will be train, rest val
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])