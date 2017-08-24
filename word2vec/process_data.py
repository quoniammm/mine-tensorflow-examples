import tensorflow as tf
import zipfile
from tqdm import tqdm
from six.moves import urllib
import os
from collections import Counter
import utils

# DOWNLOAD_URL = 'http://mattmahoney.net/dc/'
# EXPECTED_BYTES = 31344016
# DATA_FOLDER = './data/'
# FILE_NAME = 'text8.zip'

# 定义进度条
class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def download(file_name, expected_bytes, data_folder, download_url):
    file_path = data_folder + file_name

    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    #判断数据集是否存在
    if os.path.exists(file_path):
        print("数据集已经存在于: " + file_path)
        return file_path

    #下载数据集
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        file_name, _ = urllib.request.urlretrieve(download_url + file_name, file_path, pbar.hook)

    #验证数据是否下载成功
    file_stat = os.stat(file_path)
    if file_stat.st_size == expected_bytes:
        print('数据集已成功下载')
    else:
        raise Exception('文件 ' + file_name + ' 下载失败.请重新下载!') 

def read_data(file_path):
    """ Read data into a list of tokens 
    There should be 17,005,207 tokens
    """
    with zipfile.ZipFile(file_path) as f:
        print(tf.compat.as_str(f.read(f.namelist()[0]))[:1000])
        words = tf.compat.as_str(f.read(f.namelist()[0])).split() 
        # tf.compat.as_str() converts the input into the string
    return words

def build_vocab(words, vocab_size):
    """ Build vocabulary of VOCAB_SIZE most frequent words """
    dictionary = dict()
    count = [('UNK', -1)]
    count.extend(Counter(words).most_common(vocab_size - 1))
    index = 0
    utils.make_dir('processed')
    with open('processed/vocab_1000.tsv', "w") as f:
        for word, _ in count:
            dictionary[word] = index
            if index < 1000:
                f.write(word + "\n")
            index += 1
    index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, index_dictionary
