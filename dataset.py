import numpy as np
import pandas as pd

import torch
import torchaudio
from torch.utils.data import Dataset


def Random_Crop_Audio(speech_array):
    return


class SQA_Dataset(Dataset):
    def __init__(self, df_dataset, transform=None, two_crop=True):
        self.audios = df_dataset["path"]
        self.transform = transform
        self.two_crop = two_crop

    def __getitem__(self, index):
        audio = self.audios[index], self.mos_score[index]
        speech_array, _ = torchaudio.load(audio)

        if self.transform is not None:
            speech_array = self.transform(speech_array)

        if self.two_crop:
            sub_audio1, sub_audio2 = Random_Crop_Audio(speech_array)

        return sub_audio1, sub_audio2


def get_data(train_path, valid_path):
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(valid_path)

    train_dataset = SQA_Dataset(df_train, train=True, transform=None)
    val_dataset = SQA_Dataset(df_val, train=False, transform=None)

    return train_dataset, val_dataset
