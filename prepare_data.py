from __future__ import print_function

import csv
from shutil import copyfile
import os.path
import os
import shutil


def get_audio_rank():
    path = "/home/cuongdm9/data/zalo_ai/hit_song/audio-classifier-keras-cnn/"
    with open('./MillionSong/train_rank.csv') as tsvfile:
        csv_reader = csv.reader(tsvfile, dialect='excel-tab')
        i = 0
        for row in csv_reader:
            print(f'Processed {row[0].split(",")} lines.')
            if i != 0:
                songs = row[0].split(",")
                target = path + "Samples/rank/" + songs[1]
                src = path + "MillionSong/train/" + songs[0] + ".mp3"
                if os.path.exists(target):
                    shutil.copy(src, target)
                else:
                    os.mkdir(target)
                    shutil.copy(src, target)
            i = i + 1


if __name__ == '__main__':
    get_audio_rank()
