# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 19:35:47 2021

@author: owner

NG folder_dir='D:/Edogawa/Edogawa_BERT/keikakiroku/original12files'
good folder_dir='D:/Edogawa/Edogawa_BERT/keikakiroku/original12files/'

"""


import pandas as pd
import glob


def merging_csv(folder):   
    drive_dir=folder
    csv_files = glob.glob(drive_dir+'*.csv')
    files = []
    for f in csv_files:
        files.append(pd.read_csv(f)) 
    df = pd.concat(files)
    return df

