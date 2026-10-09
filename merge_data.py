# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:54:13 2026

@author: MAKSIM
"""

import os
import pandas as pd


rows = []
for ext_folder in ['train', 'test']:
    for int_folder in ['neg', 'pos']:
        cur_sbufolder_path = os.path.join('aclImdb', ext_folder, int_folder)
        for file_name in os.listdir(cur_sbufolder_path):
            if not file_name.endswith('.txt'):
                continue
            file_path = os.path.join(cur_sbufolder_path, file_name)
            with open(file_path, 'r', encoding='UTF-8') as f:
                text = f.read()
                if int_folder == 'pos': 
                    label = 'POSITIVE'
                else:
                    label = 'NEGATIVE'
                rows.append(
                    {
                        'text': text,
                        'label': label,
                        'split': ext_folder
                    } 
                )

df = pd.DataFrame(rows)
df.to_csv('stanford_dataset_merged.csv', index=False)