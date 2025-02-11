## Instruction
# 1) All files must be in the same folder 'dir'
# 2) Required files must be named as followed
#    > Hp60 file: 'Hp60.json'
#    > Sunspots & F10.7 file: 'VAL.txt'

import pandas as pd
import json
import argparse
import numpy as np


## Get arguments
parser = argparse.ArgumentParser(description='Add these argument for training')
parser.add_argument('--dir', required= True, help='directory for saving trianed mode')
args = parser.parse_args()
dir = args.dir


## Deal with JSON files
print('------------------------------> Read JSON ')
with open(f'{dir}/Hp60.json', 'r') as file:
    data_HP60 = json.load(file)
print(data_HP60['meta'])
for i in data_HP60:
    print(i)


## Deal with .txt file
with open(f'{dir}/VAL.txt') as file:
    lines = file.readlines()

print('------------------------------> Read .txt')
data_VAL = {'datetime': [], 'SN': [], 'F10.7': []}

for line in lines:
    data_VAL['datetime'].append(line[:10].replace(' ','-'))
    data_VAL['SN'].append(line[135:138])
    data_VAL['F10.7'].append(line[139:147])

with open(f'{dir}/SN.json', 'w') as file:
    json.dump(data_VAL, file)
    

## Convert to .csv
print('------------------------------> Convert to .csv')
df_data = []

for i, timestamp in enumerate(data_HP60['datetime']):
    
    df_data.append({
        'datatime': timestamp,
        'SN': data_VAL['SN'][i//24],
        'F10.7': data_VAL['F10.7'][i//24],
        'Hp60': data_HP60['Hp60'][i],
        'logHp60': np.log(data_HP60['Hp60'][i] + 1)
    })
    
print('data example')
df = pd.DataFrame(df_data)    
print(df.head())

## Save
df.to_csv(f'{dir}/dataset.csv', index= False)
print(f'\nsave .csv file to ./{dir}/dataset.csv\n')