## Instruction
# 1) All files must be in the same folder 'dir'
# 2) Required files must be named as followed
#    > kp file: 'kp.json'
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

with open(f'{dir}/kp.json') as file:
    data_kp = json.load(file)
print(data_kp['datetime'][3])

with open(f'{dir}/VAL.txt') as file:
    lines = file.readlines()


## Deal with .txt file
print('------------------------------> Read .txt')
data_VAL = {'datetime': [], 'SN': [], 'F10.7': []}
for line in lines:
    data_VAL['datetime'].append(line[:10].replace(' ','-'))
    data_VAL['SN'].append(line[135:138])
    data_VAL['F10.7'].append(line[139:147])

with open(f'{dir}/SN.json', 'w') as file:
    json.dump(data_VAL, file)
    

# Convert to .csv
print('------------------------------> Convert to .csv')
doc = {
    'datatime': [],
    'Kp': [],
    'SN': [],
    'F10.7': [],
    'Hp60': [],
    'sin_time': [],
    'cos_time': []
}

df_data = []

for i, timestamp in enumerate(data_HP60['datetime']):
    # doc['datatime'].append(timestamp)
    # doc['Kp'].append(data_kp['Kp'][i//3])
    # doc['SN'].append(data_VAL['SN'][i//24])
    # doc['SN'].append(data_VAL['F10.7'][i//24])
    # doc['Hp60'].append(data_HP60['Hp60'][i])
    time = int(timestamp.split('T')[1].split(':')[0])
    
    df_data.append({
        'datatime': timestamp,
        'sin_time': np.sin(2*np.pi*time/24),
        'Kp': data_kp['Kp'][i//3],
        'SN': data_VAL['SN'][i//24],
        'F10.7': data_VAL['F10.7'][i//24],
        'Hp60': data_HP60['Hp60'][i],
    })
    
print(f"number of timestamps: {len(doc['datatime'])}")
print('data example')
df = pd.DataFrame(df_data)    
print(df.head())

## Save
df.to_csv(f'{dir}/dataset.csv', index= False)
print(f'\nsave .csv file to ./{dir}/dataset_time.csv\n')