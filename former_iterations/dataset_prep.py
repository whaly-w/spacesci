## Instruction
# 1) All files must be in the same folder 'dir'
# 2) Required files must be named as followed
#    > kp file: 'kp.json'
#    > Hp60 file: 'Hp60.json'
#    > Sunspots file: 'VAL.txt'

import pandas as pd
import json
import argparse


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
data_SN = {'datetime': [], 'SN': []}
for line in lines:
    data_SN['datetime'].append(line[:10].replace(' ','-'))
    data_SN['SN'].append(line[135:138])

with open(f'{dir}/SN.json', 'w') as file:
    json.dump(data_SN, file)
    

# Convert to .csv
print('------------------------------> Convert to .csv')
doc = {
    'datatime': [],
    'Kp': [],
    'SN': [],
    'Hp60': []
}

df_data = []

for i, timestamp in enumerate(data_HP60['datetime']):
    doc['datatime'].append(timestamp)
    doc['Kp'].append(data_kp['Kp'][i//3])
    doc['SN'].append(data_SN['SN'][i//24])
    doc['Hp60'].append(data_HP60['Hp60'][i])
    
    df_data.append({
        'datatime': timestamp,
        'Kp': data_kp['Kp'][i//3],
        'SN': data_SN['SN'][i//24],
        'Hp60': data_HP60['Hp60'][i]
    })
    
print(f"number of timestamps: {len(doc['datatime'])}")
print('data example')
df = pd.DataFrame(df_data)    
print(df.head())

## Save
df.to_csv(f'{dir}/dataset.csv', index= False)
print(f'\nsave .csv file to ./{dir}/dataset.csv\n')