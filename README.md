## Overall About the Model
This model uses **three features** — Hp60, SN, and F10.7 — as and input to predict Hp60 in the next six hours.
- ``` model.py ``` is the model architecture.
- ``` main.py ``` is the python file for testing the model
- ``` train_v3.py ``` is the python file for training the data
- ``` dataset_prep_V2.py ``` is the python file for preparing the data

## Data Preparation
### Data Gathering
These are the link to the data for testing
- Hp60
    - https://kp.gfz-potsdam.de/en/hp30-hp60/data
    - download as JSON file and name it **Hp60.json**
- SN & F10.7:
    - https://kp.gfz-potsdam.de/en/data
    - use Geomagnetic and solar indices (Kp, ap, Ap, SN, F10.7). Copy and paste the text into a file and named it **VAL.txt**

### Data Restructure
1. Create a folder for the new data and put Hp60.json and VAL.txt inside the folder
2. run **dataset_prep_V2.py** with argument --dir as the folder's name <br/>
``` python3 dataset_prep_V3.py --dir folderName ```

## Test the Model
run **main.py** and set the argument --dir_data as the dataset directory (the name of the folder you created)  <br/>
``` python3 main.py --dir_data folderName ```
