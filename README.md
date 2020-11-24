# Cogmaps Stress Detection
- git clone https://github.com/ankitpriyarup/CogMapsStressDetection
- cd CogMapsStressDetection
- python3 -m venv env
- .\env\Scripts\activate.bat
- pip3 install -r requirements.txt

### Preprocessing
- Download this folder and place it in root directory, https://mega.nz/folder/g19WUDaJ#cKIvrEnExUjjn72NuTGh3w
- Install Ocra from https://github.com/plotly/orca
- Use preprocessing.py file

- Mental Arithmetic dataset: https://physionet.org/content/eegmat/1.0.0/
- Inside data_mental_arithemetic folder EDF dataset is already converted to CSV using 

### Analysis


## Results graph legends
- 1: Start, Training Begins
- 2: Training Ends, Relax Begins
- 3: Relax Ends, Control Begins
- 4: Control Ends, Rest Begins
- 5: Test Begins
- 6: Test Ends, Rest Begins
- 7: Control Begins
- 8: Relax
- 9: End