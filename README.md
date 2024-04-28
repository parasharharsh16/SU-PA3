# Speech Understanding-Programming Assignment 3
This assignment is intended to submit for Speech Understanding class assignment at IIT-J. This repo will interact and finetune speech model for Deep Fake Detection.

##Dataset
###Dataset Download Link
- Custom Dataset: 'https://iitjacin-my.sharepoint.com/personal/ranjan_4_iitj_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Franjan%5F4%5Fiitj%5Fac%5Fin%2FDocuments%2FDataset%5FSpeech%5FAssignment%2Ezip&parent=%2Fpersonal%2Franjan%5F4%5Fiitj%5Fac%5Fin%2FDocuments&ga=1'

- FOR dataset: 'https://www.eecs.yorku.ca/~bil/Datasets/for-2sec.tar.gz&sa=D&source=apps-viewer-frontend&ust=1714317656652133&usg=AOvVaw3t72LbwG7hK_jCb1FN-Dxh&hl=en'

##Models
- Pre-Trained Model can be downloaded from 'https://github.com/TakHemlata/SSL_Anti-spoofing'
- Fine-Tuned Model can be downloaded from 'https://drive.google.com/file/d/1S4znYpiCDt7bjO-NKWUfqOAyPVxY3fXT/view?usp=sharing'

## Set Up
Please put all the dataset folder after unziping into 'dataset' folder and models in 'models' folder.
To fine-tune the model again, delete the curruntly available finetuned model or rename it.

## Requirements
To install the requirements run-
- 'pip install -r requirements.txt'
p.s. Install Fairseq from its git repository

#Run Code

- Pre-Trained Model Evaluation 'python main.py --mode pretrained'
- Fine-Tuned Model Evaluation 'python main.py --mode finetuned'

#### Results plots png will be created in current folder with following name
- fine-tuned 'roc_curve_fine-tuned.png'
- pre-trained 'roc_curve_pre-trained.png'
