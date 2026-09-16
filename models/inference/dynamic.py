import sys
sys.path.insert(0, '../lib')
import os
import torch
import pandas as pd
import numpy as np
from lightning.pytorch import Trainer
from Progress.L_progress import CrossEntropyLoss_base
from Dataset.Merra2_dataset import LData, Merra2_full
from Model.PointOut.ResNet_classification import Resnet
from Utils.Seed import set_all_seeds
from Utils.Metrics import *
from config_loader import CONFIG

SINGLE_VAR = CONFIG.DYNAMIC_MODEL_DATASET.SINGLE_VAR
PRESS_VAR = CONFIG.DYNAMIC_MODEL_DATASET.PRESS_VAR
ADD_VAR = CONFIG.DYNAMIC_MODEL_DATASET.ADD_VAR
PRESS_LEVEL = CONFIG.DYNAMIC_MODEL_DATASET.PRESS_LEVEL
LEVEL = len(PRESS_LEVEL)

# number of channels after concat
inp_channels = len(SINGLE_VAR) + LEVEL * (len(PRESS_VAR)+len(ADD_VAR))
print(f"Input channel for this model is {inp_channels}")
print(f"3D variables to be used {PRESS_VAR}")
print(f"Surface variables to be used {SINGLE_VAR}")
print(f"Additional variables to be used {ADD_VAR}")

#==========
# dynamic prediction
#==========
def predict(seed = 42, batch_size = 64, num_workers = 2):
    
    set_all_seeds(seed)
    
    # load config
    data_path = CONFIG.DYNAMIC_DOMAIN.SLIDE_DIR
    model_temp = CONFIG.DYNAMIC_DOMAIN.MODEL_TEMP
    prediction_csv = CONFIG.DYNAMIC_DOMAIN.PREDICTION_CSV
    print(f"model template to be used for prediction is {model_temp}")

    # init dataset
    ds = LData(DatasetClass=Merra2_full,
               train_path = os.path.join(data_path, 'data.csv'),
               val_path = os.path.join(data_path, 'data.csv'),
               test_path = os.path.join(data_path, 'data.csv'),
               predict_path = os.path.join(data_path, 'data.csv'),
               batch_size = batch_size,
               num_workers = num_workers,)
    
    # init model
    
    model = Resnet(inp_channels=inp_channels, # hard coded model config
                   num_class=2,
                   vector=False)
    
    pos_weight = 6 # hard coded model config
    if pos_weight > 0:
        class_weight = np.array([
            (pos_weight + 1) / (2 * pos_weight),
            (pos_weight + 1) / 2
        ])

    trainer = Trainer(accelerator='auto', devices=1,)

    # prediction phase step by step
    df = pd.read_csv(os.path.join(data_path, 'data.csv'))
    for step in range(2, 20, 2): # import from config
        print(f"### model to be used for prediction is {model_temp.format(step)} ###")
        L_model = CrossEntropyLoss_base.load_from_checkpoint(
            model_temp.format(step), # temporary 2, will be changed to 'step'
            model = model,
            class_weight = class_weight)
        
        pred = trainer.predict(L_model, datamodule=ds)
        pred = torch.cat(pred, dim=0).cpu().detach().numpy()

        df[f'Score_step{step}'] = pred
        del L_model
    
    # export results
    df.to_csv(prediction_csv, index=False)

if __name__ == "__main__":
    predict()
