import torch
import pandas as pd
from nemo.collections.asr.models import EncDecCTCModel
from omegaconf import OmegaConf

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
    

cpt_path = 'checkpoint.ckpt'
config_path = 'config.yaml'
file_path = 'numbers2/test.csv'
    
model_config = load(open(config_path,'r'), Loader=Loader)
model_config = OmegaConf.structured(model_config)


model_config['model']['train_ds']['manifest_filepath'] = None
model_config['model']['validation_ds']['manifest_filepath'] = None
model_config['model']['test_ds']['manifest_filepath'] = None

asr_model = EncDecCTCModel(cfg=model_config['model'], trainer=None)
state_dict = torch.load(cpt_path)['state_dict']
asr_model.load_state_dict(state_dict)


data = pd.read_csv(file_path)

assert 'path' in data.columns

result = asr_model.transcribe(data['path'])

data['number'] = result

data.to_csv('result.csv',index=False)

print('Check result.csv to see result.')