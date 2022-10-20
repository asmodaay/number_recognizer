import torch
from IPython.display import Audio
from num2words import num2words
import numpy as np
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift,Gain
import pandas as pd
import os


def gen_sample(path : str) -> None:
    '''
    Function generates random number less 100000 , augemnt it in 5 copies and writes.
    '''
    number = np.random.randint(100000)
    text = num2words(number,lang = 'ru')
    sp_id = np.random.randint(len(speakers))
    wav = model.apply_tts(text=text,
                        speaker=speakers[sp_id],
                        sample_rate=sample_rate).numpy()
    sf.write(path.format(number),wav,sample_rate)
    augment(path.format(number),5,filters)

def augment(path : str ,n_copies : int ,filters : Compose) -> list:
    
    '''
    Function augment input audio in 'n_copies' with 'filters'
    '''
    
    wav,sr = sf.read(path)
    new_paths = []
    for i in range(n_copies):
        new_wav = filters(wav,sr)
        new_path = path.replace('.wav','') + f'_{i}.wav'
        sf.write(new_path,new_wav,sr)
        new_paths.append(new_path)
        
    return new_paths



#create filters
filters = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.8),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.8),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.8),
    Gain(p=0.8)
])


#load open source tts model
language = 'ru'
model_id = 'v3_1_ru'
sample_rate = 24000
speakers = ['aidar', 'baya', 'kseniya', 'xenia', 'eugene']
device = torch.device('cpu')

model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language=language,
                                     speaker=model_id)


#generate 20000 files
gen_sample('numbers2/gen/{}.wav') for _ in range(20000)

#create metadata fil
files = sorted(os.listdir('numbers2/gen'))[1:] # drop .ipynb_checkpoints
files = pd.DataFrame(files,columns = ['path'])
files['transcription'] = files['path'].apply(lambda x : num2words(int(x.split('.')[0].split('_')[0]),lang='ru'))
files['path'] = 'numbers2/gen/' + files['path']
files.to_csv('numbers2/gen.csv',index=False)