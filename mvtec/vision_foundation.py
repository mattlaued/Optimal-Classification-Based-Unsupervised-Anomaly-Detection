# Tool set for create encoding with foundation models
import os
import numpy as np
from PIL import Image

def get_path(base_path):
    # Get all subdirectories in f'{base_path}/test'
    categories = [x for x in os.listdir(f'{base_path}/test') if os.path.isdir(f'{base_path}/test/{x}')]
    # Get all filepaths in each subdirectory
    splits = ['train', 'test']
    
    path_dict = {}
    for split in splits:
        for category in categories:
            # prompt: get the available files
            cur_path = f'{base_path}/{split}/{category}'
            try:
                for file in os.listdir(cur_path):
                    file__ = file.split('.')[0]
                    path_dict[f'{split}-{category}-{file__}'] = f'{cur_path}/{file}'
            except:
                print(f'{split}/{category}/ does not exist')
    
    return categories, path_dict


def cal_emb(processor, model, model_type, path_dict, base_path, agg, cls=True):
    names = []
    embs = []
    for name, path in path_dict.items():
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert(mode='RGB')

        if model_type == 'clip':
            inputs = processor(text=["defective"], images=img, return_tensors="pt", padding=True)
        else:
            inputs = processor(images=[img], return_tensors="pt")

        outputs = model(**inputs)

        if model_type in ('dinov2', 'vit'):
            out = outputs.last_hidden_state
        elif model_type == 'clip':
            out = outputs.image_embeds.unsqueeze(1)
        elif model_type == 'mae':
            out = outputs.logits
            
        for agg_ in agg:
            if agg_ == 'cls' and cls:
                emb = out[:, 0, :].squeeze(0).detach().numpy()
            elif agg_ == 'cls' and not cls:
                continue
            elif agg_ == 'mean' and cls:
                emb = out[:,1:,:].mean(dim=1).squeeze(0).detach().numpy()
            elif agg_ == 'mean' and not cls:
                emb = out.mean(dim=1).squeeze(0).detach().numpy()
            else:
                raise NotImplementedError
            names.append(f'{model_type}-{agg_}-{name}')
            embs.append(emb)
    # Make directory for embeddings
    os.makedirs(f'{base_path}/embeddings', exist_ok=True)
    for n_, e_ in zip(names, embs):
        np.save(f'{base_path}/embeddings/{n_}.npy', e_)
    
    return names, embs
