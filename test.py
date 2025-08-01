import torch
from model.load import load_model_frommmf

def convertconfig(ckpt):
    newconfig = {}
    newconfig['config']={}
    model_type = ckpt['config']['model']
    
    for key, val in ckpt['config']['model_config'][model_type].items():
        newconfig['config'][key]=val
        
    for key, val in ckpt['config']['dataset_config']['rnaseq'].items():
        newconfig['config'][key]=val
        
    if model_type == 'performergau_resolution':
        model_type = 'performer_gau'
    
    import collections
    d = collections.OrderedDict()
    for key, val in ckpt['state_dict'].items():
        d[str(key).split('model.')[1]]=val
        
    newconfig['config']['model_type']=model_type
    newconfig['model_state_dict']=d
    newconfig['config']['pos_embed']=False
    newconfig['config']['device']='cuda'
    return newconfig

model, config = load_model_frommmf('./model/models/models.ckpt')
a = 'model/models/best_model_zinb.ckpt'
a = torch.load(a)['state_dict']
a = {key.replace('model.','').replace('grn', 'go'):val for key, val in a.items()}
b = 'model/models/best_model_finetune.ckpt'
b = torch.load(b)['state_dict']
b = {key.replace('model.','').replace('grn', 'go'):val for key, val in b.items()}
res = {'m1': a, 'm2': b, 'config': config}
torch.save(res, 'model.pt')