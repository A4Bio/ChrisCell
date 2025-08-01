import scanpy as sc
import torch
import sys
sys.path.append('/storage/guoxiaopeng/wangjue/VQCell/FoldToken2')
sys.path.append('/storage/guoxiaopeng/wangjue/VQCell/FoldToken2/FoldToken4')
from FoldToken4.main import create_parser
from FoldToken4.model_interface import MInterface
from sklearn.cluster import KMeans
from torch_scatter import scatter_mean
import random
from transformers import AutoTokenizer
from tqdm import tqdm
tokenizer = AutoTokenizer.from_pretrained('microsoft/biogpt')
def get_data(datas):

    feature_sequences = []
    feature_indexes = []

    for data in tqdm(datas):
        extract_feature_names = ['tissue', 'cell_type', 'disease', 'development_stage','sex']
        feature_sequence = []
        feature_index = []
        for i, feature_name in enumerate(extract_feature_names):
            feature = tokenizer.encode(data.obs[feature_name][0], add_special_tokens=False)
            feature_sequence += feature
            feature_index += len(feature)*[i]
        if len(feature_sequence) < 30:
            feature_sequence += [tokenizer.pad_token_id] * (30 - len(feature_sequence))
            feature_index += [-1] * (30 - len(feature_index))
        feature_sequences.append(feature_sequence[:30])
        feature_indexes.append(feature_index[:30])

    return data.X.toarray(), torch.LongTensor(feature_sequences), torch.LongTensor(feature_indexes)

args = create_parser()
args.steps_per_epoch = 0
args.hidden_dim = 128
args.latent_dim = 32
args.layers = 1
args.vq_space = 12
args.levels = [12]
model = MInterface(**vars(args))
model_path = '/storage/guoxiaopeng/wangjue/VQCell/FoldToken2/FoldToken4/results/vqcell_v7_alldata_text/checkpoints/model.pt'
params = torch.load(model_path)
params = {key.replace('_forward_module.',''):val for key, val in params.items()}
model.load_state_dict(params, strict=False)
model.cuda()
print('load pretrained model successfully')

data_path = '/storage/guoxiaopeng/wangjue/data/celldata/scGPT/data/cellxgene/test_data/all_test.h5ad'
data0 = sc.read(data_path)
X, features, feature_indexes = get_data(data0)
data = {'X':torch.from_numpy(data0.X.toarray()).cuda(), 'temp':1e-8, 'property':features.cuda()}

cluster_num = len(data0.obs['cell_type'].unique())

def func(level, name):
    vq_code, vq_emb, Cell_rep, h_V_emb, features, gene_indexes, attention = model.encode(data, 0, level=level)
    # vq_code_uni, uni_index = torch.unique(vq_code, return_inverse=True)
    # vq_emb_uni = scatter_mean(vq_emb, uni_index, dim=0)
  #  vq_code *= 2**4
    print(vq_code.unique().shape)
    print(len(data0.obs['cell_type'].unique()))

    torch.save([attention.mean(1), gene_indexes, vq_code], 'attention.pth')

    Cell_rep, vq_code, vq_emb, features, gene_indexes = map(lambda x:x.detach().cpu().numpy(), [Cell_rep, vq_code, vq_emb, features, gene_indexes])
    sc_data = sc.AnnData(Cell_rep)
    #sc_data = sc.AnnData(data0.X.toarray())
    sc.pp.neighbors(sc_data, use_rep='X', random_state=0)
    sc.tl.umap(sc_data)
    age = []
    for a in data0.obs['development_stage'].tolist():
        if 'year-old' in a:
            age.append(a.replace(' human stage', ''))
        else:
            age.append('others')
    sc_data.obs['age'] = age
    sc_data.obs['sex'] = data0.obs['sex'].tolist()
    sc_data.obs['tissue'] = data0.obs['tissue'].tolist()
    sc_data.obs['disease'] = data0.obs['disease'].tolist()
    sc.pl.umap(sc_data,color=['age'],wspace=0.3,size=30,save=name + '_age', frameon=True)
    sc.pl.umap(sc_data,color=['sex'],wspace=0.3,size=30,save=name + '_sex', frameon=True)
    sc.pl.umap(sc_data,color=['tissue'],wspace=0.3,size=30,save=name + '_tissue', frameon=True)
    sc.pl.umap(sc_data,color=['disease'],wspace=0.3,size=30,save=name + '_disease', frameon=True)

func(12, str(12))