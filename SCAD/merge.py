import os 
import pandas as pd


path = 'data/split_norm/'
names = ['Sorafenib', 'PLX4720_451Lu', 'NVP-TAE684', 'Etoposide']

for name in names:
    l_path = path + name + '/stratified/'
    for i in range(1, 6):
        r_path = l_path + 'source_5_folds/split' + str(i) + '/X_train_source.tsv'
        p_path = r_path.replace('stratified', 'stratified_emb_vqcell_ft')
        r = pd.read_csv(r_path, sep='\t', index_col=0, decimal='.')
        p = pd.read_csv(p_path, sep='\t', index_col=0, decimal='.')
        res = pd.concat([r, p], axis=1)
        res.to_csv(r_path.replace('stratified', 'stratified_emb_merge'), sep='\t', index=True)

        r_path = l_path + 'source_5_folds/split' + str(i) + '/X_val_source.tsv'
        p_path = r_path.replace('stratified', 'stratified_emb_vqcell_ft')
        r = pd.read_csv(r_path, sep='\t', index_col=0, decimal='.')
        p = pd.read_csv(p_path, sep='\t', index_col=0, decimal='.')
        res = pd.concat([r, p], axis=1)
        res.to_csv(r_path.replace('stratified', 'stratified_emb_merge'), sep='\t', index=True)

        r_path = l_path + 'target_5_folds/split' + str(i) + '/X_train_target.tsv'
        p_path = r_path.replace('stratified', 'stratified_emb_vqcell_ft')
        r = pd.read_csv(r_path, sep='\t', index_col=0, decimal='.')
        p = pd.read_csv(p_path, sep='\t', index_col=0, decimal='.')
        res = pd.concat([r, p], axis=1)
        res.to_csv(r_path.replace('stratified', 'stratified_emb_merge'), sep='\t', index=True)

        r_path = l_path + 'target_5_folds/split' + str(i) + '/X_test_target.tsv'
        p_path = r_path.replace('stratified', 'stratified_emb_vqcell_ft')
        r = pd.read_csv(r_path, sep='\t', index_col=0, decimal='.')
        p = pd.read_csv(p_path, sep='\t', index_col=0, decimal='.')
        res = pd.concat([r, p], axis=1)
        res.to_csv(r_path.replace('stratified', 'stratified_emb_merge'), sep='\t', index=True)


