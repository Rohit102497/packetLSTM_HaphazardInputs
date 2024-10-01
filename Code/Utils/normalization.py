import torch

def normalize_z_score(X_hap, mask):
    X=torch.tensor(X_hap, dtype=torch.float)
    mask=torch.tensor(mask,dtype=torch.bool)
    n_features = X.shape[1]
    m = torch.zeros(n_features)
    feat_count = torch.zeros(n_features,dtype=torch.int)
    v = torch.zeros(n_features)
    feat_observed = torch.zeros(n_features,dtype=torch.bool)


    for t in range(X.shape[0]):
        feat_indices_curr = torch.arange(n_features)[mask[t]==1]
        feat_indices_new = torch.arange(n_features)[mask[t]&(~feat_observed)]
        feat_indices_old = torch.arange(n_features)[mask[t]&feat_observed]
        feat_observed = feat_observed | mask[t]
        feat_count[feat_indices_curr]+=1
        if t==0:
            m[feat_indices_curr] = X[t][feat_indices_curr]
        else:
            m[feat_indices_new] = X[t][feat_indices_new].float()
            count = feat_count[feat_indices_old]
            m_t = m[feat_indices_old]+(X[t][feat_indices_old]-m[feat_indices_old])/count
            v[feat_indices_old] = v[feat_indices_old]+(X[t][feat_indices_old]-m[feat_indices_old])*(X[t][feat_indices_old]-m_t)
            m[feat_indices_old] = m_t
            if len(feat_indices_old)>0:
                if torch.min(v[feat_indices_old])>0.0:
                    X[t][feat_indices_old] = (((X[t][feat_indices_old]-m[feat_indices_old])).float()/torch.sqrt(v[feat_indices_old]/(count-1)))
    return X.numpy()