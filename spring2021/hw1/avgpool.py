import os
import numpy as np
files = os.listdir('./soundnet/')
layer = '/tf_fea18.npy'
for filename in files:
    if os.path.exists('./soundnet/'+filename+layer):
        raw_feat=np.load('./soundnet/'+filename+layer)
        raw_feat1 = np.max(raw_feat,0).tolist()
        np.savetxt('./avgpool18/'+filename+'.csv', raw_feat1)
