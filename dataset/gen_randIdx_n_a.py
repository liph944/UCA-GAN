import numpy as np

randIdx_normal=np.arange(1000)
randIdx_abnormal=np.arange(9000)
np.random.shuffle(randIdx_normal)
np.random.shuffle(randIdx_abnormal)
np.save("randIdx_normal1.npy",randIdx_normal)
np.save("randIdx_abnormal1.npy",randIdx_abnormal)