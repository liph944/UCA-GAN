import numpy as np

randIdx = np.arange(54000)
np.random.shuffle(randIdx)
np.save("fmnist_randIdx1.npy",randIdx)