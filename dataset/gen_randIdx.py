import numpy as np

randIdx = np.arange(45000)
np.random.shuffle(randIdx)
np.save("randIdx1.npy",randIdx)