import numpy as np

randIdx_anomaly = np.arange(57169)
np.random.shuffle(randIdx_anomaly)
np.save("OCT_randIdx_anomaly1.npy",randIdx_anomaly)
np.random.shuffle(randIdx_anomaly)
np.save("OCT_randIdx_anomaly2.npy",randIdx_anomaly)
np.random.shuffle(randIdx_anomaly)
np.save("OCT_randIdx_anomaly3.npy",randIdx_anomaly)
np.random.shuffle(randIdx_anomaly)
np.save("OCT_randIdx_anomaly4.npy",randIdx_anomaly)
np.random.shuffle(randIdx_anomaly)
np.save("OCT_randIdx_anomaly5.npy",randIdx_anomaly)
