import numpy as np

randIdx_test_normal = np.arange(250)
randIdx_test_anomaly = np.arange(750)

np.random.shuffle(randIdx_test_normal)
np.random.shuffle(randIdx_test_anomaly)
np.save("OCT_randIdx_test_normal1.npy",randIdx_test_normal)
np.save("OCT_randIdx_test_anomaly1.npy",randIdx_test_anomaly)

np.random.shuffle(randIdx_test_normal)
np.random.shuffle(randIdx_test_anomaly)
np.save("OCT_randIdx_test_normal2.npy",randIdx_test_normal)
np.save("OCT_randIdx_test_anomaly2.npy",randIdx_test_anomaly)

np.random.shuffle(randIdx_test_normal)
np.random.shuffle(randIdx_test_anomaly)
np.save("OCT_randIdx_test_normal3.npy",randIdx_test_normal)
np.save("OCT_randIdx_test_anomaly3.npy",randIdx_test_anomaly)

np.random.shuffle(randIdx_test_normal)
np.random.shuffle(randIdx_test_anomaly)
np.save("OCT_randIdx_test_normal4.npy",randIdx_test_normal)
np.save("OCT_randIdx_test_anomaly4.npy",randIdx_test_anomaly)

np.random.shuffle(randIdx_test_normal)
np.random.shuffle(randIdx_test_anomaly)
np.save("OCT_randIdx_test_normal5.npy",randIdx_test_normal)
np.save("OCT_randIdx_test_anomaly5.npy",randIdx_test_anomaly)