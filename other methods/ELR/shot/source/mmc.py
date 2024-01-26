import numpy as np



def generate_mmc_center(var, dim_dense, num_class):
    mmc_centers = np.zeros((num_class, dim_dense))
    mmc_centers[0][0] = 1
    for i in range(1, num_class):
        for j in range(i):
            mmc_centers[i][j] = - (1 / (num_class - 1) + np.dot(mmc_centers[i], mmc_centers[j])) / mmc_centers[j][j]
        mmc_centers[i][i] = np.sqrt(np.absolute(1 - np.linalg.norm(mmc_centers[i])) ** 2)
    for k in range(num_class):
        mmc_centers[k] = var * mmc_centers[k]

    return mmc_centers


mmc_centers = generate_mmc_center(10, 256, 65)

print(mmc_centers.shape)
print(mmc_centers)