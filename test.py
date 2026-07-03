import numpy as np
data = np.load("results/survey_15param.npz")
print(data["params"].shape)
print(data["params"][:3])
