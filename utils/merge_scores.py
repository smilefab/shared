import numpy as np
import os
os.chdir('C:/Users/Fabian/Documents/Master/LR - IP/Project/shared/logs')

[folder] = [name for name in os.listdir('.') if os.path.isdir(name)]
scores = [score for score in os.listdir(os.path.join('.', folder)) if os.path.isfile(score)]

score = np.concatenate(scores, axis=0)

np.save('merged_score.npy', score)