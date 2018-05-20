import wordtovec
import numpy as np
a = np.array([0,0,1,1,2,2])
b = np.array([1,2,0,2,1,0])
lexiconSize = 3
numberHiddenUnits = 5
wordtovec.trainNetwork(a,b,lexiconSize,numberHiddenUnits)
