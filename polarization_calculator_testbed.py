import math
import numpy as np
from matplotlib import pyplot
from random import uniform


def cosine_sim(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norma = np.linalg.norm(vec1)
    normb = np.linalg.norm(vec2)
    cos = dot / (norma * normb)
    return cos


# 0
'''
anchor_a = (1, 1)
anchor_b = (100, 100)

context_a = np.array([(1, 1), (50, 50), (100, 100)])
context_b = np.array([(100, 100), (50, 50), (1, 1)])
'''

# 0

anchor_a = (1, 10)
anchor_b = (50, 50)

context_a = np.array([(uniform(0, 1), uniform(0, 1)) for i in range(50)])
context_b = np.array([(uniform(0, 1), uniform(0, 1)) for i in range(50)])
cxt = np.concatenate((context_a, context_b))
cor = ['#17becf'] * len(context_a) + ['#ff5d47'] * len(context_b)
pyplot.scatter(cxt[:, 0], cxt[:, 1], s=20, c=cor)
pyplot.axis([0, 1, 0, 1])
pyplot.show()

# .5
'''
anchor_a = (1, 1)
anchor_b = (100, 100)

context_a = np.array([(1, 1), (20, 20), (40, 40)])
context_b = np.array([(60, 60), (80, 80), (100, 100)])
'''

'''
anchor_a = (1, 1)
anchor_b = (100, 100)

context_a = np.array([(1, 1), (1, 1), (1, 1)])
context_b = np.array([(100, 100), (100, 100), (100, 100)])
'''

all_vec = np.concatenate((context_a, context_b))
a_term = len(context_a) * (np.linalg.norm(np.mean(context_a, axis=0) - np.mean(all_vec, axis=0)) ** 2)
b_term = len(context_b) * (np.linalg.norm(np.mean(context_b, axis=0) - np.mean(all_vec, axis=0)) ** 2)
denom_terms = []
for vec in all_vec:
    denom_terms.append(np.linalg.norm(vec - np.mean(all_vec, axis=0)) ** 2)
    # return value
print("R^2: " + str((a_term + b_term) / sum(denom_terms)))
print("R^2 adj: " + str((abs(a_term - b_term)) / sum(denom_terms)))

tight_arr_a = []
for term in context_a:
    tight_arr_a.append(cosine_sim(anchor_a, term))
tight_a = (1 / len(tight_arr_a)) * sum(tight_arr_a)
tight_arr_b = []
for term in context_b:
    tight_arr_b.append(cosine_sim(anchor_b, term))
tight_b = 1 / len(tight_arr_b) * sum(tight_arr_b)
tight = 1 / 2 * (tight_a + tight_b)

print("tight: " + str(tight))

disp_arr_a = []
for term in context_a:
    disp_arr_a.append(1 - cosine_sim(anchor_b, term))
disp_a = (1 / len(disp_arr_a)) * sum(disp_arr_a)
disp_arr_b = []
for term in context_b:
    disp_arr_b.append(1 - cosine_sim(anchor_a, term))
disp_b = 1 / len(disp_arr_b) * sum(disp_arr_b)
disp = 1 / 2 * (disp_a + disp_b)

print("disagreement: " + str(disp))

print("controversy rating: " + str((math.sqrt(tight) + math.sqrt(disp)) / 2))
