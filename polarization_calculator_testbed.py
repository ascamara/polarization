import math
import numpy as np


def cosine_sim(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norma = np.linalg.norm(vec1)
    normb = np.linalg.norm(vec2)
    cos = dot / (norma * normb)
    return cos


anchor_a = (14, 14)
anchor_b = (14, 15)

context_a = np.array([(15, 15), (14, 14), (15, 17)])
context_b = np.array([(16, 13), (15, 13), (15, 14)])

all_vec = np.concatenate((context_a, context_b))
a_term = len(context_a) * (np.linalg.norm(np.mean(context_a, axis=0) - np.mean(all_vec, axis=0)) ** 2)
b_term = len(context_b) * (np.linalg.norm(np.mean(context_b, axis=0) - np.mean(all_vec, axis=0)) ** 2)
denom_terms = []
for vec in all_vec:
    denom_terms.append(np.linalg.norm(vec - np.mean(all_vec, axis=0)) ** 2)
    # return value
print(a_term + b_term) / sum(denom_terms)

tight_arr_a = []
for term in context_a:
    tight_arr_a.append(cosine_sim(anchor_a, term))
tight_a = (1 / len(tight_arr_a)) * sum(tight_arr_a)
tight_arr_b = []
for term in context_b:
    tight_arr_b.append(cosine_sim(anchor_b, term))
tight_b = 1 / len(tight_arr_b) * sum(tight_arr_b)
tight = 1 / 2 * (tight_a + tight_b)

print(tight)

disp_arr_a = []
for term in context_a:
    disp_arr_a.append(1 - cosine_sim(anchor_b, term))
disp_a = (1 / len(disp_arr_a)) * sum(disp_arr_a)
disp_arr_b = []
for term in context_b:
    disp_arr_b.append(1 - cosine_sim(anchor_a, term))
disp_b = 1 / len(disp_arr_b) * sum(disp_arr_b)
disp = 1 / 2 * (disp_a + disp_b)

print(disp)

print((math.sqrt(tight) + math.sqrt(disp)) / 2)
