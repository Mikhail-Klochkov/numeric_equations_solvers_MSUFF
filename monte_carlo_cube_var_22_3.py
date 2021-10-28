
import numpy as np
N = 57
x_point = np.ones(N) * 0.5
Number_generated_examples = int(1e5)

def f_vertix(e):
    assert e.shape[0] == 57
    return np.sqrt((e ** 2).sum())

# many time for 10 ^ 5 calculation (need) slow implementation
if __name__ == '__main__':
    probs = {dim: [1-coor, coor] for dim, coor in enumerate(x_point)}
    summ_accum = 0
    for i in range(Number_generated_examples):
        if i % 1000:
            print(f'iter: {i}')
            print(f'ans: {ans}')
        # sampling strategy
        point = np.empty(N)
        for dim in range(N):
            p_current = probs[dim]
            point[dim] = np.random.choice([0, 1], 1, p = p_current)
        summ_accum += f_vertix(point)
        ans = summ_accum / (i+1)
    print('answer: {ans}')