import os
import random

for a in range(1, 41):
    for b in range(1, 11):
        name = f'{a}_{b}.png'
        x = random.randint(1, 100)
        if x < 70:
            os.system(f'mv images/{name} ./train/')
        elif x < 80:
            os.system(f'mv images/{name} ./eval/')
        else:
            os.system(f'mv images/{name} ./test/')
