import os
import random

for a in os.listdir('./'):
    if a in ['eval', 'train', 'test', 'split.py']:
        continue

    r = random.randint(0, 101)
    if r < 70:
        os.system(f'mv {a} train/')
    elif r < 80:
        os.system(f'mv {a} eval/')
    else:
        os.system(f'mv {a} test/')