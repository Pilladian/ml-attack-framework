import os
import random

for a in os.listdir('./'):
    if a in ['eval', 'test', 'train']:
        continue

    rand = random.randint(0, 101)
    if rand < 70:
        os.system(f'mv {a} train/')
    elif rand < 80:
        os.system(f'mv {a} eval/')
    else:
        os.system(f'mv {a} test/')