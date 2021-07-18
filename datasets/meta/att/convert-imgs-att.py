import os

os.system('mkdir images')

for i in range(1, 41):
    for a in range(1, 11):
        os.system(f'convert s{i}/{a}.pgm {i}_{a}.png && mv {i}_{a}.png images/')
