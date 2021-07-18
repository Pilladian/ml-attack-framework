import os

no_glasses = [0, 1, 3, 5, 8, 9, 10, 11, 12, 15, 16, 18, 21, 22, 23, 24, 25, 26, 29, 30, 32, 33, 35, 38, 39]
glasses = [37, 36, 34, 31, 28, 27, 20, 19, 17, 14, 13, 7, 6, 4, 2]

dirs = ['train', 'test', 'eval']

for d in dirs:
    for x in os.listdir(f'./{d}'):
        l = x.split('_')
        if int(l[0]) in glasses:
            name = l[0] + '_1_' + l[1]
        else:
            name = l[0] + '_0_' + l[1]
        os.system(f'mv {d}/{x} {d}/{name}.png')
