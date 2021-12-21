# -*- encoding: utf-8 -*-
"""
analyse the training result
"""
import re
import matplotlib.pyplot as plt


def load(path='../log/property.log', element='Validation'):
    tmp_path = '../log/{}.log'.format(element)
    with open(path, 'r', encoding='utf-8') as f:
        lines = []
        for line in f.readlines():
            if element in line:
                lines.append(line)

        with open(tmp_path, 'w', encoding='utf-8') as tmp:
            tmp.writelines(lines)


def draw(path='../log/tmp.log'):
    pattern = re.compile(r'[0-9].+')
    losses = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            p = pattern.search(line).group()
            if float(p) < 1:
                losses.append(float(p))

    print("MAE loss: {}".format(min(losses)))
    plt.plot(range(len(losses)), losses)
    plt.show()


def main():
    # load(path='../log/property_1.log', element='Validation')
    draw(path='../log/Validation.log')


if __name__ == '__main__':
    main()
