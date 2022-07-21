import argparse
import numpy as np
import matplotlib.pyplot as plt

EVO_AttnGAN = 'C:/Users/jsan/PycharmProjects/Evolutionary_AttnGAN/models/bird_evo_attngan2_mutation_count.npy'
IE_AttnGAN = 'C:/Users/jsan/PycharmProjects/Evolutionary_AttnGAN/models/bird_ie_attngan2_mutation_count.npy'




def parse_args():
    parser = argparse.ArgumentParser(description='Run plotter for offspring types')
    parser.add_argument('--path', type=str, default=EVO_AttnGAN)
    parser.add_argument('--title', type=str, default='Evo-AttnGAN')
    args = parser.parse_args()
    return args


def selected_offspring(path, title):
    mutations = np.load(path, allow_pickle=True)
    mutations = np.ndarray.tolist(mutations)
    end = 800
    frequency = 10

    font = {'family': 'normal',
            'size': 14}

    plt.rc('font', **font)

    x = list(range(int(end)))[::frequency]
    print(len(x))
    y1 = mutations['minimax'][0:end][::frequency]
    print(len(y1))
    y2 = mutations['least_squares'][0:end][::frequency]
    y3 = mutations['heuristic'][0:end][::frequency]

    plt.plot(x, y1, label='Minimax')
    plt.plot(x, y2, label='Least Squares')
    plt.plot(x, y3, label='Heuristic')

    if title.startswith('IE-AttnGAN'):
        y4 = mutations['crossover'][0:end][::frequency]
        plt.plot(x, y4, label='Crossover')


    plt.xlabel('Epoch')
    plt.ylabel('Num Selected')
    plt.title('{} - Selected Offspring per Epoch'.format(title))
    plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    selected_offspring(args.path, args.title)

