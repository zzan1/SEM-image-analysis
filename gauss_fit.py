from scipy.stats import norm
from scipy.stats import rv_histogram
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    width = np.loadtxt(r"Results/PR_5L/Width_PR_5K-0-3333-0-5017.csv", delimiter=',')

    # n, bins, patches = plt.hist(width, bins=50, weights=np.ones(len(width)) / len(width), density=False, range=(0.8, 1.6), facecolor='g',
                                # alpha=0.75)

    n, bins, patches = plt.hist(width, bins=2,
                                range=(0.8, 1.6), facecolor='g',
                                alpha=0.75)
    mu, sigma = norm.fit(width)

    y = norm.pdf(bins, mu, sigma)

    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title('Histogram of IQ')

    plt.text(60, .025, rf'$\mu={mu},\ \sigma={sigma}$')
    # plt.xlim(40, 160)
    # plt.ylim()
    plt.grid(True)
    plt.plot(bins, y, "k--")

    plt.show()
