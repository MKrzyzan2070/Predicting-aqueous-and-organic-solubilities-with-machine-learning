import numpy as np
import scipy.stats as stats
import scipy.special as special


def kendall_top_k(a, b, k=None,
                  p=0):
    if k is None:
        k = a.size
    if a.size != b.size:
        raise NameError('The two arrays need to have same lengths')
    k = min(k, a.size)
    a_top_k = np.argpartition(a, -k)[-k:]
    b_top_k = np.argpartition(b, -k)[-k:]
    common_items = np.intersect1d(a_top_k, b_top_k)
    only_in_a = np.setdiff1d(a_top_k, common_items)
    only_in_b = np.setdiff1d(b_top_k, common_items)
    kendall = (1 - (stats.kendalltau(a[common_items], b[common_items])[0] / 2 + 0.5)) * (
                common_items.size ** 2)  # case 1
    if np.isnan(kendall):  # degenerate case with only one item (not defined by Kendall)
        kendall = 0
    for i in common_items:  # case 2
        for j in only_in_a:
            if a[i] < a[j]:
                kendall += 1
        for j in only_in_b:
            if b[i] < b[j]:
                kendall += 1
    kendall += 2 * p * special.binom(k - common_items.size, 2)  # case 4
    kendall /= ((only_in_a.size + only_in_b.size + common_items.size) ** 2)  # normalization
    return kendall
