import functools
import math
from Slice_method import Distri_M, sample_input_output, hw
import numpy as np
from operator import mul
from multiprocessing import Pool


# P((hm,hy)|(h∗m,h∗y))
def prob_observation_given_estimation(ob, est, sigma_m, sigma_y):
    """
    Compute the probability of the noise that accounts for the observation
    :param ob: (hm, hy)
    :param est: (h*m, h*y)
    :param sigma_m: sigma_m
    :param sigma_y: sigma_y
    :return:P((hm,hy)|(h∗m,h∗y))=Pr(ωm=hm−h∗m)·Pr(ωy=hy−h∗y)
    """
    hm, hy = ob
    hm_e, hy_e = est
    exp_m, exp_y = ((hm - hm_e) / 2) ** 2, ((hy - hy_e) / 2) ** 2
    prob_noise_m = math.exp(-exp_m / 2) / (sigma_m * math.sqrt(2 * math.pi))  # Pr(ωm=hm−h∗m)
    prob_noise_y = math.exp(-exp_y / 2) / (sigma_y * math.sqrt(2 * math.pi))  # Pr(ωy=hy−h∗y)
    return prob_noise_m * prob_noise_y


# P((hm,hy)|k)
def prob_one_ob_given_key(ob, k, sigma1, sigma2):
    """
    Compute the probability of the observed Hamming weights given the key
    :param ob: (hm, hy)
    :param k: key
    :param sigma1: sigma_m
    :param sigma2: sigma_y
    :return: Pr((hm,hy)|k) =∑_(h∗m,h∗y) Pr((hm,hy)|(h∗m,h∗y))·Pr((h∗m,h∗y)|k)
    """
    real_weights = [(i, j) for i in range(9) for j in range(9)]
    prob = 0
    for weights in real_weights:
        prob_obs_est = prob_observation_given_estimation(ob, weights, sigma1, sigma2)  # Pr((hm,hy)|(h∗m,h∗y))
        prob_weight_key = Distri_M(k)[weights[0]][weights[1]]  # Pr((h∗m,h∗y)|k)
        prob += prob_obs_est * prob_weight_key
    return prob


# P(k|(hm,hy))
def prob_key_given_obs(obs, k, sigma1, sigma2):
    """
    Compute  the  probability  distribution of  the  key  given  the  observed  Hamming  weights
    :param obs: [(hm, hy)_1, ..., (hm, hy)_n]
    :param k: key
    :param sigma1: sigma_m
    :param sigma2: sigma_y
    :return: P(k|(hm,hy)) = Pr((hm,hy)_1|k)...Pr((hm,hy)_n|k)
    """
    print("guess key", k)
    prob_k_given_obs = 1
    for ob in obs:
        prob_k_given_obs *= prob_one_ob_given_key(ob, k, sigma1, sigma2)
    # probs_k_given_ob = [prob_one_ob_given_key(ob, k, sigma1, sigma2) for ob in obs]
    # return functools.reduce(mul, probs_k_given_ob, 1)
    return prob_k_given_obs * 100


def guess_key2(keys, obs, sigma1, sigma2):
    """
    Compute P(k|(hm,hy)) for all key in range 0, 255 according to obs and choose the one that has max probability
    :param obs: [(hm, hy)_1, ..., (hm, hy)_n]
    :param sigma1: sigma_m
    :param sigma2: sigma_y
    :return: max likely-hood key
    """
    key_probs = [prob_key_given_obs(obs, key, sigma1, sigma2) for key in keys]

    # for key in range(0, 256):
    #     print("guess k", key)
    #     key_prob = prob_key_given_obs(obs, key, sigma1, sigma2)
    #     if key_prob > max_prob:
    #         max_key = key
    #         max_prob = key_prob
    max_p = np.max(key_probs)
    return max_p, keys[0] + np.argmax(key_probs)


pool = Pool(processes=4)


def paralle_guess(obs, sigma1, sigma2):
    key1 = [key for key in range(0, 64)]
    key2 = [key for key in range(64, 128)]
    key3 = [key for key in range(128, 192)]
    key4 = [key for key in range(192, 256)]
    inputs = [(key1, obs, sigma1, sigma2), (key2, obs, sigma1, sigma2), (key3, obs, sigma1, sigma2),
              (key4, obs, sigma1, sigma2)]
    procs = [pool.apply_async(guess_key2, args=i) for i in
             inputs]
    re = [proc.get() for proc in procs]
    print(re)


def simulate(n, k, sigma1, sigma2):
    sample = sample_input_output(n, k)
    sample_weights = [(hw(s[0]), hw(s[1])) for s in sample]
    observations = [(w[0] + np.random.normal(0, sigma1), w[1] + np.random.normal(0, sigma2)) for w in sample_weights]
    print(paralle_guess(observations, sigma1, sigma2))


simulate(500, 39, 1, 1)

# if __name__ == '__main__':

# lis = [2, 3, 4]
# print(functools.reduce(mul, lis))
# pool = Pool()
# inputs = [[[0, 1], 2], [(2, 3), 0]]
# outputs = [pool.apply_async(ha, args=i) for i in
#            inputs]
# o = [i.get() for i in outputs]
# print(o)
