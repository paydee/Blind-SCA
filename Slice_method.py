import random,math, numpy

#Sbox
SBOX = [
   0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
   0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
   0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
   0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
   0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
   0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
   0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
   0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
   0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
   0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
   0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
   0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
   0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
   0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
   0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
   0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
]

table_hamming = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8]

#Calculate Hamming weight of a byte
def hw(m):
    h = table_hamming[m]
    return h

#Normalize an empirical distribution
def normal(M,n):
    for i in range(9):
        for j in range(9):
            M[i][j] = M[i][j]/n
    return M

#Compare empirical distribution with theoretical one
def stat_dist(M,D):
    S = 0
    for i in range(9):
        for j in range(9):
            S += (M[i][j] - D[i][j])**2
    return S

#Generate a theoretical distribution given a key k
def Distri_M(k):
    M = [ [ 0 for i in range(9) ] for j in range(9) ]
    for m in range (0,256):
        x = m ^ k
        y = SBOX[x]
        M[hw(m)][hw(y)] += 1
    M = normal(M,256)
    return M

#Generate empirical distribution D
def Distri_D(estimated_weights_m,estimated_weights_y):
    D = [ [ 0 for i in range(9) ] for j in range(9) ]
    n = len(estimated_weights_m)
    for i in range(0,n):
        D[estimated_weights_m[i]][estimated_weights_y[i]] += 1
    return normal(D,n)

#Finding key by stats distance method
def find_key(D):
    key = 0
    for i in range(1,256):
        if stat_dist(Distri_M(key),D) > stat_dist(Distri_M(i),D):
            key = i
    return key

#Generate series of random input m and corresponding output y
#Number of samples: n
def sample_input_output(n,key):
    Result = []
    count = 0
    while 1:
        if count == n:
            break
        m = random.randint(0,255)
        x = m ^ key
        y = SBOX[x]
        temp = (m,y)
        Result.append(temp)
        count += 1
    return Result

#Generate noised leakages for series of m and y
def noised_leakage(weight_series,a,b,sigma):
    consom = [0]*len(weight_series)
    std_dev = sigma*a
    for i in range(len(weight_series)):
        omega = random.gauss(0,std_dev) 
        consom[i] = a*weight_series[i] + b + omega
    return consom

#Apply Method of slices for series of consom(m) and consom(y)
def slice_categorize(consom_series):
    n = len(consom_series)
    order_tagged = []
    for i in range(n):
        tag = [i,consom_series[i]]
        order_tagged.append(tag)
    sorted_consom = sorted(order_tagged,key = lambda t:t[1])
    weight_distribution = []
    for i in range(0,9):
        C_n_k = math.comb(8,i)
        weight_ratio = C_n_k/256
        weight_slice_cardinal = round(n*weight_ratio)
        weight_distribution.append(weight_slice_cardinal)
    sorted_weight = []
    for i in range(0,9):
        for j in range(0,weight_distribution[i]):
            temp1 = sorted_consom.pop(0)
            temp2 = [temp1[0],i]
            sorted_weight.append(temp2)
    re_ordered_weight = sorted(sorted_weight,key = lambda t:t[0])
    estimated_weights = [re_ordered_weight[i][1] for i in range(0,n)]
    return estimated_weights

#Generate a an attack simulation (parameters am, bm, ay, by are chosen randomly)
def attack_simulate(key,number_samples,sigma_m,sigma_y):
    a_m = random.uniform(1,5)
    b_m = random.uniform(50,100)
    a_y = random.uniform(1,5)
    b_y = random.uniform(50,100)

    series_input_output = sample_input_output(number_samples,key)
    weight_series_m = [hw(series_input_output[i][0]) for i in range(number_samples)]
    weight_series_y = [hw(series_input_output[i][1]) for i in range(number_samples)]
    noised_leakage_m = noised_leakage(weight_series_m,a_m,b_m,sigma_m)
    noised_leakage_y = noised_leakage(weight_series_y,a_y,b_y,sigma_y)
    estimated_weights_m = slice_categorize(noised_leakage_m)
    estimated_weights_y = slice_categorize(noised_leakage_y)
    D = Distri_D(estimated_weights_m,estimated_weights_y)
    estimated_key = find_key(D)
    if estimated_key == key:
        return 1
    else:
        return 0

#Test case for an attack
#sigma_m = 1
#sigma_y = 1
#val = attack_simulate(123, 256,sigma_m,sigma_y)
#if val == 1:
#    print("Success!")
#else:
#    print("Failure!")

#Measure success rate (parameters a,b,sigma of x and y are fixed through tries)
def success_rate(number_tries, number_samples,sigma_m,sigma_y):
    count = 0
    for i in range(number_tries):
        key = random.randint(0,255)
        val = attack_simulate(key,number_samples,sigma_m,sigma_y)
        if val == 1:
            count += 1        
    rate = count/number_tries
    return rate


#Test case for success rates

sigma_m = random.uniform(1, 5)
sigma_y = sigma_m

number_tries = 100
number_samples = 1024

print("Success rate for [sigma_m = sigma_y = %f]:"%(sigma_m),success_rate(number_tries,number_samples,sigma_m,sigma_y))



