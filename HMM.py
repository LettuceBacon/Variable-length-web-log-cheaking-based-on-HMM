from urllib.parse import unquote
from csv import reader
import numpy as np
from json import loads
from json import dumps


# fill up a request with itself to max length
def fill_up(a_request, max_len):
    temp = a_request
    while len(a_request) < max_len:
        a_request += temp
    return a_request


# read web logs from 'web_logs.csv'
def read_data(requests, max_len):
    weblog_file = open('web_logs.csv')
    csv_reader = reader(weblog_file, delimiter=',')
    for a_log in csv_reader:
        cmd_url = a_log[2].split(' ')
        a_request = cmd_url[1]
        a_request = unquote(a_request).lower()
        a_request = fill_up(a_request, max_len)
        requests.append(a_request)
    weblog_file.close()


# decide a character's class from four possibilities
def class_of(a_char, char_class):
    if a_char == '/':
        a_char_class = char_class.index("slash")
    elif a_char.isnumeric():
        a_char_class = char_class.index("numerical")
    elif a_char.isalpha():
        a_char_class = char_class.index("alpha")
    else:
        a_char_class = char_class.index("other")
    return a_char_class


# statisic training logs to generate TPM, OLM and IPD
def process_data(max_len, requests, char_set, char_class, TPM, OLM, IPD, n, m):
    pre_char_class = 0
    for a_request in requests:
        a_request = a_request.lower()

        # count inital probability
        IPD[char_set.find(a_request[0])] += 1

        this_char_class = class_of(a_request[0], char_class)
        pre_char_class = this_char_class

        # count observation likelihood and transection probability
        OLM[this_char_class, char_set.find(a_request[0])] += 1
        for i in range(1, max_len):
            this_char_class = class_of(a_request[i], char_class)
            OLM[this_char_class, char_set.find(a_request[i])] += 1
            TPM[pre_char_class, this_char_class] += 1
            pre_char_class = this_char_class

    for i in range(n):
        IPD[i] = IPD[i] / sum(IPD)
    # print(IPD)

    for i in range(n):
        row_sum = TPM.sum(axis=1)[i]
        for j in range(n):
            TPM[i, j] = TPM[i, j] / row_sum
    # print(TPM)

    for i in range(n):
        row_sum = OLM.sum(axis=1)[i]
        for j in range(m):
            OLM[i, j] = OLM[i, j] / row_sum
    # print(OLM)


# write HMM to a file, 'HMM_model.json'
def write_HMM(n, m, IPD, TPM, OLM):
    python_value = {
        'n': n,
        'm': m,
        'IPD': IPD.tolist(),
        'TPM': TPM.tolist(),
        'OLM': OLM.tolist()
    }
    json_data = dumps(python_value)
    # print(json_data)
    HMM_model_file = open('HMM_model.json', 'w')
    HMM_model_file.write(json_data)
    HMM_model_file.close()


# forward algorithm
def hmm_forward(TPM, IPD, OLM, a_request, n, m, char_set, max_len):
    T = max_len
    alpha = np.zeros((T, n))
    P = 0.0

    for i in range(n):
        alpha[0, i] = IPD[i] * OLM[i, char_set.find(a_request[0])]

    for t in range(T - 1):
        for i in range(n):
            temp_value = 0.0
            for j in range(n):
                temp_value += alpha[t, j] * TPM[j, i]
            alpha[t + 1, i] = temp_value *\
                OLM[i, char_set.find(a_request[t + 1])]
    for i in range(n):
        P += alpha[T - 1, i]

    return P, alpha


# backward algorithm
def hmm_backword(TPM, IPD, OLM, a_request, n, m, char_set, max_len):
    T = max_len
    beta = np.zeros((T, n))
    P = 0.0

    for i in range(n):
        beta[T - 1, i] = 1
    t = T - 2
    while t >= 0:
        for i in range(n):
            temp_value = 0.0
            for j in range(n):
                temp_value += TPM[i, j] *\
                    OLM[j, char_set.find(a_request[t + 1])] * beta[t + 1, j]
            beta[t, i] = temp_value
        t -= 1

    for i in range(n):
        P += IPD[i] * OLM[i, char_set.find(a_request[0])] * beta[0, i]

    return P, beta


# main
if __name__ == "__main__":
    # max length of url
    # if a request is less than this length, then fill up it with itself
    max_len = 200

    # the list which stores urls, each item is like
    # "/.../..."
    requests = []

    # hidden states: slash, number, alphabet, other character
    char_class = ['slash', 'numerical', 'alpha', 'other']
    n = len(char_class)

    # observations:
    # /
    # 1234567890
    # abcdefghijklmnopqrstuvwxyz
    # ;,/?:@&+=$#-_.!~*'()`%^{}[]|\"<>
    # space
    char_set = '''\
/1234567890abcdefghijklmnopqrstuvwxyz;,?:@&+=$#-_.!~*'()`%^{}[]|\"<> '''
    m = len(char_set)

    # initial probability distribution
    IPD = np.zeros(n)

    # transection probability matrix
    TPM = np.zeros((n, n))

    # observations likelihood matrix
    OLM = np.zeros((n, m))

    # generate HMM model
    read_data(requests, max_len)
    process_data(max_len, requests, char_set, char_class, TPM, OLM, IPD, n, m)
    write_HMM(n, m, IPD, TPM, OLM)

    # forward and backward algorithm
    samples_file = open('samples.json', 'r')
    json_value = loads(samples_file.read())
    normal_rq = unquote(json_value['request']['normal_rq']).lower()
    normal_rq = fill_up(normal_rq, max_len)
    abnormal_rq = unquote(json_value['request']['abnormal_rq']).lower()
    abnormal_rq = fill_up(abnormal_rq, max_len)
    samples_file.close()

    P, alpha_matrix =\
        hmm_forward(TPM, IPD, OLM, normal_rq, n, m, char_set, max_len)
    print('forward normal: ' + str(P))
    # print(alpha_matrix)
    print('.........................................................')
    P, alpha_matrix =\
        hmm_forward(TPM, IPD, OLM, abnormal_rq, n, m, char_set, max_len)
    print('forward abnormal: ' + str(P))
    # print(alpha_matrix)
    print('.........................................................')
    P, beta_matrix =\
        hmm_backword(TPM, IPD, OLM, normal_rq, n, m, char_set, max_len)
    print('backward normal: ' + str(P))
    # print(beta_matrix)
    print('.........................................................')
    P, beta_matrix =\
        hmm_backword(TPM, IPD, OLM, abnormal_rq, n, m, char_set, max_len)
    print('backward abnormal: ' + str(P))
    # print(beta_matrix)
