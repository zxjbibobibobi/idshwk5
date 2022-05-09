import pandas as pd
import math
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn import model_selection


def get_entropy(domain):
    tmp = {}
    prelen = len(domain)
    for i in range(0, prelen):
        if domain[i] in tmp.keys():
            tmp[domain[i]] = tmp[domain[i]] + 1
        else:
            tmp[domain[i]] = 1
    shannon = 0
    for i in tmp.keys():
        p = float(tmp[i]) / prelen
        shannon = shannon - p * math.log(p, 2)
    return shannon


def get_length(domain):
    return len(domain.split('.')[0])


def get_ratio(domain):
    yuan_list = ['a', 'e', 'i', 'o', 'u']
    domain = domain.lower()
    count_word = 0
    count_yuan = 0
    yuan_ratio = 0
    for i in range(0, len(domain)):
        if ord('a') <= ord(domain[i]) <= ord('z'):
            count_word = count_word + 1
            if domain[i] in yuan_list:
                count_yuan = count_yuan + 1
    if count_word == 0:
        return yuan_ratio
    else:
        yuan_ratio = float(count_yuan) / count_word
        return yuan_ratio


def DataInit(filename):
    data = []
    f = open(filename)
    for line in f:
        # loading and labeling coarse data
        domain = f.readline()
        domain.strip()
        data.append(domain.split(','))
    if filename == 'train.txt':
        DataForm = pd.DataFrame(columns=['label', 'domain'])
        i = 0
        for d in data:
            print(d[1])
            if 'notdga' in d[1]:
                DataForm.loc[i] = [0, d[0]]
            else:
                DataForm.loc[i] = [1, d[0]]
            i += 1
        # obtaining feature
        DataForm['length'] = DataForm['domain'].map(lambda x: get_length(x)).astype(int)
        DataForm['entropy'] = DataForm['domain'].map(lambda x: get_entropy(x)).astype(float)
        DataForm['ratio'] = DataForm['domain'].map(lambda x: get_ratio(x)).astype(float)
        # feature scaling
        scaler = preprocessing.StandardScaler()
        len_scale_param = scaler.fit(DataForm['length'].values.reshape(-1, 1))
        DataForm['len_scaled'] = scaler.fit_transform(DataForm['length'].values.reshape(-1, 1), len_scale_param)
        shan_scale_param = scaler.fit(DataForm['entropy'].values.reshape(-1, 1))
        DataForm['entropy_scaled'] = scaler.fit_transform(DataForm['entropy'].values.reshape(-1, 1), shan_scale_param)
        # data filtering
        DataPreprocessed = DataForm.filter(regex='label|ratio|len_scaled|entropy_scaled|domain')
        DataPreprocessed = shuffle(DataPreprocessed)
        # matrix generating
        trainingData = DataPreprocessed.iloc[:, :].values
        sample = trainingData[:, 1:]
        sample_label = trainingData[:, 0]
        return sample, sample_label
    else:
        DataForm = pd.DataFrame(columns=['domain'])
        i = 0
        for d in data:
            DataForm.loc[i] = d[0]
            i += 1
        # obtaining feature
        DataForm['length'] = DataForm['domain'].map(lambda x: get_length(x)).astype(int)
        DataForm['entropy'] = DataForm['domain'].map(lambda x: get_entropy(x)).astype(float)
        DataForm['ratio'] = DataForm['domain'].map(lambda x: get_ratio(x)).astype(float)
        # feature scaling
        scaler = preprocessing.StandardScaler()
        len_scale_param = scaler.fit(DataForm['length'].values.reshape(-1, 1))
        DataForm['len_scaled'] = scaler.fit_transform(DataForm['length'].values.reshape(-1, 1), len_scale_param)
        shan_scale_param = scaler.fit(DataForm['entropy'].values.reshape(-1, 1))
        DataForm['entropy_scaled'] = scaler.fit_transform(DataForm['entropy'].values.reshape(-1, 1), shan_scale_param)
        # data filtering
        DataPreprocessed = DataForm.filter(regex='label|ratio|len_scaled|entropy_scaled|domain')
        DataPreprocessed = shuffle(DataPreprocessed)
        # matrix generating
        trainingData = DataPreprocessed.iloc[:, :].values
        sample = trainingData[:, 1:]
        return sample


def main():
    total_sample, sample_label = DataInit('train.txt')
    sample = []
    for t in total_sample:
        sample.append(t[1:])

    model = SVC(kernel='rbf', C=0.4).fit(sample, sample_label.astype('int'))

    test = DataInit('test.txt')
    domain_list = []
    test_data = []
    for t in test:
        test_data.append(t[1:])
        domain_list.append(t[0])
    result = model.predict(test_data)
    with open('result.txt','w') as f:
        for i in range(len(result)):
            f.write(domain_list[i])
            f.write(',')
            if result[i] == 0:
                f.write('notdga')
            else:
                f.write('dga')
            f.write('\n')
        # stat = list(result == test_label)
        # conclusion = '识别准确率为' + str(stat.count(True) / len(stat))
        # f.write(conclusion)
        # f.write('\n')
        # score = model_selection.cross_val_score(model, test_data, test_label.astype('int'),cv=3)
        # s = '交叉检验评分为：' + str(score)
        # f.write(s)
        # stat = list(result == test_label)
        # conclusion = '识别准确率为' + str(stat.count(True) / len(stat))
        # print(conclusion)



main()
