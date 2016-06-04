import numpy as np
from sklearn.metrics import accuracy_score
from time import time as time
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import arff # https://pypi.python.org/pypi/liac-arff
from label_propagaton import LabelPropagation as CustomLabelPropagation

def do_stuff(training_data, training_label, test_data, test_label, algo, kernel,  neighbors , kernel_gamma, eps, minPts, alpha):

    if not algo == 'custom':
        Classifier = LabelPropagation if algo == 'propagation' else LabelSpreading
        if kernel == 'rbf':
            clf = Classifier(kernel=str('rbf'), alpha=alpha, gamma=kernel_gamma)
        elif kernel == 'knn':
            clf = Classifier(kernel=str('knn'), alpha=alpha, n_neighbors=neighbors)

    data_for_testing = np.concatenate((training_data, test_data))
    label_for_testing = np.concatenate((training_label, np.ones_like(test_label)*-1))

    true_label_for_testing = np.concatenate((training_label, test_label))

    t = time()  # get labels for test data
    if algo == 'custom': 
        test_prediction = CustomLabelPropagation(kernel=kernel, unlabeledValue=-1, neighbors=neighbors, gamma=kernel_gamma, eps=eps, minPts= minPts).fit(data_for_testing, label_for_testing)
    else:
        clf.fit(data_for_testing, label_for_testing)
        test_prediction = clf.predict(data_for_testing)
    time_test = time() - t

    # for both test and validation data get the accuracy scores
    score_test = accuracy_score(true_label_for_testing, test_prediction)

    result =  {'score_test':score_test, # Kriegen Studis zu sehen
               #'training_labels':training_label,
               #'test_labels':test_prediction,
#                'gamma':kernel_gamma,
#                'neighbors':neighbors,
#                'alpha':alpha,
               'extra_scores_test':{"time":"%2.3fms"%(time_test*1000)}}
    return result

def loadData(_name):
    data = arff.load(open("data/" + _name + '.arff'))
    Data = []
    Target = []
    for d in data['data']:
        Data.append(d[:-1])
        Target.append(int(d[-1]))
    return [ np.array(Data),  np.array(Target) ]
def loadDataSet (_name):
    testData , testTarget   = loadData(_name + '_test')
    trainData, trainTarget = loadData(_name + '_train')
    return [testData , testTarget, trainData, trainTarget]

if __name__ == "__main__":
    X = np.array([[1, 7], [8, 8],[1,5],[8,7],[9,5]])
    y = np.array(['a','b','null','null','null'])
    print( CustomLabelPropagation(unlabeledValue='null', eps=3, minPts=3).fit(X, y))
#     X = np.array([[1, 7], [8, 8],[1,5],[8,7]])
#     y = np.array([1,2,0,0])
#          
#     print( CustomLabelPropagation( eps=3, minPts=3).fit(X, y))
#     X = np.array([[1, 7], [8, 8],[1,5],[8,6],[9,9],[8,9],[1,4],[3,1],[4,1],[1,2],[6,1]])
#     y = np.array([0,1,12,-1,-1,-1,-1,-1,-1,-1,-1])
#     print( CustomLabelPropagation( eps=2, minPts=4, unlabeledValue = 0).fit(X, y))
    
    # test datas 
    datas = [
            ['ForestFires',3,100,40,5], 
            ['spirals_tl',3,3.9,2,3],
            ['yeast_tl',3,3.9,3,4],
            ['graph_tl',30,5.9,2,2],
            ['bc_tl',3,3.9,1,1] 
            ]
    for name, neighbors, gamma, eps, minPts in datas :
        print("-----------")
        print('\t' + name)
        print("-----------")
           
        testData , testTarget , trainData , trainTarget = loadDataSet(name)
        print(do_stuff(testData, testTarget, trainData, trainTarget, 'propagation', 'rbf', neighbors, gamma, eps, minPts, 0.2))
        print(do_stuff(testData, testTarget, trainData, trainTarget, 'custom', 'knn', neighbors, gamma, eps, minPts, 0.8))
        print(do_stuff(testData, testTarget, trainData, trainTarget, 'custom', 'rbf', neighbors, gamma, eps, minPts, 0.8))
        print(do_stuff(testData, testTarget, trainData, trainTarget, 'custom', None, neighbors, gamma, eps, minPts, 0.8))

    
    
