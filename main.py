# Michael Rizig
# Combination Algorithm that utilizes a voting system to make prediction
# 5/1/25
# Professor Alexiou

from BayesClassifier import BayesClassifier
from SupervisedDiagnosis import Supervised
from UnSupervisedDiagnosis import Unsupervised
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

if __name__ == "__main__":
    # init all 3 models
    bayes = BayesClassifier()
    supervised = Supervised()
    unsupervised = Unsupervised()
    print("--------------BAYES--------------")
    print(bayes.evaluate_model())
    print("--------------SUPERVISED--------------")
    print(supervised.evaluate_model())
    print("--------------UNSUPERVISED--------------")
    print(unsupervised.evaluate())
    bayes_output = bayes.predict(bayes.xtest)
    supervised_output = supervised.predict(bayes.xtest)
    unsupervised_output = unsupervised.predict(bayes.xtest)
    voted_output = []
    # variables for confusion matrix
    tp =0
    fp= 0
    fn=0
    tn=0
    for i in range(len(bayes_output)):
        true_count = 0
        if bayes_output[i] == 1:
            true_count += 1
        if supervised_output[i] == 1:
            true_count += 1
        if unsupervised_output[i] == 1:
            true_count += 1

        if true_count >= 2:
            voted_output.append(1)
        else:
            voted_output.append(0)
        if voted_output[i] == 1:
            if bayes.ytest[i] == 1:
                tp+=1
            else:
                fp+=1
        else:
            if bayes.ytest[i] == 1:
                fn+=1
            else:
                tn+=1

    print("fn", fn ," fp ", fp , " tn " , tn , " tp ", tp)
   # print(voted_output)
    # print("--------------COMBINED--------------")
    print(classification_report(bayes.ytest.tolist(),voted_output))
    # print(bayes_output)
    # print(supervised_output)
    # print(unsupervised_output)
