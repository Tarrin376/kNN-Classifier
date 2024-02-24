# In this file please complete the following task:
#
# Task 2 [4] Basic evaluation
#
# Evaluate your classifiers. On your own, implement a method that will create a confusion matrix based on the provided
# classified data. Then implement methods that will output precision, recall, F-measure, and accuracy of your classifier
# based on your confusion matrix. Use macro-averaging approach and be mindful of edge cases. The template contains
# a range of functions you need to implement for this task.
#
# You can start working on this task immediately. Please consult at the very least Week 3 materials.
#
# You are expected to rely on solutions from Task_1_5 here! Do not reimplement kNN from scratch. You can ONLY rely
# on functions that were originally in the template in your final submission! Any functions you have created on your
# own and need here, must be defined here.

import Task_1_5
import Dummy
import numpy

# This function computes the confusion matrix based on the provided data.
#
# INPUT: classified_data   : a numpy arrays containing paths to images, actual classes and predicted classes.
#                            Please refer to Task 1 for precise format description.
# OUTPUT: confusion_matrix : the confusion matrix computed based on the classified_data.
#                            The order of elements MUST be the same as  in the classification scheme.
#                            The columns correspond to actual classes and rows to predicted classes.
#                            In other words, confusion_matrix[0] should be understood
#                            as the row of values predicted as Female, and [row[0] for row in confusion_matrix] as the
#                            column of values that were actually Female

def confusionMatrix(classified_data):
    confusion_matrix = []
    return confusion_matrix


# These functions compute per-class true positives and false positives/negatives based on the provided confusion matrix.
#
# INPUT: confusion_matrix : the confusion matrix computed based on the classified_data. The order of elements is
#                           the same as  in the classification scheme. The columns correspond to actual classes
#                           and rows to predicted classes.
# OUTPUT: a list of appropriate true positive, false positive or false
#         negative values per a given class, in the same order as in the classification scheme. For example, tps[1]
#         corresponds for TPs for Male class.


def computeTPs(confusion_matrix):
    tps = []
    return tps


def computeFPs(confusion_matrix):
    fps = []
    return fps


def computeFNs(confusion_matrix):
    fns = []
    return fns


# These functions compute the evaluation measures based on the provided values. Not all measures use of all the values.
#
# INPUT: tps, fps, fns, data_size
#                       : the per-class true positives, false positive and negatives, and size of the classified data.
# OUTPUT: appropriate evaluation measures created using the macro-average approach.

def computeMacroPrecision(tps, fps, fns, data_size):
    precision = float(0)
    return precision


def computeMacroRecall(tps, fps, fns, data_size):
    recall = float(0)
    return recall


def computeMacroFMeasure(tps, fps, fns, data_size):
    f_measure = float(0)
    return f_measure


def computeAccuracy(tps, fps, fns, data_size):
    accuracy = float(0)
    return accuracy


# In this function you are expected to compute precision, recall, f-measure and accuracy of your classifier using
# the macro average approach.

# INPUT: classified_data   : a numpy array containing paths to images, actual classes and predicted classes.
#                            Please refer to Task 1 for precise format description.
#       confusion_func     : function to be invoked to compute the confusion matrix
#
# OUTPUT: computed measures
def evaluateKNN(classified_data, confusion_func=confusionMatrix):
    precision = float(-1)
    recall = float(-1)
    f_measure = float(-1)
    accuracy = float(-1)
    # Have fun with the computations!

    # once ready, we return the values
    return precision, recall, f_measure, accuracy


##########################################################################################
# You should not need to modify things below this line - it's mostly reading and writing #
# Be aware that error handling below is...limited.                                       #
##########################################################################################


# This function reads the necessary arguments (see parse_arguments function in Task_1_5),
# and based on them evaluates the kNN classifier.
def main():
    opts = Task_1_5.parseArguments()
    if not opts:
        exit(1)
    print(f'Reading data from {opts["classified_data"]}')
    classified_data = Task_1_5.readCSVFile(opts['classified_data'])
    print('Evaluating kNN')
    result = evaluateKNN(classified_data, eval(opts['cf']))
    print('Result: precision {}; recall {}; f-measure {}; accuracy {}'.format(*result))


if __name__ == '__main__':
    main()
