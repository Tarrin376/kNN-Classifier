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
    num_classes = len(Task_1_5.classification_scheme)
    # The confusion matrix
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]
    # Maps each class to its index in the 'classification_scheme' array.
    indices = {class_type:index for index, class_type in enumerate(Task_1_5.classification_scheme)}

    for data in classified_data[1:]:
        # Index of the actual class in the confusion matrix.
        actual_class_index = indices.get(data[1])
        # Index of the predicted class in the confusion matrix.
        predicted_class_index = indices.get(data[2])
        # Increment the number of times that the given 'predicted class' was found with the given 'actual class'.
        confusion_matrix[predicted_class_index][actual_class_index] += 1
    
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
    num_classes = len(Task_1_5.classification_scheme)
    # Array of the number of times each class was predicted correctly (true positives).
    tps = [confusion_matrix[i][i] for i in range(num_classes)]
    return tps


def computeFPs(confusion_matrix):
    num_classes = len(Task_1_5.classification_scheme)
    # Array of the number of times each class was incorrectly predicted to belong to that class (false positives).
    fps = [sum(confusion_matrix[i]) - confusion_matrix[i][i] for i in range(num_classes)]
    return fps


def computeFNs(confusion_matrix):
    num_classes = len(Task_1_5.classification_scheme)
    transpose_mat = numpy.transpose(confusion_matrix)
    # Array of the number of times each class was incorrectly predicted not to belong to that class (false negatives).
    fns = [sum(transpose_mat[i]) - transpose_mat[i][i] for i in range(num_classes)]
    return fns


# These functions compute the evaluation measures based on the provided values. Not all measures use of all the values.
#
# INPUT: tps, fps, fns, data_size
#                       : the per-class true positives, false positive and negatives, and size of the classified data.
# OUTPUT: appropriate evaluation measures created using the macro-average approach.

def computeMacroPrecision(tps, fps, fns, data_size):
    # To prevent divide by zero error.
    if data_size == 0:
        return 0

    # Precision sum used to calculate the macro precision.
    precision_sum = 0
    for i in range(data_size):
        # Calculate precision of the current class and add to precision sum.
        precision = tps[i] / (tps[i] + fps[i]) if tps[i] + fps[i] > 0 else 0
        precision_sum += precision
    
    # The macro precision.
    return precision_sum / data_size


def computeMacroRecall(tps, fps, fns, data_size):
    # To prevent divide by zero error.
    if data_size == 0:
        return 0
    
    # Recall sum used to calculate the macro recall.
    recall_sum = 0
    for i in range(data_size):
        # Calculate recall of the current class and add to recall sum.
        recall = tps[i] / (tps[i] + fns[i]) if tps[i] + fns[i] > 0 else 0
        recall_sum += recall

    # The macro recall.
    return recall_sum / data_size


def computeMacroFMeasure(tps, fps, fns, data_size):
    # To prevent divide by zero error.
    if data_size == 0:
        return 0

    # F-measure sum used to calculate the macro f-measure.
    f_measure_sum = 0
    for i in range(data_size):
        # Calculate precision of the current class.
        precision = tps[i] / (tps[i] + fps[i]) if tps[i] + fps[i] > 0 else 0
        # Calculate recall of the current class.
        recall = tps[i] / (tps[i] + fns[i]) if tps[i] + fns[i] > 0 else 0

        # To prevent divide by zero error.
        if precision + recall != 0:
            # Calculate f-measure of the current class and add to f-measure sum.
            f_measure = (2 * precision * recall) / (precision + recall)
            f_measure_sum += f_measure

    # The macro f-measure.
    return f_measure_sum / data_size


def computeAccuracy(tps, fps, fns, data_size):
    # To prevent divide by zero error.
    if data_size == 0:
        return 0
    
    # Ratio of the number of true positives to the number of rows in the dataset.
    accuracy = sum(tps) / data_size
    return accuracy


# In this function you are expected to compute precision, recall, f-measure and accuracy of your classifier using
# the macro average approach.

# INPUT: classified_data   : a numpy array containing paths to images, actual classes and predicted classes.
#                            Please refer to Task 1 for precise format description.
#       confusion_func     : function to be invoked to compute the confusion matrix
#
# OUTPUT: computed measures
def evaluateKNN(classified_data, confusion_func=confusionMatrix):
    confusion_matrix = confusion_func(classified_data)
    num_classes = len(Task_1_5.classification_scheme)

    # Number of true positives in the confusion matrix.
    tps = computeTPs(confusion_matrix)
    # Number of false positives in the confusion matrix.
    fps = computeFPs(confusion_matrix)
    # Number of false negatives in the confusion matrix.
    fns = computeFNs(confusion_matrix)
    
    precision = computeMacroPrecision(tps, fps, fns, num_classes)
    recall = computeMacroRecall(tps, fps, fns, num_classes)
    f_measure = computeMacroFMeasure(tps, fps, fns, num_classes)
    accuracy = computeAccuracy(tps, fps, fns, len(classified_data) - 1)

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
