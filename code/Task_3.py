# In this file please complete the following task:
#
# Task 3 [6] Cross validation
#
# Evaluate your classifiers using the k-fold cross-validation technique covered in the lectures
# (use the training data only). Output their average precisions, recalls, F-measures and accuracies.
# You need to implement the validation yourself. Remember that folds need to be of roughly equal size.
# The template contains a range of functions you need to implement for this task.
#

# You are expected to rely on solutions from Task_1_5/Task_2 here! Do not reimplement kNN from scratch.
# You can ONLY rely on functions that were originally in the template in your final submission!
# Any functions you have created on your own in these files and need here, must be defined here.

import os
import Task_1_5
import Task_2
import numpy
from Task_1_5 import computeMeasure1,computeMeasure2,computeMeasure3,selfComputeMeasure1,selfComputeMeasure2


# This function takes the data for cross evaluation and returns training_data a list of lists s.t. the first element
# is the round number, second is the training data for that round, and third is the testing data for that round
#
# INPUT: training_data      : a numpy array was read from the training data csv (see parse_arguments function)
#        f                  : the number of folds to split the data into (which is also same as # of rounds)
# OUTPUT: folds             : a list of lists s.t. the first element is the round number, second is the numpy array
#                             representing the training data for that round, and third is the numpy array representing
#                             the testing data for that round
#                             You must make sure that the training and testing data are ready for use
#                             (e.g. contain the right headers already)

def splitDataForCrossValidation(training_data, f):
    header = numpy.array([training_data[0].copy()])
    data = training_data[1:]

    # Number of rows in the data (excluding the header).
    num_rows = len(data)
    # The size of each fold after splitting the data into 'f' folds.
    fold_size = num_rows // f
    # Remainder after splitting data into 'f' parts.
    rem = num_rows % f
    # List of lists that stores every fold.
    folds = []

    for i in range(0, f):
        # Start and end indices of fold i's test data.
        start = (i * fold_size) + 1 if rem >= 0 and num_rows % f != 0 and i > 0 else (i * fold_size)
        end = min(num_rows, start + fold_size + 1 if rem > 0 else start + fold_size)

        # Testing and training data for fold i.
        fold_training = numpy.concatenate((header, numpy.array(data[:start]), numpy.array(data[end:])), axis=0)
        fold_testing = numpy.concatenate((header, numpy.array(data[start:end])), axis=0)

        # Add an array containing the current fold number, testing data, and the training data.
        folds.append([i, fold_testing, fold_training])
        # Decrement the remainder to ensure that the splits are evenly divided with at most a difference of 1.
        rem -= 1
    
    return folds


# In this function, please implement validation of the data that is produced by the cross evaluation function PRIOR to
# the addition of rows with the average measures.
#
# INPUT:  data              : the numpy array that was produced by the crossEvaluateKNN function BEFORE the
#                             addition of the rows with evaluation measures
#         f                 : number of folds to validate against
#
# OUTPUT: boolean value     : True if the data contains the header ["Path", "ActualClass", "PredictedClass","FoldNumber"]
#                             (there can be more column names, but at least these four at the start must be present)
#                             AND the values in the "Path" column (if there are any) are file paths
#                             AND the values in the "ActualClass" and "PredictedClass" columns
#                             (if there are any) are classes from the scheme
#                             AND the values in the "FoldNumber" column are integers in [0,f) range
#                             AND there are as many Path entries as ActualClass and PredictedClass and FoldNumber entries
#                             AND the number of entries per each integer in [0,f) range for FoldNumber are approximately
#                             the same (they can differ by at most 1)
#
#                             False otherwise

def validateDataFormat(data, f):
    header = ",".join(data[0])

    # Check the validity of the header row in the data.
    if not header.startswith("Path,ActualClass,PredictedClass,FoldNumber"):
        return False

    # Check if there exists a row that has a length less than 4.
    if any(len(row) < 4 for row in data[1:]):
        return False
    
    # Check if there exists an image path in the data that doesn't lead to an existing file.
    if any(not os.path.isfile(row[0]) for row in data[1:]):
        return False
    
    # Check if there exists a value in the 'ActualClass' column that is not from the scheme.
    if any(row[1] not in Task_1_5.classification_scheme for row in data[1:]):
        return False
    
    # Check if there exists a value in the 'PredictedClass' column is not from the scheme.
    if any(row[2] not in Task_1_5.classification_scheme for row in data[1:]):
        return False
    
    # Check if any fold number entry is outside the bounds of 'f'.
    if (any(row[3] < 0 or row[3] >= f for row in data[1:])):
        return False
    
    # Frequency of the number of entries for each integer in [0,f) range for FoldNumber.
    freq = {}
    for row in data[1:]:
        freq[row[3]] = freq.get(row[3], 0) + 1

    minFreq = min(freq.values())
    maxFreq = max(freq.values())

    # Check if the difference betweeen the lowest and highest frequency entries is larger than 1.
    if maxFreq - minFreq > 1:
        return False

    return True


# This function takes the classified data from each cross validation round and calculates the average precision, recall,
# accuracy and f-measure for them.
# Invoke either the Task 2 evaluation function or the dummy function here, do not code from scratch!
#
# INPUT: classified_data_list
#                           : a list of numpy arrays representing classified data computed for each cross validation round
#        evaluation_func    : the function to be invoked for the evaluation (by default, it is the one from
#                             Task_2, but you can use dummy)
# OUTPUT: avg_precision, avg_recall, avg_f_measure, avg_accuracy
#                           : average evaluation measures. You are expected to evaluate every classified data in the
#                             list and average out these values in the usual way.

def evaluateCrossValidation(classified_data_list, evaluation_func=Task_2.evaluateKNN):
    rounds = len(classified_data_list)

    # Edge case, to prevent divide by zero exception.
    if rounds == 0:
        return 0, 0, 0, 0
    
    avg_precision = 0
    avg_recall = 0
    avg_f_measure = 0
    avg_accuracy = 0

    # Iterate through each classified round and compute its statistics and add them to the average sums.
    for round in classified_data_list:
        precision, recall, f_measure, accuracy = evaluation_func(round)
        avg_precision += precision
        avg_recall += recall
        avg_f_measure += f_measure
        avg_accuracy += accuracy

    # Compute the average of each statistic.
    return avg_precision / rounds, avg_recall / rounds, avg_f_measure / rounds, avg_accuracy / rounds


# In this task you are expected to perform cross-validation where f defines the number of folds to consider.
# "processed" holds the information from training data along with the following information: for each image,
# stated the id of the fold it landed in, and the predicted class it was assigned once it was chosen for testing data.
# After everything is done, we add the average measures at the end. The writing to csv is done in a different function.
# You are expected to invoke the Task 1 kNN classifier or the Dummy classifier here, do not implement these things
# from scratch!
#
# INPUT: training_data      : a numpy array that was read from the training data csv (see parse_arguments function)
#        k                  : the value of k neighbours, to be passed to the kNN classifier
#        measure_func       : the function to be invoked to calculate similarity/distance
#        similarity_flag    : a boolean value stating that the measure above used to produce the values is a distance
#                             (False) or a similarity (True)
#        knn_func           : the function to be invoked for the classification (by default, it is the one from
#                             Task_1_5, but you can use dummy)
#        split_func         : the function used to split data for cross validation (by default, it is the one above)
#        f                  : number of folds to use in cross validation
# OUTPUT: processed       : a list of lists which expands the training_data with columns stating the fold number to
#                             which a given image was assigned and the predicted class for that image; and with rows
#                             that contain the average evaluation measures (see the h and v variables)
#                             IF validation of the processed variable fails (prior to addition of evaluation measures),
#                             return only the header!
# Again, please remember to have a look at the Dummy file!
def crossEvaluateKNN(training_data, k, measure_func, similarity_flag, f, knn_func=Task_1_5.kNN,
                     split_func=splitDataForCrossValidation):
    # This adds the header
    processed = [['Path', 'ActualClass', 'PredictedClass', 'FoldNumber']]
    # List of lists containing each fold.
    folds = split_func(training_data, f)
    # List that will hold the classified data computed for each fold.
    classified_data_list = []

    for fold in folds:
        # The classified fold testing data.
        classified = knn_func(fold[2], k, measure_func, similarity_flag, fold[1])
        classified_data_list.append(classified)

        for row in classified[1:]:
            # Add the row consisting of the 'Path', 'ActualClass', 'PredictedClass', and 'FoldNumber' respectively.
            processed.append([row[0], row[1], row[2], fold[0]])
    
    # Validate the data format of 'processed' given the numbers of folds 'f'.
    if not validateDataFormat(processed, f):
        return numpy.array([processed[0]])
    
    # Averages of all classified rounds.
    avg_precision, avg_recall, avg_fMeasure, avg_accuracy = evaluateCrossValidation(classified_data_list)

    # The measures are now added to the end. You should invoke validation BEFORE this step.
    h = ['avg_precision', 'avg_recall', 'avg_f_measure', 'avg_accuracy']
    v = [avg_precision, avg_recall, avg_fMeasure, avg_accuracy]

    processed = numpy.array(processed)
    processed = numpy.append(processed, [h], axis=0)
    processed = numpy.append(processed, [v], axis=0)

    return processed


##########################################################################################
# You should not need to modify things below this line - it's mostly reading and writing #
# Be aware that error handling below is...limited.                                       #
##########################################################################################


# This function reads the necessary arguments (see parse_arguments function in Task_1_5),
# and based on them evaluates the kNN classifier using the cross-validation technique. The results
# are written into an appropriate csv file.
def main():
    opts = Task_1_5.parseArguments()
    if not opts:
        exit(1)
    print(f'Reading data from {opts["training_data"]}')
    training_data = Task_1_5.readCSVFile(opts['training_data'])
    print('Evaluating kNN')
    result = crossEvaluateKNN(training_data, opts['k'], eval(opts['measure']), opts['simflag'], opts['f'],
                              eval(opts['al']), eval(opts['sf']))
    path = os.path.dirname(os.path.realpath(opts['training_data']))
    out = f'{path}/{Task_1_5.student_id}_cross_validation.csv'
    print(f'Writing data to {out}')
    Task_1_5.writeCSVFile(out, result)


if __name__ == '__main__':
    main()
