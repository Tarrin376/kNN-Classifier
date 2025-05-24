import os
import Task_1_5
import Task_2
import numpy
from Task_1_5 import computeMeasure1,computeMeasure2,computeMeasure3,selfComputeMeasure1,selfComputeMeasure2

def splitDataForCrossValidation(training_data, f):
    header = numpy.array([training_data[0].copy()])
    data = training_data[1:]

    # The size of each fold after splitting the data into 'f' folds.
    fold_size = len(data) // f
    # Remainder after splitting data into 'f' parts.
    rem = len(data) % f
    # List of lists that stores every fold.
    folds = []
    # Start index of the current fold's test data.
    start = 0

    for i in range(f):
        # End index of fold i's test data.
        end = min(len(data), start + fold_size + 1 if rem > 0 else start + fold_size)

        # Testing and training data for fold i.
        fold_training = numpy.concatenate((header, numpy.array(data[:start]), numpy.array(data[end:])), axis=0)
        fold_testing = numpy.concatenate((header, numpy.array(data[start:end])), axis=0)

        # Add an array containing the current fold number, testing data, and the training data.
        folds.append([i, fold_testing, fold_training])
        # Decrement the remainder to ensure that the splits are evenly divided with at most a difference of 1.
        rem -= 1
        # Update next fold's start index
        start = end
    
    return folds

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

    minFreq = min(freq.values()) if len(freq) > 0 else 0
    maxFreq = max(freq.values()) if len(freq) > 0 else 0

    # Check if the difference betweeen the lowest and highest frequency entries is larger than 1.
    if maxFreq - minFreq > 1:
        return False

    return True

def evaluateCrossValidation(classified_data_list, evaluation_func=Task_2.evaluateKNN):
    rounds = len(classified_data_list)

    # To prevent divide by zero error.
    if rounds == 0:
        return 0, 0, 0, 0
    
    avg_precision = 0
    avg_recall = 0
    avg_f_measure = 0
    avg_accuracy = 0

    # Iterate through each classified round and compute its statistics and add them to the average sums.
    for cur_round in classified_data_list:
        precision, recall, f_measure, accuracy = evaluation_func(cur_round)
        avg_precision += precision
        avg_recall += recall
        avg_f_measure += f_measure
        avg_accuracy += accuracy

    # Compute the average of each statistic.
    return avg_precision / rounds, avg_recall / rounds, avg_f_measure / rounds, avg_accuracy / rounds

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
