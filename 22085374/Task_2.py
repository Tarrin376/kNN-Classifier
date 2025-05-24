import Task_1_5
import numpy

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
