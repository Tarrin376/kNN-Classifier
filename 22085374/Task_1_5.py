import argparse
import csv
import distutils.util
import distutils
import os
import numpy
import cv2
import skimage
import heapq
import math
from skimage.metrics import mean_squared_error, structural_similarity
from scipy.spatial import distance

# Task 1 [10] My first not-so-pretty image classifier
#
# By using the kNN approach and three distance or similarity measures, build image classifiers.
#•	You must implement the kNN approach yourself
#•	You must invoke the distance or similarity measures from libraries (it is fine to invoke different measures from
#   one library). Non-trivial adjustments to a library-invoked measure do not meet the requirements!
#•	Histogram-based measures are not allowed
#•	Jaccard distances/similarities are not allowed
#•	You can use between 0 and 3 distance measures and between 0 and 3 similarity measures
#   (there is no requirement that at least one of each kind should be present)
#
# The classifier is expected to use only one measure at a time and take information as to which one to invoke at a given
# time as input. The template contains a range of functions you must implement and use appropriately for this task.
#
# You can start working on this task immediately. Please consult at the very least Week 2 materials.

# Task 5 [4] Similarities
#
# Independent inquiry time! In Task 1, you were instructed to use libraries for image similarity measures.
# Pick two of the three measures you have used and implement them yourself.
# You are allowed to use libraries to e.g., calculate the root, power, average or standard deviation of some set
# (but, for example, numpy.linalg.norm is not permitted).
# The template contains a range of functions you need to implement for this task.
#
# Disclaimer: if you decide to implement MSE, do not implement RMSE (and vice versa)
#
# You can start working on this task immediately. Please consult at the very least Week 1 materials.


# Please replace with your student id, including the "c" at the beginning!!!
student_id = 'c22085374'

# This is the classification scheme you should use for kNN
classification_scheme = ['Female', 'Male', 'Primate', 'Rodent', 'Food']


# In this function, please implement validation of the data that is supplied to or produced by the kNN classifier.
#
# INPUT:  data              : numpy array that was read from the training data or data to classify csv
#                             (see parse_arguments function) or produced by the kNN function
#         predicted         : a boolean value stating whether the "PredictedClass" column should be present
#
# OUTPUT: boolean value     : True if the data contains the header ["Path", "ActualClass"] if predicted variable
#                             is False and ["Path", "ActualClass", "PredictedClass"] if it is True
#                             (there can be more column names, but at least these three at the start must be present)
#                             AND the values in the "Path" column (if there are any) are file paths
#                             AND the values in the "ActualClass" column (if there are any) are classes from scheme
#                             AND (if predicted is True) the values in the "PredictedClass" column (if there are any)
#                             are classes from scheme
#                             AND there are as many Path entries as ActualClass (and PredictedClass, if predicted
#                             is True) entries
#
#                             False otherwise

def validateDataFormat(data, predicted):
    header = ",".join(data[0])

    # Check the validity of the header row in the data
    if header != f"Path,ActualClass{',PredictedClass' if predicted else ''}":
        return False

    # Check if there exists a row that has a length less than 3 or 2 (depending on if the 'PredictedClass' column is present or not).
    if any(len(row) < (3 if predicted else 2) for row in data[1:]):
        return False
    
    # Check if there exists an image path in the data that doesn't lead to an existing file.
    if any(not os.path.isfile(row[0]) for row in data[1:]):
        return False
    
    # Check if there exists a value in the 'ActualClass' column that is not from the scheme
    if any(row[1] not in classification_scheme for row in data[1:]):
        return False
    
    # If the data contains a 'PredictedClass' column, check whether any value in the column is not from the scheme
    if predicted and any(row[2] not in classification_scheme for row in data[1:]):
        return False
    
    return True

# This function does reading and resizing of an image located in a give path on your drive.
# DO NOT REMOVE ANY COLOURS. DO NOT MODIFY PATHS. DO NOT FLATTEN IMAGES.
#
# INPUT:  imagePath         : path to image. DO NOT MODIFY - take from the file as-is. Things like appending "..\"
#                             to the file path within the code are not permitted.
#         width, height     : width and height dimensions to which you are asked to resize your image
#
# OUTPUT: image             : numpy array representing the read and resized image in RGB format
#                             (empty if the image is not found at a given path).
#                             Removing colour channels (e.g. transforming array to grayscale) or flattening the image
#                             ARE NOT PERMITTED.
#

def readAndResize(image_path, width=60, height=30):
    # Read in the rgb image data from the image path given.
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Resize the image with the given width and height
    resized_img = numpy.array(cv2.resize(image, (width, height)))

    return resized_img


# These functions compute the distance or similarity value between two images according to a particular
# similarity or distance measure. Return nan if images are empty. These three measures must be
# computed by libraries according to portfolio requirements.
#
# INPUT:  image1, image2    : two numpy arrays representing images in RGB formats. Do NOT presume a particular height
#                             or width! If you need images flattened or in grayscale or in any other format, then these
#                             manipulations will need to take place WITHIN the computeMeasure functions.
#
# OUTPUT: value             : the distance or similarity value between image1 and image2 according to a chosen approach.
#                             Defaults to nan if images are empty.
#

def computeMeasure1(image1, image2):
    # Mean Squared Error (MSE)

    # Check if either image is empty
    if len(image1) == 0 or len(image2) == 0:
        return float('nan')

    return mean_squared_error(image1, image2)

def computeMeasure2(image1, image2):
    # Cosine Distance

    # Check if either image is empty
    if len(image1) == 0 or len(image2) == 0:
        return float('nan')
    
    # Flatten image1 to 1-dimensional array
    flattened_img1 = image1.flatten().astype(float)
    # Flatten image2 to 1-dimensional array
    flattened_img2 = image2.flatten().astype(float)

    return distance.cosine(flattened_img1, flattened_img2)

def computeMeasure3(image1, image2):
    # Structural Similarity

    # Check if either image is empty
    if len(image1) == 0 or len(image2) == 0:
        return float('nan')

    return structural_similarity(image1, image2, channel_axis=-1, data_range=255)


# These functions compute the distance or similarity value between two images according to a particular similarity or
# distance measure. Return nan if images are empty. As name suggests, selfComputeMeasure 1 has to be your own
# implementation of the measure you have used in computeMeasure1 (same for 2). These two measures cannot be computed by
# libraries according to portfolio requirements.
#
    # INPUT:  image1, image2    : two numpy arrays representing images in RGB formats. Do NOT presume a particular height
    # #                           or width! If you need images flattened or in grayscale or in any other format, then these
    # #                           manipulations will need to take place WITHIN the computeMeasure functions.
#
# OUTPUT: value             : the distance or similarity value between image1 and image2 according to a chosen approach.
#                             Defaults to nan if images are empty.
#

def selfComputeMeasure1(image1, image2):
    # Mean Squared Error (MSE)

    # Check if either image is empty
    if len(image1) == 0 or len(image2) == 0:
        return float('nan')
    
    # Flatten image1 to 1-dimensional array
    flattened_img1 = image1.flatten().astype(float)
    # Flatten image2 to 1-dimensional array
    flattened_img2 = image2.flatten().astype(float)

    mse = ((flattened_img1 - flattened_img2) ** 2).mean()
    return mse

def selfComputeMeasure2(image1, image2):
    # Cosine Distance

    # Check if either image is empty
    if len(image1) == 0 or len(image2) == 0:
        return float('nan')

    # Calculates the dot product of two vectors 'a' and 'b'.
    def dot(a, b):
        return sum(a * b)
    
    # Flatten image1 to 1-dimensional array
    flattened_img1 = image1.flatten().astype(float)
    # Flatten image2 to 1-dimensional array
    flattened_img2 = image2.flatten().astype(float)

    # Norm of image1
    img1Norm = math.sqrt(dot(flattened_img1, flattened_img1))
    # Norm of image2
    img2Norm = math.sqrt(dot(flattened_img2, flattened_img2))

    # Cosine distance formula
    return 1 - dot(flattened_img1, flattened_img2) / (img1Norm * img2Norm)


# This function is supposed to return a dictionary of classes and their occurrences as taken from k nearest neighbours.
#
# INPUT:  measure_classes   : a list of lists that contain two elements each - a distance/similarity value
#                             and class from scheme
#         k                 : the value of k neighbours
#         similarity_flag   : a boolean value stating that the measure used to produce the values above is a distance
#                             (False) or a similarity (True)
# OUTPUT: nearest_neighbours_classes
#                           : a dictionary that, for each class in the scheme, states how often this class
#                             was in the k nearest neighbours
#
def getClassesOfKNearestNeighbours(measures_classes, k, similarity_flag):
    nearest_neighbours_classes = {cur_class: 0 for cur_class in classification_scheme}
    # Priority Queue using a Min-Heap to get K nearest neighbours
    pq = []

    for mClass in measures_classes:
        # If the Priority Queue does not have K elements on it, add class to Priority Queue.
        if len(pq) < k:
            heapq.heappush(pq, [mClass[0] if similarity_flag else -mClass[0], mClass[1]])
        
        # Invert the measure if it is a distance measure, to maintain Min-Heap structure.
        measure = mClass[0] if similarity_flag else -mClass[0]

        # If the current row is closer compared to the furthest row on the Priority Queue.
        if measure > pq[0][0]:
            # Remove furthest row from Priority Queue.
            heapq.heappop(pq)
            # Add current row to Priority Queue.
            heapq.heappush(pq, [measure, mClass[1]])

    # Remove all the K nearest neighbours from the Priority Queue.
    while pq:
        cur = heapq.heappop(pq)
        # Increment count for how many times a particular class from the scheme was found in the K nearest neighbours.
        nearest_neighbours_classes[cur[1]] = nearest_neighbours_classes.get(cur[1]) + 1

    return nearest_neighbours_classes


# Given a dictionary of classes and their occurrences, returns the most common class. In case there are multiple
# candidates, it follows the order of classes in the scheme. The function returns empty string if the input dictionary
# is empty, does not contain any classes from the scheme, or if all classes in the scheme have occurrence of 0.
#
# INPUT: nearest_neighbours_classes
#                           : a dictionary that, for each class in the scheme, states how often this class
#                             was in the k nearest neighbours
#
# OUTPUT: winner            : the most common class from the classification scheme. In case there are
#                             multiple candidates, it follows the order of classes in the scheme. Returns empty string
#                             if the input dictionary is empty, does not contain any classes from the scheme,
#                             or if all classes in the scheme have occurrence of 0
#

def getMostCommonClass(nearest_neighbours_classes):
    # Keeps track of the most number of times a given class was found in the K nearest neighbours.
    bestTimesFound = 0
    # Keeps track of the class that occurs the most number of times in the K nearest neighbours.
    winner = ''

    for class_type in classification_scheme:
        # Number of times the class type was found in the K nearest neighbours.
        timesFound = nearest_neighbours_classes.get(class_type)

        # If it was found more than the current most frequent class, update.
        if timesFound > bestTimesFound:
            bestTimesFound = timesFound
            winner = class_type

    return winner if bestTimesFound > 0 else ''


# In this function I expect you to implement the kNN classifier. You are free to define any number of helper functions
# you need for this! You need to use all of the other functions in the part of the template above.
#
# INPUT:  training_data       : a numpy array that was read from the training data csv
#         k                   : the value of k neighbours
#         measure_func        : the function to be invoked to calculate similarity/distance (any of the above)
#         similarity_flag     : a boolean value stating that the measure above used to produce the values is a distance
#                             (False) or a similarity (True)
#         data_to_classify    : a numpy array  that was read from the data to classify csv;
#                             this data is NOT be used for training the classifier, but for running and testing it
#                             (see parse_arguments function)
#     most_common_class_func  : the function to be invoked to find the most common class among the neighbours
#                             (by default, it is the one from above)
# get_neighbour_classes_func  : the function to be invoked to find the classes of nearest neighbours
#                             (by default, it is the one from above)
#         read_func           : the function to be invoked to find to read and resize images
#                             (by default, it is the one from above)
#  OUTPUT: classified_data    : a numpy array which expands the data_to_classify with the results on how your
#                             classifier has classified a given image.
#                             IF the training_data or data_to_classify is empty OR
#                             training_data, data_to_classify, or produced classified_data fail validation,
#                             the returned array contains ONLY the header row


def kNN(training_data, k, measure_func, similarity_flag, data_to_classify,
        most_common_class_func=getMostCommonClass, get_neighbour_classes_func=getClassesOfKNearestNeighbours,
        read_func=readAndResize):
    # This sets the header list
    classified_data = [['Path', 'ActualClass', 'PredictedClass']]

    # Check that the training data and the data to classify is valid.
    if (len(training_data) == 0 or len(data_to_classify) == 0 or not validateDataFormat(training_data, False) 
        or not validateDataFormat(data_to_classify, False)):
        return classified_data

    # Precompute the training data images and images that will be classified.
    data_to_classify_images = [read_func(data[0]) for data in data_to_classify[1:]]
    training_data_images = [read_func(tData[0]) for tData in training_data[1:]]

    for dIndex, data in enumerate(data_to_classify[1:]):
        # Read image from file path.
        data_image = data_to_classify_images[dIndex]
        # Stores the distance/similarity of all the training_data to the current image to classify.
        measures_classes = [None] * (len(training_data) - 1)

        for tIndex, tData in enumerate(training_data[1:]):
            # Measure the distance/similarity between the current image to classify and the current training row image.
            measure = measure_func(data_image, training_data_images[tIndex])
            measures_classes[tIndex] = [measure, tData[1]]
        
        # Get the classes found in the K nearest neighbours.
        nearest_neighbours_classes = get_neighbour_classes_func(measures_classes, k, similarity_flag)
        # Find the class that occurred the most in the K nearest neighbours.
        winner = most_common_class_func(nearest_neighbours_classes)
        # Add data along with its computed class type 'winner'.
        classified_data.append([data[0], data[1], winner])
    
    # If the classified data format is not valid, return only the header.
    if not validateDataFormat(classified_data, True):
        return [classified_data[0]]

    return numpy.array(classified_data)


##########################################################################################
# Do not modify things below this line - it's mostly reading and writing #
# Be aware that error handling below is...limited.                                       #
##########################################################################################


# This function reads the necessary arguments (see parse_arguments function), and based on them executes
# the kNN classifier. If the "unseen" mode is on, the results are written to a file.

def main():
    opts = parseArguments()
    if not opts:
        exit(1)
    print(f'Reading data from {opts["training_data"]} and {opts["data_to_classify"]}')
    training_data = readCSVFile(opts['training_data'])
    data_to_classify = readCSVFile(opts['data_to_classify'])
    unseen = opts['mode']
    print('Running kNN')
    print(opts['simflag'])
    result = kNN(training_data, opts['k'], eval(opts['measure']), opts['simflag'], data_to_classify,
                 eval(opts['mcc']), eval(opts['gnc']), eval(opts['rrf']))
    if unseen:
        path = os.path.dirname(os.path.realpath(opts['data_to_classify']))
        out = f'{path}/{student_id}_classified_data.csv'
        print(f'Writing data to {out}')
        writeCSVFile(out, result)


# Straightforward function to read the data contained in the file "filename"
def readCSVFile(filename):
    lines = []
    with open(filename, newline='') as infile:
        reader = csv.reader(infile)
        for line in reader:
            lines.append(line)
    return numpy.array(lines)


# Straightforward function to write the data contained in "lines" to a file "filename"
def writeCSVFile(filename, lines):
    with open(filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(lines)


# This function simply parses the arguments passed to main. It looks for the following:
#       -k              : the value of k neighbours
#                         (needed in Tasks 1, 2, 3 and 5)
#       -f              : the number of folds to be used for cross-validation
#                         (needed in Task 3)
#       -measure        : function to compute a given similarity/distance measure
#       -simflag        : flag telling us whether the above measure is a distance (False) or similarity (True)
#       -u              : flag for how to understand the data. If -u is used, it means data is "unseen" and
#                         the classification will be written to the file. If -u is not used, it means the data is
#                         for training purposes and no writing to files will happen.
#                         (needed in Tasks 1, 3 and 5)
#       training_data   : csv file to be used for training the classifier, contains two columns: "Path" that denotes
#                         the path to a given image file, and "Class" that gives the true class of the image
#                         according to the classification scheme defined at the start of this file.
#                         (needed in Tasks 1, 2, 3 and 5)
#       data_to_classify: csv file formatted the same way as training_data; it will NOT be used for training
#                         the classifier, but for running and testing it
#                         (needed in Tasks 1, 2, 3 and 5)
#       mcc, gnc, rrf, vf,cf,sf,al
#                       : staff variables, do not use
#
def parseArguments():
    parser = argparse.ArgumentParser(description='Processes files ')
    parser.add_argument('-k', type=int)
    parser.add_argument('-f', type=int)
    parser.add_argument('-m', '--measure')
    parser.add_argument('-s', '--simflag', type=lambda x:bool(distutils.util.strtobool(x)))
    parser.add_argument('-u', '--unseen', action='store_true')
    parser.add_argument('-train', type=str)
    parser.add_argument('-test', type=str)
    parser.add_argument('-classified', type=str)
    parser.add_argument('-mcc', default="getMostCommonClass")
    parser.add_argument('-gnc', default="getClassesOfKNearestNeighbours")
    parser.add_argument('-rrf', default="readAndResize")
    parser.add_argument('-cf', default="confusionMatrix")
    parser.add_argument('-sf', default="splitDataForCrossValidation")
    parser.add_argument('-al', default="Task_1_5.kNN")
    params = parser.parse_args()

    opt = {'k': params.k,
           'f': params.f,
           'measure': params.measure,
           'simflag': params.simflag,
           'training_data': params.train,
           'data_to_classify': params.test,
           'classified_data': params.classified,
           'mode': params.unseen,
           'mcc': params.mcc,
           'gnc': params.gnc,
           'rrf': params.rrf,
           'cf': params.cf,
           'sf': params.sf,
           'al': params.al
           }
    return opt


if __name__ == '__main__':
    main()
