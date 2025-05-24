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

student_id = 'c22085374'
classification_scheme = ['Female', 'Male', 'Primate', 'Rodent', 'Food']

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

def readAndResize(image_path, width=60, height=30):
    # Read in the rgb image data from the image path given.
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # If no image was found at the specified file path, return empty array.
    if image is None:
        return numpy.array([])
    
    # Resize the image with the given width and height.
    resized_img = cv2.resize(image, (width, height))

    return resized_img

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

def getMostCommonClass(nearest_neighbours_classes):
    # Keeps track of the most number of times a given class was found in the K nearest neighbours.
    bestTimesFound = 0
    # Keeps track of the class that occurs the most number of times in the K nearest neighbours.
    winner = ''

    for class_type in classification_scheme:
        # Number of times the class type was found in the K nearest neighbours.
        timesFound = nearest_neighbours_classes.get(class_type, 0)

        # If it was found more than the current most frequent class, update.
        if timesFound > bestTimesFound:
            bestTimesFound = timesFound
            winner = class_type

    return winner if bestTimesFound > 0 else ''
    
def kNN(training_data, k, measure_func, similarity_flag, data_to_classify,
        most_common_class_func=getMostCommonClass, get_neighbour_classes_func=getClassesOfKNearestNeighbours,
        read_func=readAndResize):
    # This sets the header list
    classified_data = [['Path', 'ActualClass', 'PredictedClass']]

    # Check that the training data and the data to classify is valid.
    if (len(training_data) == 0 or len(data_to_classify) == 0 or not validateDataFormat(training_data, False) 
        or not validateDataFormat(data_to_classify, False)):
        return numpy.array(classified_data)

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
        return numpy.array([classified_data[0]])

    return numpy.array(classified_data)

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

def readCSVFile(filename):
    lines = []
    with open(filename, newline='') as infile:
        reader = csv.reader(infile)
        for line in reader:
            lines.append(line)
    return numpy.array(lines)

def writeCSVFile(filename, lines):
    with open(filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(lines)

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
