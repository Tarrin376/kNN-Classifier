# In this file please complete the following tasks:
#
# Task 4 [3] The curse of k
#
# Independent inquiry time! Picking the right number of neighbours k in the kNN approach is tricky.
# Find a way you could approach this more rigorously. In comments:
# •	state the name of the approach you could use,
# •	give a one-sentence explanation of the approach, and
# •	provide a reference to it (use Cardiff University Harvard style, DOI MUST BE PRESENT).
#
# The reference must be a handbook or peer-reviewed publication; a link to an online tutorial will not be accepted.
# Ensure that your resources are respectable and are not e.g., predatory journals.

# We could use the kTree method. This method involves constructing a decision tree, known as a kTree, 
# to learn the optimal k value for each test data point based on the training samples.

# Zhang, S., Li, X., Zong, M., Zhu, X., & Wang, R. (2018). Efficient kNN Classification With Different Numbers of Nearest Neighbors. 
# IEEE Transactions on Neural Networks and Learning Systems, 29(5), 1774-1785. doi:10.1109/TNNLS.2017.2673241.


#
# You can start working on this task immediately. Please consult at the very least Week 2 materials.
#
# Task 6 [3] I can do better!
#
# Independent inquiry time! There are much better approaches out there for image classification.
# Your task is to find one, and using the comment section of your project, do the following:
# •	State the name of the approach
# •	Provide a permalink to a resource in the Cardiff University library that describes the approach
# •	Briefly explain how the approach you found is better than kNN in image classification (2-3 sentences is enough).
#  Focus on synthesis, not recall!
#
# You can start working on this task immediately. Please consult at the very least Week 2 materials.

# An approach that we could use is Convolutional Neural Networks (CNNs). Using CNNs provides more accurate results
# compared to kNN for image classification as CNNs use features within the images to classify them, unlike kNN
# which compares images by raw pixel data. Additionally, as CNNs use convolutions, they can better handle issues that kNN faces 
# in image classification such as rotation, scaling, and repositioning which enables it to effectively compare two images that have different
# resolutions, positions, etc.

# https://librarysearch.cardiff.ac.uk/permalink/44WHELF_CAR/1fseqj3/alma9911772737802420