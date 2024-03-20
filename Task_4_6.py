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

# We could use the kTree method. This approach involves learning learning the optimal k values for different
# test data by constructing a decision tree called a kTree that can learn the optimal k values for all training samples.

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

# An approach that we could use is creating fuzzy rules repeatedly based on the most notable features in an image.
# Unlike kNN, this approach enables us to be much more accurate for image classification as we directly compare
# images based on their main features without relying too much on our training data. This classification method is
# very fast and is easily expandable as no re-learning of classifiers is required, only new fuzzy rules need to be
# generated. This will take much less time compared to having to re-train the data when classes are added using kNN.

# https://librarysearch.cardiff.ac.uk/permalink/44WHELF_CAR/b7291a/cdi_elsevier_sciencedirect_doi_10_1016_j_ins_2015_08_030