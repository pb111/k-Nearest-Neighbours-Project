# k-Nearest Neighbours with Python and Scikit-Learn

In this project, I build a K Nearest Neighbours classifier 


==============================================================================


## Table of Contents

I have categorized this project into various sections which are listed below:-





## 1. Introduction to k Nearest Neighbours (kNN) algorithm

In machine learning, kNN is the simplest of all machine learning algorithms. It is a non-parametric algorithm used for classification and regression tasks. Non-parametric means there is no assumption required for data distribution. So, kNN does not require any underlying assumption to be made. In both classification and regression tasks, the input consists of the k closest training examples in the feature space. The output depends upon whether kNN is used for classification or regression purposes.

-	In kNN classification, the output is a class membership. The given data point is classified based on the majority of type of its neighbours. The data point is assigned to the most frequent class among its k nearest neighbours. Usually k is a small positive integer. If k=1, then the data point is simply assigned to the class of that single nearest neighbour.

-	In kNN regression, the output is simply some property value for the object. This value is the average of the values of k nearest neighbours.


kNN is a type of instance-based learning or lazy learning. Lazy learning means it does not require any training data points for model generation. All training data will be used in the testing phase. This makes training faster and testing slower and costlier. So, the testing phase requires more time and memory resources.

In kNN, the neighbours are taken from a set of objects for which the class or the object property value is known. This can be thought of as the training set for the kNN algorithm, though no explicit training step is required. In both classification and regression kNN algorithm, we can assign weight to the contributions of the neighbours. So, nearest neighbours contribute more to the average than the more distant ones.



## 2. K Nearest Neighbours (kNN) intuition

The kNN algorithm intuition is very simple to understand. It simply calculates the distance between a sample data point and all the other training data points. The distance can be Euclidean distance or Manhattan distance. Then, it selects the k nearest data points where k can be any integer. Finally, it assigns the sample data point to the class to which the majority of the k data points belong.


### Euclidean distance

### Manhattan distance






## 3. kNN algorithm in action
Now, I will see kNN algorithm in action. Suppose, we have a dataset with two variables, which when plotted, looks like the following figure.

## D-kNN 1

In kNN algorithm, k is the number of nearest neighbours. Generally, k is an odd number because it helps to decide the majority of the class. When k=1, then the algorithm is known as the nearest neighbour algorithm.

Now, we want to classify a new data point `X` into `Blue` class or `Red` class. Suppose the value of k is 3. The kNN algorithm starts by calculating the distance between `X` and all the other data points. It then finds the 3 nearest points with least distance to point `X`. This is clearly shown in the figure below. The 3 nearest points have been encircled.

## D-kNN 2

In the final step of the kNN algorithm, we assign the new data point `X` to the majority of the class of the 3 nearest points. From the above figure, we can see that the 2 of the 3 nearest points belong to the class `Red` while 1 belong to the class `Blue`. Therefore, the new data point will be classified as `Red`.

## 4. How to decide the number of neighbours in kNN

## 5. Eager learners vs lazy learners
`Eager learners` mean when giving training data points, we will construct a generalized model before performing prediction on given new points to classify. We can think of such learners as being ready, active and eager to classify new data points. 
`Lazy learning` means there is no need for learning or training of the model and all of the data points are used at the time of prediction. Lazy learners wait until the last minute before classifying any data point. They merely store the training dataset and waits until classification needs to perform. Lazy learners are also known as `instance-based learners` because lazy learners store the training points or instances, and all learning is based on instances.
Unlike eager learners, lazy learners do less work in the training phase and more work in the testing phase to make a classification. 



## 6. Curse of dimensionality
kNN algorithm usually performs much better with a lower number of features than a large number of features. When the number of features increases, then it requires more data. Increase in dimension also leads to the problem of overfitting. To avoid overfitting, the data will need to grow exponentially as we increase the number of dimensions. This problem of higher dimension is known as the **Curse of Dimensionality**.
To deal with this problem of the curse of dimensionality, we have to perform `Principal Component Analysis` before applying any machine learning algorithm, or we can also use feature selection approach.


## 7. Advantages and disadvantages of kNN algorithm
The advantages of kNN algorithm are as follows:-
1.	kNN is simple to implement.
2.	It executes quickly for small training datasets.
3.	kNN performance is asymptotically similar to the performance of the Bayes classifier.
4.	It does not need any prior knowledge about the structure of the data in training set.
5.	In kNN algorithm, no retraining is required if the new training pattern is added to the existing training set.

The disadvantages of kNN algorithm are as follows:-
1.	When the training set is large, it may take a lot of space and memory.
2.	For every test data, the distance should be computed between test data and all the training data samples. Thus, a lot of time may be needed for testing.

## 8. The problem statement
## 9. Results and conclusion
## 10. Applications of kNN
## 11. References


