'''============================ myClassifiers.py file ============================= %

Description                         : Classifier
Input parameter                     : 
Output parameter                    : 
Subroutine  called                  : NA
Called by                           : NA
Reference                           :
Author of the code                  : Kundan Kumar
Date of creation                    : 14 Oct 2017
------------------------------------------------------------------------------------------------------- %
Modified on                         : 
Modification details                : 
Modified By                         : Kundan Kumar
===================================================================== %
   Copy righted by ECE Department, ITER, SOA University India.
===================================================================== %'''

# Standard scientific Python imports
import matplotlib.pyplot as plt
# import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
# the digits dataset
digits = datasets.load_digits()

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset. If we were working from image files, we could load them using
# matplotlib.pyplot.imread. Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.

images_and_labels = list(zip(digits.images,digits.target))

for index, (image,label) in enumerate(images_and_labels[:5]):
    plt.subplot(2,5,index+1)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('Training: %i' % label)
    
# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:

nSamples = len(digits.images)
data = digits.images.reshape((nSamples,-1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma= 0.0001)
# We learn the digits on the first half of the digits
classifier.fit(data[:nSamples //2],digits.target[:nSamples //2])

# Now predict the value of the digit on the second half:
expected = digits.target[nSamples // 2:]
predicted = classifier.predict(data[nSamples //2:])

print("classification report for classifier %s:\n%s\n"
      %(classifier,metrics.classification_report(expected,predicted)))

ConfusionMatrix = metrics.confusion_matrix(expected, predicted)
ttt = ConfusionMatrix
print("confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[nSamples //2:],predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:5]):
    plt.subplot(2, 5, index + 6)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()



