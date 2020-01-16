from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest,chi2
from matplotlib import pyplot as plt

def VarThr():
    X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
    sel = VarianceThreshold(threshold=(0.8*(1-0.8)))
    features = sel.fit_transform(X)
    print features
 
def Univariate():
    iris = load_iris()
    X, y = iris.data, iris.target
    print "Original size of features = ",X.shape
    X_reduced = SelectKBest(chi2,k=2).fit_transform(X,y)
    print "Reduced size of features =",X_reduced.shape
    plt.scatter(X_reduced[0:49,0],X_reduced[0:49,1])
    plt.scatter(X_reduced[50:99,0],X_reduced[50:99,1])
    plt.scatter(X_reduced[100:149,0],X_reduced[100:149,1])
    plt.title('Iris dataset in reduced two dimensional space')
if __name__ == "__main__":
    VarThr()
    Univariate()