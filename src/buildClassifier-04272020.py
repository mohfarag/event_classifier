import sys
from sklearn import datasets
import numpy as np

#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline , FeatureUnion
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
#from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

dataPath = sys.argv[1]
dataset = datasets.load_files(dataPath)
X, y,fn = dataset.data, dataset.target, dataset.filenames
#print(fn)
for i,f in enumerate(fn):
	if '.DS_Store' in f:
		#del X[i]
		#del y[i]
		X= np.delete(X,i)
		y = np.delete(y,i)

#posWeight = len(y)/len(y[y==1])
#print(posWeight)
#print(len(y) / (2.0 * np.bincount(y)))
#wclf = svm.SVC(kernel='linear', class_weight={1: posWeight})

Xtrain, Xtest,ytrain,ytest = train_test_split(X,y, test_size=0.2,random_state=2)

vectorizer = TfidfVectorizer(decode_error='ignore')
Xtrain_tfidf = vectorizer.fit_transform(Xtrain)

wclf = svm.SVC(kernel='linear', class_weight='balanced',probability=True)
wclf.fit(Xtrain_tfidf,ytrain)
#print(len(y))
#print(len(X))

#print(set(y))
#print(len(fn))

# predict probabilities
Xtest_tfidf = vectorizer.transform(Xtest)
# predict class values
yhat = wclf.predict(Xtest_tfidf)
# calculate scores
model_f1 = f1_score(ytest, yhat,average='weighted')

y_probs = wclf.predict_proba(Xtest_tfidf)
# keep probabilities for the positive outcome only
y_probs = y_probs[:, 1]

# calculate precision and recall for each threshold
model_precision, model_recall, _ = precision_recall_curve(ytest, y_probs)
model_auc = auc(model_recall, model_precision)
print('Model auc=%.3f' % (model_auc))
# summarize scores
#print('Model: f1=%.3f auc=%.3f' % (model_f1, model_auc))
print('Model f1=%.3f' % (model_f1))

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(ytest, yhat)

print('Average precision-recall score: {0:0.3f}'.format(average_precision))


acc = np.mean(yhat == ytest)
print ("Accuracy=%.3f" % acc )
print("\n")
print(classification_report(ytest, yhat))
print ("confusion matrix")
print(confusion_matrix(ytest, yhat))
    
## plot the precision-recall curves
##no_skill = len(testy[testy==1]) / len(testy)
##pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
#pyplot.plot(model_recall, model_precision, marker='.', label='Model')
## axis labels
#pyplot.xlabel('Recall')
#pyplot.ylabel('Precision')
## show the legend
#pyplot.legend()
## show the plot
#pyplot.show()


disp = plot_precision_recall_curve(wclf, Xtest_tfidf, ytest)
disp.ax_.set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
disp.plot()
plt.savefig('model-PR-curve.png')

