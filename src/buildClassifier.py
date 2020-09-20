import sys
from sklearn import datasets

#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline , FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

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


#### column transformer######
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC

from getNER import extractLocsDates_Doc,loadNLP, extractDateFromMetaHTML
from extractTxtFromHTML import extractTextFromHTML


textDataPath = sys.argv[1]
htmlFilesPath = sys.argv[2]
urlFilesPath = sys.argv[3]

#htmlFolders = ['0','1']
html_dataset = datasets.load_files(htmlFilesPath,shuffle=False)
htmlX, htmly,htmlfn = html_dataset.data, html_dataset.target, html_dataset.filenames
#print(fn)
j=[]
for i,f in enumerate(htmlfn):
    if '.DS_Store' in f:
        #del X[i]
        #del y[i]
        j.append(i)
        htmlX= np.delete(htmlX,i)
        htmly = np.delete(htmly,i)
if len(j):
	for ji in j:
		htmlfn = np.delete(htmlfn,ji)
dataset = datasets.load_files(textDataPath,shuffle=False)
X, y,fn = dataset.data, dataset.target, dataset.filenames
#print(fn)
j = []
for i,f in enumerate(fn):
    if '.DS_Store' in f:
        #del X[i]
        #del y[i]
        j.append(i)
        X= np.delete(X,i)
        y = np.delete(y,i)
if len(j):
	for ji in j:
		fn = np.delete(fn,j)
#urlsFilenames = ['relevant', 'not-relevant']
from os import listdir
from os.path import isfile, join
urlsFilenames = [f for f in listdir(urlFilesPath) if isfile(join(urlFilesPath, f)) and f != '.DS_Store']
print(urlsFilenames)

urlData = {}
for urlfilename in urlsFilenames:
	urlfile = join(urlFilesPath, urlfilename)
	with open(urlfile) as f:
		urlList = f.readlines()
		#if 'not' in urlfile:
		cat = urlfilename.split(".")[0]
		urlData[cat] = urlList
urlX = []
for i, doc in enumerate(y):
	#here filenames are complete
	#fn_ind = fn[i].split('.')[0]
	#print(fn[i])
	fn_last = fn[i].rsplit('/', 1) [1]
	#print(fn_last)
	fn_ind = fn_last.split('.')[0]
	print(fn_ind)
	l = urlData[str(y[i])]
	urlX.append(l[int(fn_ind)-1])
	
urlX = np.asarray(urlX)

allDataset= np.empty(shape=(len(X), 3), dtype=object)
allDataset[:,0] = X
allDataset[:,1] = htmlX
allDataset[:,2] = urlsX
# for i, doc in enumerate(X):
#             allDataset[i, 0] = doc['data']
#             #locs,dates = extractLocsDates(doc['text'])
#             locs,dates = extractLocsDates(doc['data'])
#             date = extractDateFromMetaHTML(doc['url'],doc['html'])
#             if date:
#                 dates.append(date)
#             features[i, 1] = locs
#             features[i, 2] = dates
            
            
#posWeight = len(y)/len(y[y==1])
#print(posWeight)
#print(len(y) / (2.0 * np.bincount(y)))
#wclf = svm.SVC(kernel='linear', class_weight={1: posWeight})

#Change X to a list of dictionary. Each dictionary contains text, url, and html fields

#Xtrain, Xtest,ytrain,ytest = train_test_split(X,y, test_size=0.2,random_state=2)
Xtrain, Xtest,ytrain,ytest = train_test_split(allDataset,y, test_size=0.2,random_state=2)


##################################################################
class TextStats(TransformerMixin, BaseEstimator):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        return [{'length': len(text),
                 'num_sentences': text.count('.')}
                for text in posts]


# class DateExtractor(TransformerMixin, BaseEstimator):
#     """Extract date from each document html and Url. it returns list of date objects"""
# 
#     def fit(self, x, y=None):
#         return self
# 
#     def transform(self, docs):
#       dates = []
#       for doc,url in docs.items():
#           date = extractDateFromMetaHTML(url,doc)
#           dates.append(date)
#         #return [{'length': len(text), 'num_sentences': text.count('.')} for text in posts]
#         return dates

### Started working here, complete from here
class NamedEntitiesExtractor(TransformerMixin, BaseEstimator):
    """Extract the Locations and Dates Named Entities from a document.

    Takes a sequence of strings and produces a dict of sequences.  Keys are `Locs` and `Dates`.
    """
    def fit(self, x, y=None):
        return self

    def transform(self, docs):
        # construct object dtype array with three columns
        # first column = 'text', second column = 'locs', and third column = 'dates'
        # here we assume each document is a dictionary with three fields: text, hmtl, url
        
        features = np.empty(shape=(len(docs), 3), dtype=object)
        loadNLP()
        
        #features[:, 0] = docs
        features[:,0] = docs[:,0]
        for i, doc in enumerate(docs):
            #features[i, 0] = doc['data']
            #locs,dates = extractLocsDates(doc['text'])
            locs,dates = extractLocsDates(doc[i,0])
            #date = extractDateFromMetaHTML(doc['url'],doc['html'])
            features[i, 1] = locs
            date = extractDateFromMetaHTML(doc[i,2],doc[i,1])
            if date:
                dates.append(date)
            
            features[i, 2] = dates
            #headers, _, bod = doc.partition('\n\n')
            #features[i, 1] = bod

            #prefix = 'Subject:'
            #sub = ''
            #for line in headers.split('\n'):
            #    if line.startswith(prefix):
            #        sub = line[len(prefix):]
            #        break
            #features[i, 0] = sub

        return features
        
class LocationVectorizer(TransformerMixin, BaseEstimator):
    """Transforms each document locations list into a vector of TF-weights.
    Takes a sequence of locs list and produces a sequence of locations vectors ready for ML.
    """
    def fit(self, X, y=None):
        locFeature = []
        allLocs =[l for l in locs for locs in X]
        locsDic = Counter(allLocs)
        minCnt = len(X)
        updList = [(l,c) for l,c in locsDic.items() if c >= minCnt]
        self.locsDic = dict(updList)
        return self

    def transform(self, docs):
        
        for doc in docs:
            loc_d = Counter(doc)
            s = sum(loc_d.values())
            loc_d = dict([(l,c*1.0/s) for l,c in loc_d])
            l_f = [c for l,c in loc_d if l in self.locsDic]
            locFeature.append(l_f)  
        return np.array(locFeature)

class DateScorer(TransformerMixin, BaseEstimator):
    """Transforms each document dates list into a date score.
    """
    def fit(self, X, y=None):
        self.event_date = None
        allDates =[d for d in dates for dates in X]
        datesDic = Counter(allDates)
        self.event_date = datesDic.most_common(1)
        return self

    def transform(self, docs):
        #get most frequent date in all documents --> event_date
        #for each doc, calculate the date_diff == difference between most frequent date in doc and event date
        
        scores = []
        
        for  docDates in docs:
            docDate = Counter(docDates).most_common(1)
            date_diff= (docDate-self.event_date).days
            scores.append(date_diff)
        return np.array(scores)


#TODO
#NamedEntitiesExtractor --> EventVectorizer
#EventVectorizer (column transformer) will apply TfidfVectorizer on feature0 (topic), extract global list of locations from feature1 and convert each document location list into the global list
# then extract the most common date from feature2 and then convert each document date into a score using dateScoring function based on how far the document date from the most common date


pipeline = Pipeline([
    # Extract the subject & body
    ('NamedEntities', NamedEntitiesExtractor()),

    # Use ColumnTransformer to combine the features from subject and body
    ('EventModel', ColumnTransformer(
        [
            # Pulling features from the post's subject line (first column)
            ('topic', TfidfVectorizer(decode_error='ignore'), 0),

            # Pipeline for standard bag-of-words model for body (second column)
            ('loc', LocationVectorizer() , 1),

            # Pipeline for pulling ad hoc features from post's body
            ('date', DateScorer(), 2),
        ],

        # weight components in ColumnTransformer
        #transformer_weights={
        #    'subject': 0.8,
        #    'body_bow': 0.5,
        #    'body_stats': 1.0,
        #}
    )),

    # Use a SVC classifier on the combined features
    #('svc', LinearSVC(dual=False)),
    ('svc', svm.SVC(kernel='linear', class_weight='balanced',probability=True)),
], verbose=True)

## limit the list of categories to make running this example faster.
#categories = ['alt.atheism', 'talk.religion.misc']
#X_train, y_train = fetch_20newsgroups(random_state=1,
#                                      subset='train',
#                                      categories=categories,
#                                      remove=('footers', 'quotes'),
#                                      return_X_y=True)
#X_test, y_test = fetch_20newsgroups(random_state=1,
#                                    subset='test',
#                                    categories=categories,
#                                    remove=('footers', 'quotes'),
#                                    return_X_y=True)
#Xtrain, Xtest,ytrain,ytest
#pipeline.fit(X_train, y_train)
#y_pred = pipeline.predict(X_test)
#print(classification_report(y_test, y_pred))
##################################################################

# vectorizer = TfidfVectorizer(decode_error='ignore')
# Xtrain_tfidf = vectorizer.fit_transform(Xtrain)
# 
# wclf = svm.SVC(kernel='linear', class_weight='balanced',probability=True)
# wclf.fit(Xtrain_tfidf,ytrain)
# # predict probabilities
# Xtest_tfidf = vectorizer.transform(Xtest)
pipeline.fit(Xtrain, ytrain)

joblib.dump(pipeline, './model.joblib')


# predict class values
#yhat = wclf.predict(Xtest_tfidf)
yhat = pipeline.predict(Xtest)
# calculate scores
model_f1 = f1_score(ytest, yhat,average='weighted')

#y_probs = wclf.predict_proba(Xtest_tfidf)
y_probs = pipeline.predict_proba(Xtest)
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
plt.savefig('model-PR-curve-pipeline.png')

