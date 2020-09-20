import spacy 
import stanza
from collections import Counter
import sys
import re
import articleDateExtractor_L
from newspaper import Article
from datetime import datetime
from dateutil.parser import parse
from collections import defaultdict




def parseDateFrmStr(dateString):
	DEFAULT1 = datetime(1970, 1, 1)
	DEFAULT2 = datetime(1972, 2, 2)
	try:
		dateTimeObj = parse(dateString, default=DEFAULT1)
		if parse(dateString, default=DEFAULT2) == dateTimeObj:
			return dateTimeObj.date()
		else:
			return None
	except:
		return None


def runSpacy():
	nlp = spacy.load('en_core_web_sm') 
	sentence = "Apple is looking at buying Christchurch, New Zeland startup for $1 billion"
	doc = nlp(sentence)   
	for ent in doc.ents: 
		print(ent.text, ent.start_char, ent.end_char, ent.label_) 
		
def extractDateFromMetaHTML(url,ihtml):
	publish_date = None
	try:
		article = Article(url)
		article.download(input_html=ihtml)
		article.parse()
		#publish_date = str(article.publish_date)
		publish_date = article.publish_date
		if publish_date:
			publish_date = publish_date.date()
			print("Using newspaper for %s and output is %s" % (url,str(publish_date)))
	except Exception as e:
		print (e)
	if publish_date == None:#"None":
		#publish_date = str(articleDateExtractor_L.extractArticlePublishedDate(url, html = ihtml))
		publish_date = articleDateExtractor_L.extractArticlePublishedDate(url, html = ihtml)
		print("Using local script for %s and output is %s" % (url,str(publish_date)))
		
	#if publish_date == None:#"None":
		#date_diff = 999999.0
	#	date_object=None
	#else:
		#date_str = publish_date.split(" ")[0]+" 00:00:00"
		#date_object = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
	#	date_object = publish_date
		#date_diff= (date_object-event_date).days
		#return date_diff
	#return date_object
	return publish_date

nlp = None

def loadNLP():
	#For packages with 4 named entity types, supported types include PER (Person), LOC (Location), ORG (Organization) and MISC (Miscellaneous); 
	#for package with 18 named entity types, supported types include PERSON, NORP (Nationalities/religious/political group), FAC (Facility), 
	#ORG (Organization), GPE (Countries/cities/states), LOC (Location), PRODUCT,EVENT, WORK_OF_ART, LAW, LANGUAGE, DATE, TIME, PERCENT, MONEY, QUANTITY, 
	#ORDINAL and CARDINAL
	#LANGUAGE	CODE	PACKAGE	# TYPES
	#Arabic		ar		AQMAR		4
	#English	en		CoNLL03		4		 
	#English	en		OntoNotes	18 ---> ner is here
	#examples:
#	import stanza
	#ex1
# 	nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
# 	doc = nlp("Chris Manning teaches at Stanford University. He lives in the Bay Area.")
# 	print(*[f'entity: {ent.text}\ttype: {ent.type}' for ent in doc.ents], sep='\n')
	#ex2
# 	nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
# 	doc = nlp("Chris Manning teaches at Stanford University. He lives in the Bay Area.")
# 	print(*[f'entity: {ent.text}\ttype: {ent.type}' for sent in doc.sentences for ent in sent.ents], sep='\n')

	#stanza.download('en') # download English model
	if nlp == None:
		nlp = stanza.Pipeline('en',processors='tokenize,ner') # initialize English neural pipeline
	#doc = nlp("Barack Obama was born in christchurch, New Zeland.")
	#print(doc)
	#print(doc.entities)
	#for ent in doc.entities:
		#print( ent.text, ent.type)
	#return nlp

#def extractLocsDates(docs,htmls,urls):
def extractLocsDates_Doc(doc):
	doc = re.sub(r'(\n)+', ' . ', doc)
	locs = []
	dates = []
		
	#for d,h,u in zip(docs,htmls,urls):
	docNLP = nlp(doc)
	#print("got docNLP")
	docEnts = docNLP.entities
	#print("got entities")
	for ent in docEnts:
		if ent.type in ['LOC','GPE']:
			locs.append(ent.text)
		elif ent.type == 'DATE':
			pd = parseDateFrmStr(ent.text)
			if pd:
				dates.append(pd)
	#dobj = extractDateFromMetaHTML(u,h)
	#if dobj:
	#	dates.append(dobj)
	#allLocs.append(locs)
	#allDates.append(dates)
	return locs,dates
	#return allLocs,allDates



# this function returns all locations and dates in a collection of documents. The return list has the locations and dates per document
def extractLocsDates(docs,htmls,urls):
	docs = [re.sub(r'(\n)+', ' . ', doc) for doc in docs]
	locs = []
	dates = []
	#For packages with 4 named entity types, supported types include PER (Person), LOC (Location), ORG (Organization) and MISC (Miscellaneous); 
	#for package with 18 named entity types, supported types include PERSON, NORP (Nationalities/religious/political group), FAC (Facility), 
	#ORG (Organization), GPE (Countries/cities/states), LOC (Location), PRODUCT,EVENT, WORK_OF_ART, LAW, LANGUAGE, DATE, TIME, PERCENT, MONEY, QUANTITY, 
	#ORDINAL and CARDINAL
	#LANGUAGE	CODE	PACKAGE	# TYPES
	#Arabic		ar		AQMAR		4
	#English	en		CoNLL03		4		 
	#English	en		OntoNotes	18 ---> ner is here
	#examples:
#	import stanza
	#ex1
# 	nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
# 	doc = nlp("Chris Manning teaches at Stanford University. He lives in the Bay Area.")
# 	print(*[f'entity: {ent.text}\ttype: {ent.type}' for ent in doc.ents], sep='\n')
	#ex2
# 	nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
# 	doc = nlp("Chris Manning teaches at Stanford University. He lives in the Bay Area.")
# 	print(*[f'entity: {ent.text}\ttype: {ent.type}' for sent in doc.sentences for ent in sent.ents], sep='\n')

	#stanza.download('en') # download English model
	nlp = stanza.Pipeline('en',processors='tokenize,ner') # initialize English neural pipeline
	#doc = nlp("Barack Obama was born in christchurch, New Zeland.")
	#print(doc)
	#print(doc.entities)
	#for ent in doc.entities:
		#print( ent.text, ent.type)
	allLocs = []
	allDates = []
	
	
		
	for d,h,u in zip(docs,htmls,urls):
		locs = []
		dates=[]
		docNLP = nlp(d)
		#print("got docNLP")
		docEnts = docNLP.entities
		#print("got entities")
		for ent in docEnts:
			if ent.type in ['LOC','GPE']:
				locs.append(ent.text)
			elif ent.type == 'DATE':
				pd = parseDateFrmStr(ent.text)
				if pd:
					dates.append(pd)
		dobj = extractDateFromMetaHTML(u,h)
		if dobj:
			dates.append(dobj)
		allLocs.append(locs)
		allDates.append(dates)
	#return locs,dates
	return allLocs,allDates
		
def runStanza(docs,htmls,urls):
	
	allLocs,allDates = extractLocsDate(docs,htmls,urls)	
	all_locs = [l for locs in allLocs for l in locs]
	all_dates = [d for dates in allDates for d in dates]
		
	locsDic = Counter(all_locs)
	datesDic = Counter(all_dates)
	#actualDatesDic = defaultdict(int)
	#for d,c in datesDic.items():
	#	dp = parseDateFrmStr(d)
	#	if dp:
	#		#actualDatesDic.append(dp)
	#		actualDatesDic[dp] += c

	
	#htmlDates = []
	#for url,html in zip(urls,htmls):
	#	d = extractDateFromMetaHTML(url,html)
	#	htmlDates.append(d)
	#	#actualDatesDic.append(d)
	#	actualDatesDic[d] +=1
	#htmlDatesDic = Counter(htmlDates)
	##actualDatesDic = Counter(actualDatesDic)
		
	#return locsDic,datesDic,htmlDatesDic,actualDatesDic
	return locsDic,datesDic

# txtPath = sys.argv[1]
# urls = ['https://www.cnn.com/2019/03/19/asia/christchurch-attack-intl/index.html','https://www.bbc.com/news/av/world-asia-47579433/new-zealand-pm-jacinda-ardern-this-can-only-be-described-as-a-terrorist-attack']
# htmlPath = sys.argv[2]
# doc1 = open(txtPath+'1.txt').read()
# doc2 = open(txtPath+'2.txt').read()
# docs = [doc1, doc2]
# html1 = open(htmlPath+'1.html').read()
# html2 = open(htmlPath+'2.html').read()
# htmls=[html1,html2]
# #locs,dates,htmlDates,actualDatesDic = runStanza(docs,htmls,urls)
# locs,dates = runStanza(docs,htmls,urls)
# print(locs)
# print(dates)
# #print(htmlDates)
# #print(actualDatesDic)