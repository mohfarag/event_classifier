import requests, requests_cache
import sys, os
from collections import defaultdict 
from bs4 import BeautifulSoup, Comment

from boilerpipe.extract import Extractor

requests_cache.install_cache('Webpages', backend='sqlite')
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0'}



def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head']:
        return False
    return True    
    
#def getTxt(htmlPage):
def extractTextFromHTML(htmlPage):
# 	webpagesTxt = ""
# 	if type(htmlPages) != type([]):
# 		htmlPages = [htmlPages]
# 	for htmlPage in htmlPages:
	title = ""
	text = ""
	wtext = {}    
	soup = BeautifulSoup(htmlPage,"html.parser")

	if soup.title:
		if soup.title.string:
			title = soup.title.string

	comments = soup.findAll(text=lambda text:isinstance(text,Comment))
	[comment.extract() for comment in comments]


	text_nodes = soup.findAll(text=True)
	#text_nodes_noLinks = soup.findAll(text=True)
	visible_text = filter(visible, text_nodes)
	text = "\n".join(visible_text)
	#textSents = getSentences(text)
	#text = "\n".join(textSents)
	text = title + '\n' + text
	#wtext = {"text":title + u' '+ text,"title":title}
	wtext = {"text": text,"title":title}
	
	#webpagesTxt.append(wtext)
	#return webpagesTxt
	return wtext
	
def extractTextFromHTML_Boilerpipe(fileHTML):
	extractor = Extractor(extractor='ArticleExtractor',html = fileHTML)
	extracted_text = extractor.getText()
	return {'text':extracted_text}
	
def extractTxt(folder):
	txts = []
	fileNames = []
	for entry in os.listdir(folder):
		if (os.path.isfile(os.path.join(folder, entry))) and  (not entry.startswith(".")):
			print(entry)	
			with open(os.path.join(folder, entry),errors='ignore') as f:				
				fileHTML = f.read()
				t = extractTextFromHTML(fileHTML)
				txts.append(t['text'])
				fileNames.append(entry)
	return txts,fileNames
	

def saveTxtFiles(txts,fileNames,folder):
	for t,f in zip(txts,fileNames):
		if t:
			tf = f.split(".html")[0]+".txt"
			with open(os.path.join(folder, tf),"w") as fi:
				fi.write(t)

if __name__ == "__main__":
	infolder = sys.argv[1]
	outFolder = sys.argv[2]
			
	txts,fileNames = extractTxt(infolder)
	saveTxtFiles(txts,fileNames,outFolder)
	

