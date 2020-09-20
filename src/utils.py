import requests, requests_cache
import sys
from collections import defaultdict 

requests_cache.install_cache('Webpages', backend='sqlite')
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0'}



# def getWebpageText(URLs = []):
#     webpagesText = []
#     r = {}
#     errorsDownload = 0
#     errorsText = 0
#     text = {}
#     if type(URLs) != type([]):
#         URLs = [URLs]
#     for url in URLs:
#         try:
#             r = requests.get(url.strip(),timeout=10,verify=True,headers=headers)     
#         except Exception as e:
#         	print("couldn't download: " + url + ", because " + e)
#         	print(sys.exc_info())
#             errors+=1     
#         if r and r.status_code == requests.codes.ok:
#             #ct = r.headers['Content-Type']
#             ct = r.headers.get('Content-Type','')
#             if ct.find('text/html')!= -1:
#                 page = r.content
#                 text = extractTextFromHTML(page)
#                 if text:
#                     text['html']= page
#                 else:
#                     print 'No Text to be extracted from: ', url
#                     errorsText +=1
#             else:
#                 #text = {}
#                 print 'Content-Type is not text/html', ct," - ", url 
#                 errorsText +=1
#         else:
#             print 'Could not downlaod:' + url + ", return status code: " + r.status_code
# #         except Exception as e:
# #             raise e
# #             print sys.exc_info()
#             #text = ""
#             #text = {}
#         webpagesText.append(text)
#     return webpagesText
   
def downloadWebpages(URLs = []):
	webpagesHTML = []
    
	errors = 0
    #errorsText = 0
    #text = {}
	pageHTML = {}
	if type(URLs) != type([]):
		URLs = [URLs]
	for url in URLs:
		r = {}
		try:
			r = requests.get(url.strip(),timeout=10,verify=True,headers=headers)     
		except Exception as e:
			print("couldn't download: " + url + ", because " + e)
			print(sys.exc_info())
			errors+=1     
		if r and r.status_code == requests.codes.ok:
            #ct = r.headers['Content-Type']
			ct = r.headers.get('Content-Type','')
			if ct.find('text/html')!= -1:
				pageHTML = r.content                
			else:
				print ('Content-Type is not text/html', ct," - ", url )
				errors +=1
		else:
			print ('Could not downlaod:' + url + ", return status code: " + r.status_code)
		
		webpagesHTML.append(pageHTML)
	print("No. of pages downloaded and extracted HTML from are: " + str(len(URLs)-errors))
	return webpagesHTML   
   
def countDomains(URLs):
	domainsList = map(getDomain, URLs)
	domainsDic = defaultdict(int)
	for d in domainsList:
		domainsDic[d] = domainsDic[d]+1
	
	return domainsDic


def shuffleURLsFromDomainsQueues(domainQueues):
	shuffledURLs = []
	while(True):
		empty = True
		for d in domainQueues:
			if domainQueues[d]:
				shuffledURLs.append(domainQueues[d].pop())
				empty = False
		if empty:
			break
	return shuffledURLs
			

def createDomainsQueues(urlsFile):
	with open(urlsFile) as f:
		urls = f.readlines()
	domainsQueues = defaultdict(list)
	domainsList = list(map(getDomain, urls))
	#for d,u in zip(domainsList,urls):
	for i in range(len(domainsList)):
		domainsQueues[domainsList[i]].append(urls[i].strip())
	return domainsQueues

def getDomain(url):
    domain = ""
    ind = url.find("//")
    if ind != -1 :
        domain = url[ind+2:]
        ind = domain.find("/")
        domain = domain[:ind]
    return domain   
    
# def extractTextFromHTML(page):
#     
#     #try:
#     
#     text = ''
#     title = ''
#     wtext = {}
#     if page:
#         text,title = getTxt(page)
# 		if text.strip():
# 			wtext = {"text":title + u' '+ text,"title":title}
#         #wtext = {'text':text}
#     #else:
#         #print 'No Text in page'#, page
#         #wtext = {}
#     return wtext

def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head']:
        return False
    return True    
    
#def getTxt(htmlPage):
def extractTextFromHTML(htmlPages):
	webpagesTxt = []
	if type(htmlPages) != type([]):
		htmlPages = [htmlPages]
	for htmlPage in htmlPages:
		title = ""
		text = ""
		wtext = {}    
		soup = BeautifulSoup(htmlPage)

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
		#return text,title
		webpagesTxt.append(wtext)
	return webpagesTxt

