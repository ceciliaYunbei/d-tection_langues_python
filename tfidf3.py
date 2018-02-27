
# coding: utf-8

# In[111]:

#!/usr/bin/python/


import codecs, unicodedata, re, glob, os, random, string
from sklearn.feature_extraction.text import TfidfVectorizer


# In[112]:

def systemeEcriture(texte):
    """
    la fonction prend un texte en entrée
    elle rend le système d'écriture le plus couramment rencontré parmi les lettres présentes dans le texte, par exemple "LATIN" ou "HANGUL"
    """
    freqsyst = {}
    for c in texte:
        if unicodedata.category(c)[0] == "L":
            systeme = unicodedata.name(c).split()[0]
            freqsyst[systeme] = freqsyst.get(systeme, 0) + 1
    return max(freqsyst, key = freqsyst.get)

for nomFichier in os.listdir("textesLangues"):
    if re.match("\.", nomFichier) is None:
        print("La catégorie de lettres la plus fréquente du fichier " + nomFichier + " est " + systemeEcriture(open("textesLangues/"+nomFichier).read()))


# In[113]:

def distanceMots(texte):
    """
    la fonction prend un texte comme entrée
    et elle retourne un dictionnaire avec les fréquences relatives de chaque mot
    """
    motFreq={}
    mots=texte.split()
    for mot in mots:
        motFreq[mot]=motFreq.get(mot,0)+1/len(mots)
    return motFreq

def texteFreqNgram(texte, n):
    """
    la fonction prend un texte comme entrée
    et elle retourne un dictionnaire avec les fréquences relatives de chaque n-gramme pour n donné
    """
    ngramsfreq={}
    ngs=ngrams(texte, n)
    for ng in ngs:
        ngramsfreq[ng]=ngramsfreq.get(ng,0)+1/len(ngs)
    return ngramsfreq
    
def ngrams(texte, n):
    """
    me donne une liste des ngrams du texte
    """
    return [ texte[i:i+n] for i in range(len(texte)-n+1) ]

def distance1gram(texte):
    return texteFreqNgram(texte, 1)

def distance2gram(texte):
    return texteFreqNgram(texte, 2)

def distance3gram(texte):
    return texteFreqNgram(texte, 3)

def distance4gram(texte):
    return texteFreqNgram(texte, 4)

def distance5gram(texte):
    return texteFreqNgram(texte, 5)

def nb10mots(texte):
	freqMots=distanceMots(texte)
	freq10Mots={}
	for mot in sorted(freqMots, key=freqMots.get)[-10:]:
		freq10Mots[mot]=1/10
	return freq10Mots
	
def distanceFreq(freq1, freq2):
    """
    prend deux dictionnaires qui renvoient des mots à leur fréquence relative
    rend la distance entre ces fréquences    
    """
    return sum([abs(freq1.get(mot, 0) - freq2.get(mot, 0)) for mot in set(freq1) | set(freq2) ])/2
	


# In[114]:

distanceAlgos=[nb10mots, distanceMots, distance1gram, distance2gram, distance3gram, distance4gram, distance5gram]


# In[115]:

def lePlusProche(boutDeTexte, alg, dicoPourAlg):
	freqDic=alg(boutDeTexte)
	distanceParLangue={}
	for code in dicoPourAlg:
		distanceParLangue[code] = distanceFreq(freqDic,dicoPourAlg[code])
	return min(distanceParLangue, key=distanceParLangue.get)


# In[116]:

def creerDicoTrain(distanceAlgos, nomDossierTrain):
	"""
	algos: une liste de fonctions données (chaque fonction calcule un dict de fréquence)
	nomDossierTrain: le dossier contenant toutes les données d'entrainement
	sortie : un dictionnaire complexe précalculant toute l'information contenue dans le dossier nomDossierTrain
	{ nom de l'algo --> { langue --> { token --> fréquence relative } } }
	
	"""
	dicoTrain={}
	for alg in distanceAlgos:
		dicoAlgo={}
		for nomFichier in os.listdir(nomDossierTrain):
			if nomFichier[0]==".": continue
			# obtenir le code de langue à partir du nom de fichier
			code=nomFichier[:2]
			freqActuelles = alg(open(os.path.join(nomDossierTrain,nomFichier)).read())
			dicoAlgo[code]=freqActuelles
		dicoTrain[alg.__name__]=dicoAlgo
	return dicoTrain


# In[117]:

def decoupage(string, longeur = 20):
    lonTotal = len(string)
    nbDecoupe = int(lonTotal/longeur)
    decoupes = []
    for i in range(nbDecoupe):
        decoupes.append(string[i*longeur:(i+1)*longeur])
    decoupes.append(string[nbDecoupe*longeur:])
    return decoupes


# In[118]:

def tfidf():
    trainingFiles = {}
    for files in os.listdir("textesLangues"):
        if re.match("\.", files) or files.endswith('~') : continue
        if files[:2] not in trainingFiles:
            trainingFiles[files[:2]] = open("textesLangues/"+ files).read()
        else:
            trainingFiles[files[:2]] += open("textesLangues/"+ files).read()
    for k,v in trainingFiles.items():
        trainingFiles[k] = decoupage(re.sub('[\s+]', '', v), 500)
    trainingTexts = {k: zip(v, [k]*len(v)) for k,v in trainingFiles.items()}
    tmp = []
    training = []
    for lang in trainingTexts.keys():
        training.extend(list(trainingTexts.get(lang)))
    vectorizer = TfidfVectorizer(analyzer='char', decode_error='ignore')
    X = [pair[0].replace(' ', '') for pair in training]
    y = [pair[1] for pair in training]
    transformer = vectorizer.fit(X)
    X_train = transformer.transform(X)
    from sklearn.linear_model import SGDClassifier 
    cls = SGDClassifier()
    cls.fit(X_train, y)
    return cls, transformer


# In[119]:

Algos = [tfidf]
Algos += distanceAlgos


# In[120]:

def longueurQualite(nbCar, maxiTest=3):
    dicoTrain = creerDicoTrain(distanceAlgos, "textesLangues")
    print("Tout est lu et précalculé. On regarde les fichiers test...")
    dictAlgoReussite={} # renverra nom d'algo -> float de réussite (entre 0 et 1)
    for alg in Algos:
        print("_________Algo", alg.__name__)
        resultats = []
        cls, transformer = tfidf()
        for fichiertest in os.listdir("tests/"):
            if fichiertest.startswith('.'): continue
            print("____",fichiertest)
            codereel=fichiertest[:2]
            if alg in distanceAlgos:
                texte=open("tests/"+fichiertest).read() 
            else:
                texte=open("tests/"+fichiertest).read().replace(' ', '').translate(string.punctuation)
           
            if len(texte)//nbCar>0:
                resultatsFichier=[]
                for i in random.sample(range(len(texte)//nbCar),min(maxiTest,len(texte)//nbCar)):
                    boutDeTexte=texte[i*nbCar+1:(i+1)*nbCar]
                    if alg in distanceAlgos:
                        codeobtenu = lePlusProche(boutDeTexte,alg,dicoTrain[alg.__name__])
                        resultatsFichier+=[codereel==codeobtenu]
                    else:
                        codeobtenu = cls.predict(transformer.transform([boutDeTexte]))
                        resultatsFichier+=[codereel==codeobtenu[0]]
                    sommeMoyenne=sum(resultatsFichier)/len(resultatsFichier)
                    print (sommeMoyenne)
                    resultats+=[sommeMoyenne]
            if resultats:    
                dictAlgoReussite[alg.__name__] = sum(resultats)/len(resultats)
    return dictAlgoReussite


# In[121]:

def testSystematique(segLongueurs=[5, 10, 20, 50, 100, 500, 1000, 5000, 10000]):
	loAlgoQual={}
	for lo in segLongueurs:
		print("_____________________",lo)
		loAlgoQual[lo]=longueurQualite(lo)
	with open("loAlgoQual.tsv","w") as fichierSortie:
		fichierSortie.write("\t".join(["Algos","Longueur","Taux"])+"\n")
		for lo in segLongueurs:
			for nomAlg in loAlgoQual[lo]:
				fichierSortie.write("\t".join([nomAlg,str(lo),str(loAlgoQual[lo][nomAlg])])+"\n")
			
#testSystematique([5, 10000])         
points = [5, 10, 20, 50, 100] + [i*500 for i in range(1,21)]
testSystematique(points)  


# In[ ]:



