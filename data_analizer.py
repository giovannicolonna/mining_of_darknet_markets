import csv
import nltk
import logging
import numpy as np
import matplotlib.pyplot as plt


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import MDS,TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
from gensim import corpora,models
from gensim.similarities import SparseMatrixSimilarity
from gensim.matutils import Scipy2Corpus,Sparse2Corpus,corpus2csc
from random import randint


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO


### STEMMING AND TF-IDF BUILDING PHASE
print "Data analysis:\n"
print "Tokenizing and text cleaning (low freq terms eliminations)..."
datasetMap = {}   # map keeping: key: 1, title   -   value: [array of stems]
stemmer = PorterStemmer()


exitFile = open('dataset_big','r')

id_insertion = 0
frequency = defaultdict(int)
for line in exitFile:
    splittedLine = line.split(',')
    #tokenize
    tokens = nltk.word_tokenize(splittedLine[3])
    #stem
    stemmedString = []
    for item in tokens:
        word = stemmer.stem(item)
        stemmedString.append(word)
        frequency[word] += 1
    datasetMap[str(id_insertion) + ': ' + splittedLine[2]] = stemmedString
    id_insertion += 1
# [0] is the name, [1] is the price in BTC, [2] title, [3] description, [4] shipping from


# remove words appearing few times
newDatasetMap = {}
emptycounter = 0
num_terms = 0
total_words_array = []
for key in datasetMap:
    newDescription = []
    for word in datasetMap[key]:
        # datasetMap[key] is an array of words
        if frequency[word] > 9:  #get rid of typos
            if len(word) > 2: #avoid words made by one or two letters (e.g. "g" or "mg")
                num_terms += 1
                newDescription.append(word)
                total_words_array.append(word)
        else:
            print "Rare token: " + word
    if len(newDescription) > 1:
        newDatasetMap[key] = newDescription
    else:
        emptycounter += 1
print "Number of discarded descriptions: " + str(emptycounter) + " on a total of " + str(len(datasetMap.keys()))
exitFile.close()

dictionary = corpora.Dictionary(newDatasetMap.values())
dictionary.save('dataset.dict') #temporary dictionary
#
# #raw corpus
raw_corpus = [dictionary.doc2bow(t) for t in newDatasetMap.values()]
corpora.MmCorpus.serialize('dataset.mm', raw_corpus) #store to disk
#
# #corpus loading
dictionary = corpora.Dictionary.load('dataset.dict')
corpus = corpora.MmCorpus('dataset.mm')

# # Tfidf calculation with gensim
# print "Computing tf-idf with gensim..."  #NOT SO ACCURATE CLUSTERING - comment these 4 lines to exclude gensim
# tfidf = models.TfidfModel(corpus)
# corpus_tfidf = tfidf[corpus]
# tfidf_matrix = corpus2csc(corpus_tfidf).transpose()

#Tfidf calculation with scikit
print "Computing tf-idf with scikit..."  #BETTER CLUSTERING, comment these lines to exclude scikit
tfidf_vectorizer = TfidfVectorizer(stop_words='english',use_idf=True,sublinear_tf=True,max_df=0.6)
docs = []
for doc_array in newDatasetMap.values(): #convert datasetMap in a suitable format for scikitlearn vectorizer
    strt = ''
    for elem in doc_array:
        strt = strt + elem + " "
    strt = strt.strip()
    docs.append(strt)
tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
# project to 2 dimensions LSI for visualization, LSI needs gensim corpus, comment following line to exclude LSI
#corpus_tfidf = Sparse2Corpus(tfidf_matrix,documents_columns=False) #this line needed only with scikit tfidf matrix and LSI

# use of TruncatedSVD for dimension reduction
print "TruncatedSVD algorithm to perform 2-dimension reduction...."
#it uses scikit tfidf matrix
svd = TruncatedSVD(n_components=2)
normalizer = Normalizer(copy=False) # normalization evidences a shaped plot
lsa = make_pipeline(svd, normalizer)
X_reduced = lsa.fit_transform(tfidf_matrix)
# writing out coordinates to file
fcoords = open("coords.csv",'wb')
for vector in X_reduced:
    if len(vector) != 2:
        continue
    #fcoords.write("%6.4f\t%6.4f\n" % (vector[0], vector[1]))
    string = str(vector[0])+'\t'+str(vector[1])+'\n'
    fcoords.write(string)
fcoords.close()

# # use of LSI for dimension reduction - needs corpus_tfidf obtained by Sparse2Corpus if tfidf are calculated with scikit
# print "LSI model building for visualization..."
# lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
# # write out coordinates to file
# fcoords = open("coords.csv", 'wb')
# for vector in lsi[corpus]:
#     if len(vector) != 2:
#         continue
#     fcoords.write("%6.4f\t%6.4f\n" % (vector[0][1], vector[1][1]))
# fcoords.close()

#Finding optimal number of clusters, comment/uncomment the type of X to be clustered
print "Finding optimal clustering..."
MAX_K = 10
#X = np.loadtxt('coords.csv', delimiter="\t") #KMEANS ARE PERFORMED ON COORDINATES OF POINTS OBTAINED BY DIMENS REDUCTION
X = tfidf_matrix #KMEANS PERFORMED ON TFIDF MATRIX
ks = range(1, MAX_K + 1)

inertias = np.zeros(MAX_K)
diff = np.zeros(MAX_K)
diff2 = np.zeros(MAX_K)
diff3 = np.zeros(MAX_K)
for k in ks:
    kmeans = KMeans(k,verbose=2,init='k-means++').fit(X)
    inertias[k - 1] = kmeans.inertia_
    # first difference
    if k > 1:
        diff[k - 1] = inertias[k - 1] - inertias[k - 2]
    # second difference
    if k > 2:
        diff2[k - 1] = diff[k - 1] - diff[k - 2]
    # third difference
    if k > 3:
        diff3[k - 1] = diff2[k - 1] - diff2[k - 2]

elbow = np.argmin(diff3[3:]) + 3
plt.plot(ks, inertias, "b*-")
plt.plot(ks[elbow], inertias[elbow], marker='o', markersize=12,
         markeredgewidth=2, markeredgecolor='r', markerfacecolor=None)
plt.ylabel("Inertia")
plt.xlabel("K")
plt.show()

# KMeans clustering - comment/uncomment the type of X to be clustered
print "KMeans clustering, number of clusters tuned according to the elbow point..."
print str(elbow+1) + " clusters are optimal."
#X = np.loadtxt('coords.csv',delimiter="\t")   #KMEANS PERFORMED ON COORDINATES OF POINTS OBTAINED BY DIMENSION REDUCTION
X = tfidf_matrix      #KMEANS PERFORMED ON TF-IDF MATRIX (better results)
kmeansDef = 0
if elbow < 10:
    kmeansDef = KMeans(elbow+1,verbose=2,init='k-means++').fit(X)
else:
    kmeansDef = KMeans(10,verbose=2,init='k-means++').fit(X)
y = kmeansDef.labels_


#Plotting (loads coordinates obtained by either LSI by gensim or TruncatedSVD by scikit)
print "Plotting visualization of clustering..."
Xp = np.loadtxt('coords.csv',delimiter="\t")
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
for i in range(Xp.shape[0]):
    plt.scatter(Xp[i][0], Xp[i][1], c=colors[y[i]], s=10)
plt.show()


#Clusters writing to file
clusters = defaultdict(list)
print "Writing clustering output..."
k = 0
cluster_out = open('clustering_output',"wb")
for label in y:
    clusters[label].append(newDatasetMap.keys()[k])  # appends to each cluster[label] the titles of the elements
    k += 1

for clust in clusters:
    cluster_out.write("\n**************\n")
    cluster_out.write("Cluster "+str(clust)+":\n")
    # first 30
    top = 30
    for name in clusters[clust]:
        cluster_out.write(name+" ")
        if top == 0: break
        top = top - 1
cluster_out.close()


for cluster_id in clusters:
    # cluster_id contains names, which are keys of newDatasetMap
    clusterWordFrequency = defaultdict(int)
    for titles in clusters[cluster_id]:
        # each title is a key of newDatasetMap
        for word in newDatasetMap[titles]:
            clusterWordFrequency[word] += 1
    #end of cluster, print 30 most common words of the cluster:
    i = 0
    print "30 most frequent word in cluster " + str(cluster_id)+": \n"
    for word in sorted(clusterWordFrequency, key=clusterWordFrequency.get,reverse=True):
        print word,clusterWordFrequency[word]
        i += 1
        if i == 30:
            break
    print


# if SVD is used, following block prints most relevant words of cluster
#original_space_centroids = svd.inverse_transform(kmeansDef.cluster_centers_) # if kmeans is performed on SVD points, uncomment
#order_centroids = original_space_centroids.argsort()[:, ::-1]  # if kmeans is performed on SVD points, uncomment
order_centroids = kmeansDef.cluster_centers_.argsort()[:, ::-1] # if kmeans is performed on tfidf matrix, uncomment

terms = tfidf_vectorizer.get_feature_names()
for i in range(len(clusters)):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
    print()

# # if is used LSI, we use LDA topics
# dictionary = corpora.Dictionary.load("dataset.dict")
# corpus = corpora.MmCorpus("dataset.mm")
# # Project to LDA space
# lda = models.LdaModel(corpus, id2word=dictionary, num_topics=len(clusters))
# lda.print_topics(len(clusters))
