import logging
import random
from collections import defaultdict,OrderedDict,Counter

import lda
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas
import pycountry
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from simplenb import naivebayes
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import TSNE
from wordcloud import WordCloud,STOPWORDS

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO
print ("********** DATASET ANALYSIS **********\n")
forGephi = []

def display_topics(model, feature_names, no_top_words):
    string = ''
    for topic_idx, topic in enumerate(model.components_):
        string = string + ("Cluster %d:" % (topic_idx)) + '\n'
        string = string + " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])
        string = string + '\n'
    string = string+'\n'
    return string
# Read and load dataset in memory

print ("Loading dataset_big file in memory...\n")
datasetMap = defaultdict(str)
datasetGeo = defaultdict(str)

with open('dataset_big', 'r') as dataset_file:
    unique_id = 0
    for line in dataset_file:
        splittedLine = line.split(',')
        # Line is made in this way:


        # splittedLine[1] = price in BTC (when present)
        # splittedLine[2] = title of the listing
        # splittedLine[3] = description
        # splittedLine[4] = shipping from
        key = str(unique_id) + " : " + splittedLine[2]
        value = splittedLine[3]
        if len(splittedLine) > 5: # commas can slip into last cell
            geo_value = splittedLine[len(splittedLine) - 1]
        else:
            geo_value = splittedLine[4]
        if value:  # no empty descriptions
            datasetMap[key] = value
            unique_id += 1
        else:
            continue
        if geo_value:
            geo_value.rstrip()
            if geo_value != 'shipping from:':
                if not any(char.isdigit() for char in geo_value):
                    datasetGeo[key] = geo_value # datasetGeo has the same key of datasetMap


print ("Dataset has been load in memory.\n")


# Text cleaning: stemming, common words removal, typos removal
print ("---STEP ONE - CLUSTERING OF LISTING DESCRIPTIONS----\n")
print ("Text cleaning phase...\n")

NUMBER_OF_CLUSTERS = 6
def text_cleaning_descriptions():
    stemmer = PorterStemmer()
    stop = set(stopwords.words('english'))
    stop.add('get')  # Add some ad-hoc stopwords often appearing in listings
    stop.add('use')
    stop.add('aaa')
    stop.add('good')
    stop.add('best')
    stop.add('custom')
    stop.add('list')
    stop.add('free')
    stop.add('send')
    stop.add('ship')
    stop.add('onion')
    stop.add('feedback')
    stop.add('qualiti')
    stop.add('quality')
    stop.add('grams')
    stop.add('address')
    stop.add('order')
    stop.add('pleas')
    stop.add('price')
    stop.add('product')
    stop.add('check')
    stop.add('discuss')
    stop.add('name')
    stop.add('shipping')
    stop.add('one')
    stop.add('track')
    stop.add('day')
    stop.add('time')
    stop.add('packag')
    frequency = defaultdict(int)
    for key in datasetMap:
        currentDescription = datasetMap[key]
        # Performing cleaning on currentDescription
        # Tokenizing
        tokens = nltk.word_tokenize(currentDescription)
        tokens_nostop = []
        for token in tokens:
            if token not in stop:
                tokens_nostop.append(token)
        # Stemming
        stems = []
        for token in tokens_nostop:
            word = stemmer.stem(token)
            if word not in stop:
                stems.append(word)
            else:
                continue
            frequency[word] += 1
        datasetMap[key] = stems
    # Now discard unique tokens or monograms or bigrams and typos (too rare words)
    for key in datasetMap:
        description = datasetMap[key]
        newDescription = []
        for word in description:
            if len(word) > 2:  # A word must be at least 3 letters long
                if frequency[word] > 5:  # A word must appear at least 5 times
                    newDescription.append(word)

                # else:
                    # print "Rare token found: " + word
            else:
                frequency[word] = 0
                continue
        datasetMap[key] = newDescription
    print "Text cleaning completed.\n"
text_cleaning_descriptions()


# All'interno dei metodi di clustering possiamo scegliere in fase di wordcloud se usare il clustering LDA o KMeans


def clustering_descriptions():
    # Tf-Idf building
    print "Computing tf-idf with scikit-learn...\n"

    # datasetMap.keys() is the array with all the listings titles
    # datasetMap.values() is the array with all the descriptions, each description is an array of words
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, use_idf=True)
    # Maximum document frequency is 50%, minimum in 2 documents
    # We need to redefine descriptions, a description can't be an array of words but a single string
    docs = []
    for descriptionArray in datasetMap.values():
        currentString = ''
        for word in descriptionArray:
            currentString = currentString + word + " "
        currentString.strip()
        docs.append(currentString)
    # docs is now an array of descriptions, each description is a single string with spaces
    X = vectorizer.fit_transform(docs)
    featureNames = vectorizer.get_feature_names()
    print "Tf-idf matrix generated:"
    print("n_samples: %d, n_features: %d" % X.shape)
    print

    # Clustering with K-Means

    print "Clustering with K-Means...\n"

    # print "Searching for optimal number of clusters through the elbow point method...\n"
    # # This method is very long, comment this block and set a cluster value to skip
    # MAX_K = 25
    # ks = range(1, MAX_K + 1)
    # inertias = np.zeros(MAX_K)
    # diff = np.zeros(MAX_K)
    # diff2 = np.zeros(MAX_K)
    # diff3 = np.zeros(MAX_K)
    # for k in ks:
    #     print "---Test with " + str(k) + " clusters."
    #     kmeans = KMeans(n_clusters=k, verbose=0, init='k-means++').fit(X)
    #     inertias[k - 1] = kmeans.inertia_
    #     # first difference
    #     if k > 1:
    #         diff[k - 1] = inertias[k - 1] - inertias[k - 2]
    #     # second difference
    #     if k > 2:
    #         diff2[k - 1] = diff[k - 1] - diff[k - 2]
    #     # third difference
    #     if k > 3:
    #         diff3[k - 1] = diff2[k - 1] - diff2[k - 2]
    # elbow = np.argmin(diff3[3:]) + 3
    #
    # # Uncomment following lines (plt.) to show plot of elbow point method
    # print "Plotting results of elbow point method...\n"
    # plt.plot(ks, inertias, "b*-")
    # plt.plot(ks[elbow], inertias[elbow], marker='o', markersize=12,
    #          markeredgewidth=2, markeredgecolor='r', markerfacecolor=None)
    # plt.ylabel("Inertia")
    # plt.xlabel("K")
    # plt.show()

    # Computing K-Means with optimal number of clusters

    optimal_n_clusters = NUMBER_OF_CLUSTERS # Set this value by hand if skipped elbow method
    print "Optimal value for number of clusters: " + str(optimal_n_clusters) + ' clusters\n'
    print "Performing K-Means clustering with " + str(optimal_n_clusters) + " clusters...\n"
    kmeans = KMeans(n_clusters=optimal_n_clusters, verbose=2, init='k-means++').fit(X)
    y = kmeans.labels_
    print "K-Means clustering completed.\n"

    print "Clustering with LDA...\n"
    tf_vectorizer = CountVectorizer(max_df=0.5, min_df=2, stop_words='english')
    X_tf = tf_vectorizer.fit_transform(docs)
    tf_feature_names = tf_vectorizer.get_feature_names()
    ldac = lda.LDA(n_topics=optimal_n_clusters, n_iter=1200,
                   random_state=0)
    ldac.fit(X_tf)
    print display_topics(ldac,tf_feature_names,30)

    doc_topic = ldac.doc_topic_
    titles = datasetMap.keys()
    LDALabels = []
    for i in range(len(titles)):
        #print("{} (top topic: {})".format(titles[i], doc_topic[i].argmax()))
        LDALabels.append(doc_topic[i].argmax())  # contains LDA clustering labels
    print


    # Clustering results saving to file
    # datasetMap.keys() is the array of the titles, we will write those to file
    # in order to write the descriptions, array is docs (simple array) docs[0] is the first description

    print "Writing clustering results to file: clustering_output (first 30 elements per cluster)...\n"
    clusteringMap = defaultdict(list)
    clusteringLDAMap = defaultdict(list)
    titles = datasetMap.keys()
    for i in range(len(titles)):
        clusteringMap[y[i]].append(titles[i])   #y[i] for kmeans, LDALabels[i] for LDA
        clusteringLDAMap[LDALabels[i]].append(titles[i])

    out_file = open('clustering_output', 'wb')
    # Only writing first 50 titles per cluster
    out_file.write("Clustering of descriptions.\n\n\n")
    out_file.write("**** KMeans Clustering results: ****\n\nFor each cluster, 50 titles of insertions are reported.\n\n")
    for clustername in clusteringMap.keys():
        limit = 0
        out_file.write("Cluster: " + str(clustername)+'\n')
        for elem in clusteringMap[clustername]:
            if limit == 49:
                break
            out_file.write(str(elem)+"||")
            limit += 1
        out_file.write('\n')
        # Cluster top terms of kmeans

    out_file.write("\nTop-terms per cluster...\n")
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(optimal_n_clusters):
        out_file.write("Cluster %d:\n" % i)
        for ind in order_centroids[i, :10]:
            out_file.write(' %s\n' % terms[ind])
        out_file.write('\n')

    out_file.write("\n\n\n**** LDA Clustering results ****\n\n")
    for clustername in clusteringLDAMap.keys():
        limit = 0
        out_file.write("Cluster: " + str(clustername)+'\n')
        for elem in clusteringLDAMap[clustername]:
            if limit == 49:
                break
            out_file.write(str(elem)+"||")
            limit += 1
        out_file.write('\n')
    out_file.write("\nTop-terms per cluster...\n")
    out_file.write(display_topics(ldac,tf_feature_names,30))
    out_file.close()

    #costruzione counter per wordclouds

    print "Wordcloud generation...\n"
    # IMPORTANTE -  PER GENERARE WORDCLOUD DEI CLUSTER LDA, SOSTITUIRE CLUSTERINGMAP CON CLUSTERING LDA MAP
    desiredClusteringMap = clusteringLDAMap  # sostituibile con clusteringLDAMap
    for clustername in desiredClusteringMap.keys():
        insertions_of_cluster = desiredClusteringMap[clustername] # elenco titoli inserzioni
        wordcloudClusterMap = {}
        complessive_words = []
        for insertion in insertions_of_cluster: # per ogni titolo ricaviamo la descrizione
            currentDesc = datasetMap[insertion]  # currentDesc e' un array di parole
            for word in currentDesc:
                complessive_words.append(word)

        # ora abbiamo tutte le parole di un cluster
        c = Counter(complessive_words)
        stop = set(STOPWORDS)
        stop.add('get')  # Add some ad-hoc stopwords often appearing in listings
        stop.add('use')
        stop.add('good')
        stop.add('best')
        stop.add('custom')
        stop.add('list')
        stop.add('free')
        stop.add('send')
        stop.add('ship')
        stop.add('onion')
        stop.add('feedback')
        stop.add('qualiti')
        stop.add('quality')
        stop.add('grams')
        stop.add('address')
        stop.add('order')
        stop.add('pleas')
        stop.add('price')
        stop.add('product')
        stop.add('check')
        stop.add('discuss')
        stop.add('aaa')
        stop.add('name')
        stop.add('shipping')
        stop.add('one')
        stop.add('track')
        stop.add('day')
        stop.add('time')
        stop.add('packag')
        stop.add('mg')
        stop.add('high')
        stop.add('gram')
        for word in c.most_common():
            if word[0] in stop: #small stopword list
                continue
            key = word[0]
            value = word[1]
            wordcloudClusterMap[key]=value
        wc = WordCloud(background_color="black", max_words=100,
                       stopwords=stop)
        wc = wc.generate_from_frequencies(wordcloudClusterMap)
        path = str(clustername)+"cluster_wordcloud.png"
        #wc.to_file(path) gives out an error
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    return X,LDALabels   # y for kmeans, LDALabels for LDA
X, y = clustering_descriptions()
print "Clusters visualization with SVD...\n"
######## Data visualization ########


def visualize():

    svd = TruncatedSVD(n_components=2, n_iter=30) # number = number of clusters
    X_scaled = svd.fit_transform(X)  # Not normalized, can accept also term-count matrix (X_tf in this program)
    # Write out coordinates and plots
    print "TruncatedSVD performed, writing coordinates and plotting...\n"
    fcoords = open('coords.csv', 'wb')
    for vector in X_scaled:
        if len(vector) != 2:
            continue
        string = str(vector[0]) + '\t' + str(vector[1]) + '\n'
        fcoords.write(string)
    fcoords.close()

    # Plotting dimensional-scaled data

    Xp = np.loadtxt('coords.csv', delimiter="\t")
    chars = '0123456789ABCDEF'
    colors = ['#'+''.join(random.sample(chars, 6)) for i in range(25)]  # We can have maximum 25 clusters
    for i in range(Xp.shape[0]):
        plt.scatter(Xp[i][0], Xp[i][1], c=colors[y[i]], s=10)
    plt.title('Plotting of dimensional scaled dataset (loss of precision)')
    plt.show()

visualize()

print "---STEP TWO - CLUSTERING OF LISTING TITLES---"


def text_cleaning_titles():
    stemmer = PorterStemmer()
    stop = set(stopwords.words('english'))
    stop.add('get')  # Add some ad-hoc stopwords often appearing in listings
    stop.add('use')
    stop.add('good')
    stop.add('best')
    stop.add('custom')
    stop.add('list')
    stop.add('free')
    stop.add('send')
    stop.add('ship')
    stop.add('onion')
    stop.add('feedback')
    stop.add('qualiti')
    stop.add('quality')
    stop.add('grams')
    stop.add('mg')
    stop.add('gr')
    stop.add('address')
    stop.add('order')
    stop.add('pleas')
    stop.add('price')
    stop.add('product')
    stop.add('check')
    stop.add('discuss')
    stop.add('name')
    stop.add('shipping')
    stop.add('one')
    stop.add('track')
    stop.add('day')
    stop.add('time')
    stop.add('packag')
    frequency = defaultdict(int)
    for key in datasetMap:
        currentTitle = key
        # Performing cleaning on key (title of listing)
        # Tokenizing
        tokens = nltk.word_tokenize(currentTitle)
        tokens_nostop = []
        for token in tokens:
            #Removal of numerical tokens
            if token not in stop:
                if token.isalpha():
                    tokens_nostop.append(token)
        # Stemming
        stems = []
        for token in tokens_nostop:
            word = stemmer.stem(token)
            if word not in stop:
                stems.append(word)
            else:
                continue
            frequency[word] += 1
        datasetMap[key] = stems
    # Now discard unique tokens or monograms or bigrams and typos (too rare words)
    for word in frequency.keys():
        forGephi.append(word)
    for key in datasetMap:
        title = datasetMap[key]
        newTitle = []
        for word in title:
            if len(word) > 1:  # A word must be at least 2 letters long
                if frequency[word] > 5:  # A word must appear at least 5 times
                    newTitle.append(word)
            else:
                frequency[word] = 0
                continue
        datasetMap[key] = newTitle
    print "Text cleaning completed.\n"
#Has side effect to datasetMap, is ok, not used anymore
print 'Text cleaning phase...'
print
text_cleaning_titles()



def clustering_titles():
    # Tf-Idf building
    print "Computing tf-idf with scikit-learn...\n"

    # datasetMap.keys() is the array with all the listings titles
    # datasetMap.values() is the array with all the titles as well, withtout numbers, each title is an array of words
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, use_idf=True)
    # Maximum document frequency is 50%, minimum in 2 documents
    # We need to redefine title, a title can't be an array of words but a single string
    docs = []
    for titleArray in datasetMap.values():
        currentString = ''
        for word in titleArray:
            currentString = currentString + word + " "
        currentString.strip()
        docs.append(currentString)
    # docs is now an array of titles, each title is a single string with spaces
    X = vectorizer.fit_transform(docs)
    featureNames = vectorizer.get_feature_names()
    print "Tf-idf matrix generated:"
    print("n_samples: %d, n_features: %d" % X.shape)
    print

    # Clustering with K-Means

    print "Clustering with K-Means...\n"

    # print "Searching for optimal number of clusters through the elbow point method...\n"
    # # This method is very long, comment this block and set a cluster value to skip
    # MAX_K = 14
    # ks = range(1, MAX_K + 1)
    # inertias = np.zeros(MAX_K)
    # diff = np.zeros(MAX_K)
    # diff2 = np.zeros(MAX_K)
    # diff3 = np.zeros(MAX_K)
    # for k in ks:
    #     print "---Test with " + str(k) + " clusters."
    #     kmeans = KMeans(n_clusters=k, verbose=0, init='k-means++').fit(X)
    #     inertias[k - 1] = kmeans.inertia_
    #     # first difference
    #     if k > 1:
    #         diff[k - 1] = inertias[k - 1] - inertias[k - 2]
    #     # second difference
    #     if k > 2:
    #         diff2[k - 1] = diff[k - 1] - diff[k - 2]
    #     # third difference
    #     if k > 3:
    #         diff3[k - 1] = diff2[k - 1] - diff2[k - 2]
    # elbow = np.argmin(diff3[3:]) + 3
    #
    # # Uncomment following lines (plt.) to show plot of elbow point method
    # print "Plotting results of elbow point method...\n"
    # plt.plot(ks, inertias, "b*-")
    # plt.plot(ks[elbow], inertias[elbow], marker='o', markersize=12,
    #          markeredgewidth=2, markeredgecolor='r', markerfacecolor=None)
    # plt.ylabel("Inertia")
    # plt.xlabel("K")
    # plt.show()

    # Computing K-Means with optimal number of clusters

    optimal_n_clusters = NUMBER_OF_CLUSTERS  # Set this value by hand if skipped elbow method
    print "Optimal value for number of clusters: " + str(optimal_n_clusters) + ' clusters\n'
    print "Performing K-Means clustering with " + str(optimal_n_clusters) + " clusters...\n"
    kmeans = KMeans(n_clusters=optimal_n_clusters, verbose=2, init='k-means++').fit(X)
    y = kmeans.labels_
    print "K-Means clustering completed.\n"

    print "Clustering with LDA...\n"
    tf_vectorizer = CountVectorizer(max_df=0.5, min_df=2, stop_words='english')
    X_tf = tf_vectorizer.fit_transform(docs)
    tf_feature_names = tf_vectorizer.get_feature_names()
    ldac = lda.LDA(n_topics=optimal_n_clusters, n_iter=1200,
                   random_state=0)
    ldac.fit(X_tf)


    doc_topic = ldac.doc_topic_
    titles = datasetMap.keys()
    LDALabels = []
    for i in range(len(titles)):
        # print("{} (top topic: {})".format(titles[i], doc_topic[i].argmax()))
        LDALabels.append(doc_topic[i].argmax())  # contains LDA clustering labels
    print


    # Clustering results saving to file
    # datasetMap.keys() is the array of the titles, we will write those to file
    # in order to write the descriptions, array is docs (simple array) docs[0] is the first description

    print "Writing clustering results to file: clustering_output (first 30 elements per cluster)...\n"
    clusteringMap = defaultdict(list)
    clusteringLDAMap = defaultdict(list)
    titles = datasetMap.keys()
    for i in range(len(titles)):
        clusteringMap[y[i]].append(titles[i])  # y[i] for kmeans, LDALabels[i] for LDA
        clusteringLDAMap[LDALabels[i]].append(titles[i])

    out_file = open('clustering_output_onlytitles', 'wb')
    # Only writing first 30 titles per cluster
    out_file.write("Clustering of titles.\nFor each cluster, 50 titles of listings are reported")
    out_file.write("\n\n\n**** KMeans Clustering results: ****\n\n")
    for clustername in clusteringMap.keys():
        limit = 0
        out_file.write("Cluster: " + str(clustername) + '\n')
        for elem in clusteringMap[clustername]:
            if limit == 49:
                break
            out_file.write(str(elem) + "||")
            limit += 1
        out_file.write('\n')
    out_file.write("\nTop-terms per cluster...\n")
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(optimal_n_clusters):
        out_file.write("Cluster %d:\n" % i)
        for ind in order_centroids[i, :10]:
            out_file.write(' %s\n' % terms[ind])
        out_file.write('\n')
    out_file.write("\n\n\n**** LDA Clustering results ****\n\n")
    for clustername in clusteringLDAMap.keys():
        limit = 0
        out_file.write("Cluster: " + str(clustername) + '\n')
        for elem in clusteringLDAMap[clustername]:
            if limit == 49:
                break
            out_file.write(str(elem) + "||")
            limit += 1
        out_file.write('\n')
    out_file.write("\nTop-terms per cluster...\n")
    out_file.write(display_topics(ldac, tf_feature_names, 30))
    out_file.close()

    # costruzione counter per wordclouds
    #
    print "Wordcloud generation...\n"
    # IMPORTANTE -  PER GENERARE WORDCLOUD DEI CLUSTER LDA, SOSTITUIRE CLUSTERINGMAP CON CLUSTERING LDA MAP
    desiredClusteringMap = clusteringLDAMap  #sostituibile con clusteringLDAMap
    for clustername in desiredClusteringMap.keys():
        insertions_of_cluster = desiredClusteringMap[clustername]  # elenco titoli inserzioni
        wordcloudClusterMap = {}
        complessive_words = []
        for insertion in insertions_of_cluster:  # per ogni titolo ricaviamo la descrizione (titolo senza numero)
            currentTitle = datasetMap[insertion]  # currentTitle e' un array di parole
            for word in currentTitle:
                complessive_words.append(word)

        # ora abbiamo tutte le parole di un cluster
        c = Counter(complessive_words)
        stop = set(STOPWORDS)
        stop.add('get')  # Add some ad-hoc stopwords often appearing in listings
        stop.add('use')
        stop.add('good')
        stop.add('best')
        stop.add('custom')
        stop.add('list')
        stop.add('free')
        stop.add('send')
        stop.add('ship')
        stop.add('onion')
        stop.add('feedback')
        stop.add('qualiti')
        stop.add('quality')
        stop.add('grams')
        stop.add('address')
        stop.add('order')
        stop.add('pleas')
        stop.add('price')
        stop.add('product')
        stop.add('check')
        stop.add('discuss')
        stop.add('name')
        stop.add('shipping')
        stop.add('one')
        stop.add('track')
        stop.add('day')
        stop.add('time')
        stop.add('packag')
        stop.add('mg')
        stop.add('grams')
        stop.add('gram')
        stop.add('aaa')
        for word in c.most_common():
            if word[0] in stop:  # small stopword list
                continue
            key = word[0]
            value = word[1]
            wordcloudClusterMap[key] = value

        wc = WordCloud(background_color="black", max_words=100,
                       stopwords=stop)
        wc = wc.generate_from_frequencies(wordcloudClusterMap)
        path = str(clustername) + "cluster_wordcloud.png"
        # wc.to_file(path) gives out an error
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    return X,LDALabels  #LDALabels instead of y for LDA Clustering
X2, y2 = clustering_titles()

print "Clusters visualization with SVD...\n"


def visualize2():
    svd = TruncatedSVD(n_components=2, n_iter=30)
    X_scaled = svd.fit_transform(X2)  # Not normalized, can accept also term-count matrix (X_tf in this program)
    # Write out coordinates and plots
    print "TruncatedSVD performed, writing coordinates and plotting...\n"
    fcoords = open('coordsTitles.csv', 'wb')
    for vector in X_scaled:
        if len(vector) != 2:
            continue
        string = str(vector[0]) + '\t' + str(vector[1]) + '\n'
        fcoords.write(string)
    fcoords.close()

    # Plotting dimensional-scaled data

    Xp = np.loadtxt('coordsTitles.csv', delimiter="\t")
    chars = '0123456789ABCDEF'
    colors = ['#' + ''.join(random.sample(chars, 6)) for i in range(25)]  # We can have maximum 25 clusters
    for i in range(Xp.shape[0]):
        plt.scatter(Xp[i][0], Xp[i][1], c=colors[y2[i]], s=10)
    plt.title('Plotting of dimensional scaled dataset (loss of precision)')
    plt.show()
visualize2()


# Building world map for visualization
print "---STEP THREE - SET UP FILES FOR COUNTRY DATA VISUALIZATION---\n"
print "Setting up dataset for data visualization...\n"


countryMap = defaultdict(list)
for key in datasetGeo:
    country = datasetGeo[key]
    if 'worldwide' not in country:
        countryMap[country.rstrip('\n')].append(key)

# Rebuilding country map
for k in countryMap.keys():
    if k == 'shipping from:' or k == ' north' or k == ' and no':
        del countryMap[k]
        continue
    if k == 'eu' or k == 'europe eu':
        countryMap['european union'].extend(countryMap[k])
        del countryMap[k]
    if k == 'usa and canada' or k == ' and canada':
        countryMap['united states'].extend(countryMap[k])
        countryMap['canada'].extend(countryMap[k])
        del countryMap[k]
    if k == ' mexico and canada':
        countryMap['mexico'].extend(countryMap[k])
        countryMap['canada'].extend(countryMap[k])
        del countryMap[k]
    if k == 'usa':
        countryMap['united states'].extend(countryMap[k])
        del countryMap[k]
    if k == 'australia and new zeland':
        countryMap['australia'].extend(countryMap[k])
        countryMap['new zealand'].extend(countryMap[k])
        del countryMap[k]
    if k == 'saint kitts and nevi':
        countryMap['Saint Kitts and Nevis'].extend(countryMap[k])
        del countryMap[k]

for k in countryMap.keys():
    if k == 'european union':
        countryMap['italy'].extend(countryMap[k])
        countryMap['france'].extend(countryMap[k])
        countryMap['germany'].extend(countryMap[k])
        countryMap['spain'].extend(countryMap[k])
        countryMap['portugal'].extend(countryMap[k])
        del countryMap[k]
    if k == 'scandinavia':
        countryMap['sweden'].extend(countryMap[k])
        countryMap['norway'].extend(countryMap[k])
        del countryMap[k]
    if k == 'russia':
        countryMap['Russian Federation'].extend(countryMap[k])
        del countryMap[k]
    if k == 'vatican city':
        countryMap['Holy See (Vatican City State)'].extend(countryMap[k])
        del countryMap[k]
    if k == 'laos':
        countryMap["Lao People's Democratic Republic"].extend(countryMap[k])
        del countryMap[k]
    if k == 'korea':
        countryMap["Korea, Republic of"].extend(countryMap[k])
        del countryMap[k]
    if k == 'iran':
        countryMap["Iran, Islamic Republic of"].extend(countryMap[k])
        del countryMap[k]
    if k == 'czech republic':
        countryMap["Czechia"].extend(countryMap[k])
        del countryMap[k]


# World map representation - 3 letters code building
codeList = []
numberOfInsertions = []
countryList = []
itemsold = []
classifications = OrderedDict()
for k in countryMap.keys():
    #print str(k) +" : " +str(countryMap[k])
    countryList.append(str(k))
    itemsold.append(countryMap[k])
    if str(k) == 'ivory coast':
        code = "CIV"
        #print code
        codeList.append(code)
        numberOfInsertions.append(len(countryMap[k]))
        classifications.setdefault(code,[])
        continue
    if str(k) == 'Iran, Islamic Republic of':
        code = "IRN"
        #print code
        codeList.append(code)
        numberOfInsertions.append(len(countryMap[k]))
        classifications.setdefault(code, [])
        continue
    if str(k) == 'Saint Kitts and Nevis':
        code = "KNA"
        #print code
        codeList.append(code)
        numberOfInsertions.append(len(countryMap[k]))
        classifications.setdefault(code, [])
        continue
    if str(k) == 'Korea, Republic of':
        code = "KOR"
        #print code
        codeList.append(code)
        numberOfInsertions.append(len(countryMap[k]))
        classifications.setdefault(code, [])
        continue
    if str(k) == "Lao People's Democratic Republic":
        code = "LAO"
        #print code
        codeList.append(code)
        numberOfInsertions.append(len(countryMap[k]))
        classifications.setdefault(code, [])
        continue
    try:
        code = pycountry.countries.get(name=(str(k)).title()).alpha_3
        #print code
        codeList.append(code)
        numberOfInsertions.append(len(countryMap[k]))
        classifications.setdefault(code, [])
    except:
        try:
            code = pycountry.countries.get(common_name=(str(k)).title()).alpha_3
            #print code
            codeList.append(code)
            numberOfInsertions.append(len(countryMap[k]))
            classifications.setdefault(code, [])
        except:
            try:
                name = str(k)
                tokens = nltk.word_tokenize(name)
                if len(tokens) == 1:
                    code = pycountry.countries.get(name=(str(k)).title()).alpha_3
                    #print code
                    codeList.append(code)
                    numberOfInsertions.append(len(countryMap[k]))
                    classifications.setdefault(code, [])
                else:
                    newname = ''
                    for token in tokens:
                        if token == 'and':
                            newname = newname + ' and '
                            continue
                        else:
                            newname = newname + token.title()
                    code = pycountry.countries.get(name=newname).alpha_3
                    #print code
                    codeList.append(code)
                    numberOfInsertions.append(len(countryMap[k]))
                    classifications.setdefault(code, [])
            except:
                print str(k)
                print "Code error."



print
print "Classification for items sold...\n"

##### Setting up a complex training set for naivebayes classifier
nb = naivebayes.NaiveBayes()
nb.stop_word = ["the","for","from","real","of","a","top","x","best","original","grams","gram"]


def train(nb):
    # Training set, from listings
    with open('trainingSet.txt','r') as trainingfile:
        for line in trainingfile:
            splittedLine = line.split(':')
            listing = splittedLine[1]
            listingSplitted = listing.split('->')
            name = listingSplitted[0].strip()
            classification = listingSplitted[1].strip()
            nb.train(classification,name)

    # Training with some words
    nb.train("heavy drug","cocaine uncut kokaine heroin eroina ero opium")
    nb.train("heavy drug","coke heroin kokain crack cocain")
    nb.train("heavy drug","cocaine colombia heroin eroin mg")

    nb.train("soft drug","weed purple haze superskunk skunk cannabis moroccan afghan hashish hash")
    nb.train("soft drug","hq amnesia kush haze charas marijuana dutch coffee shop thc indica sativa mushrooms mushroom shrooms")
    nb.train("soft drug","white widow haze skunk marijuana new harvest high htc wax strain strains joints")
    nb.train("soft drug","hash polm seeds peyote hash pakistan brown sugar kush mushroom smoke pollen joint")

    nb.train("medical","medical alprazolam sildenafil modafinil loprazolam xanax viagra diazepam lorazepam")
    nb.train("medical","generic viagra restoril roche xanax bayer novartis pfizer sandoz oxazepan lormetazepam codein pfizer clonazepam")
    nb.train("medical","hydrocodone tablets prazepam morphine propranolol subutex clonazepam novartis mg rivotril pills")
    nb.train("medical","sarcoplex nandrolone decanoate valium adderall adderrall ritalin codeine pharma modafinil")

    nb.train("synthetic drug","methadone hydromorphone meth mg pills crystal mdma ecstasy xtc lsd ketamine amphetamine")
    nb.train("synthetic drug","methadone hcl nbome lsd crystals superman xtc ketamin")
    nb.train("synthetic drug","amphetamin amfetamin mdmp speed paste chupa chups mdma")
    nb.train("synthetic drug","tan mdma crystals purity methamphetamin mdpv ecstasy xtc")

    nb.train("weapons","kalashnikov kalasnikov m16 m4 ak47 beretta gun stun taser firecracker 9mm walther knife")
    nb.train("weapons","mm rounds bullets ak pistol makarov")
    nb.train("weapons","pepper spray")


    nb.train("leaked account-electronic goods","leaked photos netflix laundry amazon account lifetime phishing paypal login password email linux hacking giftcard apple amazon")
    nb.train("leaked account-electronic goods","paypal accounts balance hacking websites invitation code ccbank tutorial attach download pandora alphabay autodesk")
    nb.train("leaked account-electronic goods","facebook malware rootkits referral porn sex nude account hacked invite link web hosting vpn video videos root shell server")
    nb.train("leaked account-electronic goods","bitcoin malware keylogger miner")
    nb.train("cigarettes","cigarettes marlboro tobacco chesterfield davidoff parliament camel")
    nb.train("cigarettes","lucky strike")

    nb.train("other","custom listing customized audi bmw car tuning samsung iphone asus movie")

    nb.train("carding","carding amex visa mastercard cc cvv bank credit card western union american express money cards")
    nb.train("carding","single sided id card printer cc iban skrill bills note euro")
    nb.train("carding","clean visa cc mastercard prepaid giftcard cash bitcoins")

    nb.train("watches\clothes","counterfeit watch rolex submariner emporio armani breitling chrono replica watch")
    nb.train("watches\clothes","seiko women gucci bag solar automatic watch chanel dior hermes")
    nb.train("watches\clothes","nike trainer mens running trainers size eu adidas burberry polo")

    nb.train("documents","scan id driver licenses driving license how cookbook blackbook licence passport")
    nb.train("documents","wordpress book exploiter become admin guide manual wordpress howto ways how site exploit tutorial")
    nb.train("documents","bills macgyver handbook guide ebook pdf how dummies secrets handbook make")


train(nb)
i = 0
fileOut = open('classificationOutput','wb')
for listing in itemsold:
    code = codeList[i]
    codew = code + '\n'
    fileOut.write(str(codew))
    for elem in listing:
            # per classificare una stringa (sempre come array!)
            # In[15]: docs_new = ["hyper cocaine","excellent weed","amazing hashish"]
            # In[16]: X_new_counts = count_vect.transform(docs_new)
            # In[17]: X_new_tfidf = tf_transformer.transform(X_new_counts)
            # In[18]: predicted = clf.predict(X_new_tfidf)
            # In[19]: print predicted
            # ['heavy' 'soft' 'soft']
            # string = ''
            # for char in str(elem):
            #     if char.isdigit():
            #         string = string +" "
            #         continue
            #     else:
            #         string = string + char
            # new_string = [string.strip()]   #classificazione su titolo
            # X_new_counts = count_vect.transform(new_string)
            # X_new_tfidf = tf_transformer.transform(X_new_counts)
            # prediction = classificator.predict(X_new_tfidf)
            prediction = nb.classify(str(elem))
            line = str(elem)+" -> "+prediction+"\n"
            fileOut.write(line)
            classifications[code].append(prediction)
    fileOut.write("-------\n")
    i += 1
fileOut.close()
# Most sold item for each country
mostRelevant = []

mostSoldFile = open('mostSold','wb')
for code in classifications.keys():
    current = classifications[code] #list of predictions
    data = Counter(current)
    mostRelevant.append(str(data.most_common()[0][0]))
    temp = []
    line = str(code)+'||'
    for elem in data.most_common():
        line = line + str(elem[0])+"&"+str(elem[1])+'|'
    mostSoldFile.write(line+'\n')
mostSoldFile.close()

print "Writing listing classifications on file for visualization...\n"
df = pandas.DataFrame(data={"Code": codeList, "Insertions:": numberOfInsertions, "Country:": countryList, "Most common:":mostRelevant
                            })
df.to_csv("./countryFrequency.csv", sep=',',index=False)
