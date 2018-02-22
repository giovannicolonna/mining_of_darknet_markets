import os
import csv
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


### AlphaBay dataset reader
print "****ALPHA BAY MARKET DATASET BUILDER****\n"
# files are already in csv, we need to join all the files, remove duplicates

# format hash,"market_name","item_link","vendor_name","price","name","description","image_link","add_time","ship_from",
ALPHAPATH = 'C:\Users\Giovanni\Desktop\\alphabay'
DATASET_ALPHA = 'alpha_dataset'
hashes = []
count = 0
stopwordsCached = set(stopwords.words('english'))
exitFile2 = open(DATASET_ALPHA,'w')

for root,dirs,files in os.walk(ALPHAPATH):

    for file in files:
        with open(os.path.join(root, file), "r") as currentfile:
            first = True
            reader = csv.reader(currentfile)
            for row in reader:
                if first:
                    first = False
                    continue
                hash = row[0]
                if not hash in hashes:
                    hashes.append(hash)
                    count += 1
                else:
                    continue

                # row[6] is the description

                # remove links, snippet from stackoverflow
                row[6] = re.sub(r"http\S+", "", row[6])
                # let's clean up description with only alphabetic chars
                row[6] = re.sub(r'[^a-zA-Z ]+', '', row[6])

                # let's perform stopword removal and stemming from description
                # 1 --- stopword removal
                description = row[6].lower()
                tokens = word_tokenize(description)
                filteredDescription = [word for word in tokens if word not in stopwordsCached]
                filteredDescription = [word for word in filteredDescription if word not in stopwordsCached]
                description = ''
                for word in filteredDescription:
                    description = description + word + ' '
                #title cleaning

                row[5] = re.sub(r'[^a-zA-Z0-9 ]+', '', row[5])

                result_string = row[3]+","+row[4]+","+row[5]+","+description+","+row[9]+'\n'
                result_string = result_string.lower()
                #remove links, snippet from stackoverflow
                result_string = re.sub(r"http\S+", "", result_string)
                exitFile2.write(result_string)

            # rows are ended, end of this file
                if count%100 == 0:
                    print str(count)+" items has been read."
exitFile2.close()
print str(count)+" total items."