import os
import csv
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


### Hansa dataset reader
print "****HANSA MARKET DATASET BUILDER****\n"
# files are already in csv, we need to join all the files, remove duplicates

# format "time","item_id","title","vendor","price","type","nation"
HANSAPATH = 'C:\Users\Giovanni\PycharmProjects\dm_project\hansa-marketplace-listings-december-2016.csv'

DATASET_HANSA = 'hansa_dataset'

count = 0
stopwordsCached = set(stopwords.words('english'))
exitFile2 = open(DATASET_HANSA,'w')
with open(HANSAPATH, "r") as currentfile:

    reader = csv.reader(currentfile, quotechar='"', doublequote=True, delimiter=',',quoting=csv.QUOTE_ALL, skipinitialspace=True)
    for row in reader:
        count = count + 1
        # row[2] is the title
        # remove links, snippet from stackoverflow
        row[2] = re.sub(r"http\S+", "", row[2])
        # let's clean up description with only alphabetic chars
        row[2] = re.sub(r'[^a-zA-Z ]+', '', row[2])

        # let's perform stopword removal and stemming from description
        # 1 --- stopword removal
        description = row[2].lower()
        tokens = word_tokenize(description)
        filteredDescription = [word for word in tokens if word not in stopwordsCached]
        filteredDescription = [word for word in filteredDescription if word not in stopwordsCached]
        description = ''
        for word in filteredDescription:
            description = description + word + ' '
        #title cleaning

        description = re.sub(r'[^a-zA-Z0-9 ]+', '', description)
        try:
            if len(row)>7:
                size = len(row) - 1
                row[size] = re.sub(r'[^a-zA-Z ]+', '', row[size])
                price = row[4]
                # comma removals from prices
                price = re.sub(r'[^a-zA-Z0-9 .]+', '', price)
                price = re.sub(r'[^a-zA-Z0-9 .]+', '.', price)
                result_string = row[3] + "," + price + "," + description + "," + description + "," + str(row[size]).lstrip(" ") + '\n'
                result_string = result_string.lower()
            else:
                row[6] = re.sub(r'[^a-zA-Z ]+', '', row[6])
                price = row[4]
                # comma removals from prices
                price = re.sub(r'[^a-zA-Z0-9 .]+', '', price)
                price = re.sub(r'[^a-zA-Z0-9 .]+', '.', price)
                result_string = row[3]+","+price+","+description+","+description+","+str(row[6]).lstrip(" ")+'\n'
                result_string = result_string.lower()
        except:
            try:
                price = row[4]
                # comma removals from prices
                price = re.sub(r'[^a-zA-Z0-9 .]+', '', price)
                price = re.sub(r'[^a-zA-Z0-9 .]+', '.', price)
                result_string = row[3] + "," + price + "," + description + "," + description + ",worldwide\n"
                result_string = result_string.lower()
            except:
                result_string = 'a,0,' + description + "," + description + ",worldwide\n"
                result_string = result_string.lower()
        #remove links, snippet from stackoverflow
        result_string = re.sub(r"http\S+", "", result_string)
        exitFile2.write(result_string)

        # rows are ended, end of this file
        if count%100 == 0:
            print str(count)+" items has been read."
exitFile2.close()
print str(count)+" total items."