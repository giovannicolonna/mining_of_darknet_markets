import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup

### PANDORA dataset reader

print "****PANDORA BLACK MARKET HTML PARSER****\n"

#PATH = 'C:\Users\Giovanni\PycharmProjects\dm_project\pandora-one-day'
PATHBIG = 'C:\Users\Giovanni\Desktop\pandora_full'
#DATASET_SMALL = 'pandora_dataset_small'
DATASET_BIG = 'pandora_dataset_big'

exitFile = open(DATASET_BIG,'w')
hashes = []
count = 0
total = 0
doubles = 0
exceptions = 0
stopwordsCached = set(stopwords.words('english'))

emptyprices = 0
for root,dirs,files in os.walk(PATHBIG):
    count = 0
    for file in files:
        #file is the hashcode of the insertion, can be used to avoid duplicates
        if not str(file) in hashes:
            hashes.append(str(file))
        else:
            doubles += 1
            #print "duplicate found in file: " + str(file)
            continue


        with open(os.path.join(root,file),"r") as currentfile:
            # file is an html scrap
            soup = BeautifulSoup(currentfile,'html.parser')
            itemhash = file

            # access to item details
            #soup.find_all(id='content')[0]

            title = str(soup.find_all('th', colspan="2"))
            title = BeautifulSoup(title,'html.parser').get_text()

            description = str(soup.find_all('pre'))
            description = BeautifulSoup(str(description),'html.parser').get_text()
            description = description.encode('ascii','ignore')
            # remove special chars
            description = description.replace('\\r','')
            description = description.replace('\\n','')
            description = description.replace('\\t','')

            # remove pgp keys from description
            description = re.sub('-----BEGIN(.*)','',description) #regular expression

            # remove special unicode chars /u42042
            description = description.decode('unicode_escape').encode('ascii','ignore')
            try:
                for child in soup.find_all('form', action=str("/item/"+itemhash))[0].children:
                    count += 1
                    total += 1
                    infos = BeautifulSoup(str(child), 'html.parser').get_text(separator='__',strip=True)
                    fields = infos.encode('ascii','ignore').split('__')


                    #information parsing from html
                    # fields[0] = title, not needed
                    # fields[1] = "seller:", not needed
                    # fields[2] = seller name   OK
                    # fields[3] = (Transaction: num_trans |  OK, to be parsed
                    # fields[4] = Rating: rating/5  OK, to be parsed
                    # fields[5] = ), discard
                    # fields[6] = 'Price:', discard
                    # fields[7][8][9][10][11] <---- fields[11] is btc
                    # fields[13] = ship from
                    # fields[15] = ship to


                    # desired format: "vendor_name","price","name","description","ship_from"

                    # remove links, snippet from stackoverflow
                    description = re.sub(r"http\S+", "", description)

                    # let's clean up description with only alphabetic chars
                    description = re.sub(r'[^a-zA-Z ]+', '', description)

                    #let's perform stopword removal and stemming from description
                    #1 --- stopword removal
                    description = description.lower()
                    tokens = word_tokenize(description)
                    filteredDescription = [word for word in tokens if word not in stopwordsCached]
                    filteredDescription = [word for word in filteredDescription if word not in stopwordsCached]
                    description = ''
                    for word in filteredDescription:
                        description = description + word + ' '

                    #title cleaning
                    fields[0] = re.sub(r'[^a-zA-Z0-9 ]+', '', fields[0])

                    #price cleaning (only btc)
                    price = fields[11][:-6]
                    price = re.sub(r'[^0-9.]+', '', price)

                    if price == '': emptyprices += 1

                    result_string = fields[2]+','+price+','+fields[0]+','\
                                    +description+','+fields[13]+'\n'
                    result_string = result_string.lower()


                    exitFile.write(result_string)
            #now we need to extract description
            except:
                exceptions += 1
                print "Exception in file: "+str(file)
                #continue

            if count%50==0:
                print str(count)+" items extracted from current folder."
            #print (soup.prettify())
            #print itemhash
            #print "========================="

exitFile.close()

print "Total number of listings extracted: " + str(total)
#print "Total number of duplicate listings: " + str(doubles)
print "Total number of malformed scrapes: " + str(exceptions)
print "Empty prices: " + str(emptyprices)




