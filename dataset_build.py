ALPHADATASET = 'alpha_dataset'
PANDORADATASET = 'pandora_dataset_big'
HANSADATASET = 'hansa_dataset'

FINAL_DATASET_SMALL = 'dataset_small'
FINAL_DATASET_BIG = 'dataset_big'

# desired format: "vendor_name","price","name","description","ship_from"

exitFile = open(FINAL_DATASET_BIG,'w')

with open(PANDORADATASET) as pandorafile:
    for line in pandorafile:
        exitFile.write(line)
print 'Pandora dataset copied'
with open(ALPHADATASET) as alphafile:
    for line in alphafile:
        exitFile.write(line)
print 'AlphaBay dataset copied'
with open(HANSADATASET) as hansafile:
    for line in hansafile:
        exitFile.write(line)
print 'Hansa dataset copied'

exitFile.close()
