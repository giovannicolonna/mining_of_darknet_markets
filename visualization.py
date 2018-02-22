import pandas as pd
import plotly
from plotly.graph_objs import *

df = pd.read_csv('countryFrequency.csv')
print "Building total world map listings...\n"
data = [ dict(
        type = 'choropleth',
        locations = df['Code'],
        z = df['Insertions:'],
        text = df['Country:'],
        colorscale = [[0,"rgb(221, 228, 254)"],
                      [0.001,"rgb(221, 228, 254)"],  #from 0 to 5 listings

                      [0.001,"rgb(189, 201, 246"], #from 5 to 20
                      [0.004,"rgb(189, 201, 246)"],

                      [0.004,"rgb(157, 175, 239)"], #from 20 to 50
                      [0.01,"rgb(157, 175, 239)"],

                      [0.01,"rgb(126, 149, 232)"], #from 50 to 150
                      [0.03,"rgb(126, 149, 232)"],

                      [0.03,"rgb(94, 122, 225)"], #from 150 to 400
                      [0.08,"rgb(94, 122, 225"],

                      [0.08, "rgb(63, 96, 218"], #from 400 to 1000
                      [0.2, "rgb(63, 96, 218"],

                      [0.2, "rgb(27, 67, 210)"],  #from 1000 to 3000
                      [0.4, "rgb(27, 67, 210"],

                      [0.6,"rgb(0, 44, 204)"],
                      [1,"rgb(0, 0, 190)"]],
        #colorscale = 'Portland',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar=dict(
            autotick=True,
            tickprefix='#',
            tickvals=[100,500,1000,2000,5000],
            title='Listings per country'),
      ) ]

layout = dict(
    title='Number of listings with a shipping country specified',
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection=dict(
            type='Mercator'
        )
    )
)
fig = dict( data=data, layout=layout )
plotly.offline.plot( fig, validate=False, filename='number-insertions.html' )

# MOST SOLD ITEM MAP BUILDING #
print "Building world map of most sold items...\n"
#For each kind, set up the arrays with names, codes and binary for map
softdrugcountries = []
softdrugbinary = []
softdrugcodes = []
for i in range(len(df['Code'])):
    currentcode = df['Code'][i]
    if df['Most common:'][i] == 'soft drug':
        softdrugbinary.append(1)
        softdrugcodes.append(currentcode)
        softdrugcountries.append(df['Country:'][i])

trace1 = Choropleth(   #first trace, for soft drugs
    z=softdrugbinary, # a number of ones equal to nations with most selled item = "soft drugs"
    autocolorscale=False,
    colorscale=[[0, 'rgb(255,255,255)'], [1, 'rgb(186,58,51)']],
    text=softdrugcountries,
    locations=softdrugcodes,
    name='Soft drugs',
    hoverinfo='text+name',
    showscale=False,
    zauto=False,
    zmax=1,
    zmin=0,
)
heavydrugcountries = []
heavydrugbinary = []
heavydrugcodes = []
for i in range(len(df['Code'])):
    currentcode = df['Code'][i]
    if df['Most common:'][i] == 'heavy drug':
        heavydrugbinary.append(1)
        heavydrugcodes.append(currentcode)
        heavydrugcountries.append(df['Country:'][i])
trace2 = Choropleth(
    z=heavydrugbinary,
    autocolorscale=False,
    colorscale=[[0, 'rgb(255,255,255)'], [1, 'green']],
    text=heavydrugcountries,
    locations=heavydrugcodes,
    name='Heavy drugs',
    hoverinfo='text+name',
    showscale=False,
    zauto=False,
    zmax=1,
    zmin=0,
)
leakcountries = []
leakbinary = []
leakcodes = []
for i in range(len(df['Code'])):
    currentcode = df['Code'][i]
    if df['Most common:'][i] == 'leaked account-electronic goods':
        leakbinary.append(1)
        leakcodes.append(currentcode)
        leakcountries.append(df['Country:'][i])
trace3 = Choropleth(
    z=leakbinary,
    autocolorscale=False,
    colorscale=[[0, 'rgb(255,255,255)'], [1, 'blue']],
    text=leakcountries,
    locations=leakcodes,
    name='Leaked accounts - Electronic goods',
    hoverinfo='text+name',
    showscale=False,
    zauto=False,
    zmax=1,
    zmin=0,
    )
cardcountries = []
cardbinary = []
cardcodes = []
for i in range(len(df['Code'])):
    currentcode = df['Code'][i]
    if df['Most common:'][i] == 'carding':
        cardbinary.append(1)
        cardcodes.append(currentcode)
        cardcountries.append(df['Country:'][i])
trace4 = Choropleth(
    z=cardbinary,
    autocolorscale=False,
    colorscale=[[0, 'rgb(255,255,255)'], [1, 'yellow']],
    text=cardcountries,
    locations=cardcodes,
    name='Carding fraud',
    hoverinfo='text+name',
    showscale=False,
    zauto=False,
    zmax=1,
    zmin=0,
    )
medcountries = []
medbinary = []
medcodes = []
for i in range(len(df['Code'])):
    currentcode = df['Code'][i]
    if df['Most common:'][i] == 'medical':
        medbinary.append(1)
        medcodes.append(currentcode)
        medcountries.append(df['Country:'][i])
trace5 = Choropleth(
    z=medbinary,
    autocolorscale=False,
    colorscale=[[0, 'rgb(255,255,255)'], [1, 'red']],
    text=medcountries,
    locations=medcodes,
    name='Medical drugs',
    hoverinfo='text+name',
    showscale=False,
    zauto=False,
    zmax=1,
    zmin=0,
    )
clcountries = []
clbinary = []
clcodes = []
for i in range(len(df['Code'])):
    currentcode = df['Code'][i]
    if df['Most common:'][i] == 'watches\clothes':
        clbinary.append(1)
        clcodes.append(currentcode)
        clcountries.append(df['Country:'][i])
trace6 = Choropleth(
    z=clbinary,
    autocolorscale=False,
    colorscale=[[0, 'rgb(255,255,255)'], [1, 'pink']],
    text=clcountries,
    locations=clcodes,
    name='Counterfeits',
    hoverinfo='text+name',
    showscale=False,
    zauto=False,
    zmax=1,
    zmin=0,
    )
sincountries = []
sinbinary = []
sincodes = []
for i in range(len(df['Code'])):
    currentcode = df['Code'][i]
    if df['Most common:'][i] == 'synthetic drug':
        sinbinary.append(1)
        sincodes.append(currentcode)
        sincountries.append(df['Country:'][i])
trace7 = Choropleth(
    z=sinbinary,
    autocolorscale=False,
    colorscale=[[0, 'rgb(255,255,255)'], [1, 'orange']],
    text=sincountries,
    locations=sincodes,
    name='Synthetic drugs',
    hoverinfo='text+name',
    showscale=False,
    zauto=False,
    zmax=1,
    zmin=0,
    )
otcountries = []
otbinary = []
otcodes = []
for i in range(len(df['Code'])):
    currentcode = df['Code'][i]
    if df['Most common:'][i] == 'other':
        otbinary.append(1)
        otcodes.append(currentcode)
        otcountries.append(df['Country:'][i])
trace8 = Choropleth(
    z=otbinary,
    autocolorscale=False,
    colorscale=[[0, 'rgb(255,255,255)'], [1, 'grey']],
    text=otcountries,
    locations=otcodes,
    name='Unclassified',
    hoverinfo='text+name',
    showscale=False,
    zauto=False,
    zmax=1,
    zmin=0,
    )
doccountries = []
docbinary = []
doccodes = []
for i in range(len(df['Code'])):
    currentcode = df['Code'][i]
    if df['Most common:'][i] == 'documents':
        docbinary.append(1)
        doccodes.append(currentcode)
        doccountries.append(df['Country:'][i])
trace9 = Choropleth(
    z=docbinary,
    autocolorscale=False,
    colorscale=[[0, 'rgb(255,255,255)'], [1, 'purple']],
    text=doccountries,
    locations=doccodes,
    name='Fake documents/ebooks',
    hoverinfo='text+name',
    showscale=False,
    zauto=False,
    zmax=1,
    zmin=0,
    )

# MAP BUILDING
data = Data([trace1,trace2,trace3,trace4,trace5,trace6,trace7,trace8,trace9])
layout = Layout(
    geo=dict(
        projection=dict(
            type='Mercator'
        ),
        showframe=False,
        showcoastlines = True
    ),
    hovermode='closest',
    images=list([
        dict(
            x=1,
            y=0.6,
            sizex=0.8,
            sizey=0.8,
            source='legenda.png',   #Legenda link
            xanchor='right',
            xref='paper',
            yanchor='middle',
            yref='paper'
        )
    ]),
    showlegend=True,
    title='Most sold items per nation',
    margin = dict(
        l=0,
        r=50,
        b=100,
        t=100,
        pad=4)
)
fig = Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='most_sold.html')

print "Building top nations most sold items...\n"

# Each trace is an insertion type
# in each trace, x is the number of top nations: USA Germany UK Netherlands Italy Australia China Canada
topNationsCode = ['USA','DEU','GBR','NLD','ITA','AUS','CHN','CAN']
topNations = ['USA','Germany','United Kingdom','Netherlands','Italy','Australia','China','Canada']


def builder(type):
    y = []
    with open('mostSold','r') as f:
        for j in range(len(topNationsCode)):
            for line in f:
                splittedLine = line.split('||')
                if splittedLine[0] == topNationsCode[j]:
                    itemlist = splittedLine[1].split('|')
                    for elem in itemlist:
                        elem = elem.split('&')
                        if elem[0] == type:
                            y.append(int(elem[1]))
            f.seek(0)
    return y

#trace 1, soft drugs
y = builder('soft drug')
bartrace1 = Bar(
    x=topNations,
    y=y,
    name='Soft Drugs',
    base=0,
    marker=dict(
        color='rgb(186,58,51)'
    )
)
y = builder('heavy drug')
bartrace2 = Bar(
    x=topNations,
    y=y,
    name='Heavy Drugs',
    base=0,
    marker=dict(
        color='green'
    )
)
y = builder('synthetic drug')
bartrace3 = Bar(
    x=topNations,
    y=y,
    name='Synthetic Drugs',
    base=0,
    marker=dict(
        color='orange'
    )
)
y = builder('medical')
bartrace4 = Bar(
    x=topNations,
    y=y,
    name='Medical Supplies',
    base=0,
    marker=dict(
        color='red'
    )
)
y = builder('watches\clothes')
bartrace5 = Bar(
    x=topNations,
    y=y,
    name='Counterfeits watches\clothes',
    base=0,
    marker=dict(
        color='pink'
    )
)
y = builder('leaked account-electronic goods')
bartrace6 = Bar(
    x=topNations,
    y=y,
    name='Leaked accounts/electronic goods',
    base=0,
    marker=dict(
        color='blue'
    )
)
y = builder('carding')
bartrace7 = Bar(
    x=topNations,
    y=y,
    name='Carding fraud',
    base=0,
    marker=dict(
        color='yellow'
    )
)
y = builder('documents')
bartrace8 = Bar(
    x=topNations,
    y=y,
    name='Fake documents',
    base=0,
    marker=dict(
        color='purple'
    )
)
y = builder('weapons')
bartrace9 = Bar(
    x=topNations,
    y=y,
    name='Weapons',
    base=0,
    marker=dict(
        color='black'
    )
)
y = builder('other')
bartrace10 = Bar(
    x=topNations,
    y=y,
    name='Other',
    base=0,
    marker=dict(
        color='grey'
    )

)
bardata = [bartrace1, bartrace2, bartrace3, bartrace4, bartrace5, bartrace6, bartrace7, bartrace8, bartrace9, bartrace10]
barlayout = Layout(
    barmode='stacked',
    title='Details of sold items for top 7 nations (and Italy)'
)

fig = Figure(data=bardata, layout=barlayout)
plotly.offline.plot(fig, filename='stacked-bar.html')