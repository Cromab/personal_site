import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime as dt
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


#Squirrel page configuration
st.set_page_config(page_title="Scurry, scurry", page_icon=":chipmunk:", layout="wide")

#Use local css
def local_css(file_name):
	with open(file_name) as f:
		st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")


#---- Introduction ----#
with st.container():
    st.title("Squirrels of New York")
    st.subheader("Starring Daniel Day Lewis and Leonardo Dicaprio as 22F-PM-1014-06 and 23C-PM-1014-03 in 'Scurries of New York'")
    st.write("#")
    st.markdown("""
             The Squirrel Census. Truly, one of the time honored traditions of the modern era. Central Park is overrun with the Eastern grey (:gray[*Sciurus carolinensis*]), and where would we be without the [brave souls](https://www.thesquirrelcensus.com/) who risk their lives and nuts to bring us this info?
             Well, for a start we wouldn't be able to make bold claims about the lives and stories of the squirrels of Central Park, and I think the world would be a much darker place. 
             I mean, haven't you ever wondered if a squirrels were in gangs, or a **scurry**, if you will (look it up). What about if squirrels communicate in response to predators to warn their compatriots? Are they as territorial as I am over the window seat of the airplane?  \
             
             
             We can answer these questions (maybe), and more!
             """)
    st.write("---")


#---- The Data ----#
st.subheader('The Data')
st.write('#')
st.markdown("""
            Let's consider the actual dataset in question first. We'll be using the 2018 census data, which includes a robust set of over 3000 squirrel observations.
            That chunky squirrel set is below, but there's a little bit of info we should preface this beast with. \
            
            
            Each squirrel sighting was recorded with the latitude, longitude they were spotted at,
            and they're given a unique identifier that's a concatenation of their hectare, time they were spotted, and the date spotted.
            The first question you might ask is...what's a hectare (but totally not me, I obviously knew what that was). Simple answer:
            a hectare is 100 meter by 100 meter square field of land. In this case, it's used to divide Central Park into more identifiable chunks of land.
            Beyond this, there's quite a lot of neat stuff in this dataset; there's the activity they were spotted doing, how moan-y they were feeling, 
            if their tails were a-twitchin', and even if they were indifferent to the pitiful lives of man!
            """)

#Reading in data and converting datetime object to string in 'Date' Column
squirrels = pd.read_csv('./data/2018_Central_Park_Squirrel_Census_-_Squirrel_Data.csv')
squirrels['Date'] = pd.to_datetime(squirrels['Date'], format='%m%d%Y').apply(lambda x: x.strftime('%m-%d-%Y'))
#Display dataframe in webpage
st.dataframe(
    squirrels, 
    column_config={
        "X":"Lat",
        "Y":"Lon"
    },
    hide_index=True
)

st.write("#")
st.markdown("""
            This data is all well and good, but we have longitudes and latitudes here, so why don't we try and make a map of the squirrels relative positions in Central Park?
            It would be rude not to, really.
            """)
st.write("#")

#Create altair chart to show map of squirrel locations, coloring datapoints by squirrel fur
squirrel_map = alt.Chart(squirrels).mark_circle(size=30).encode(
    y=alt.X('X', title="Longitude", scale=alt.Scale(domain=[-73.9814, -73.9484])),
    x=alt.Y('Y', title="Latitude", scale=alt.Scale(domain=[40.7640, 40.801]), axis=alt.Axis(labelBound=True)),
    color=alt.Color('Primary Fur Color', scale=alt.Scale(range=['white', 'black', 'pink', 'gray'])),
    tooltip=['Unique Squirrel ID', 'Age', 'Indifferent']
).interactive()

st.altair_chart(squirrel_map, use_container_width=True, theme=None)

st.write("#")
st.markdown("""
            The poor NULL squirrels don't even stand a chance against the grays. But honestly, this is just a graph of our latitude and longitude. 
            Let's make it a little nicer to look at and may sure our gray, cinnamon, black, and NULL squirrels are colored appropriately.
            """)
st.write("#")


#---- Geographic Mapping of Sightings ----#
#Associate Hex Colors with squirrel coats
squirrels['Hex_Color'] = ''
squirrels['Hex_Color'].loc[squirrels['Primary Fur Color'] == 'Gray'] = '#919191'
squirrels['Hex_Color'].loc[squirrels['Primary Fur Color'] == 'Black'] = '#000000'
squirrels['Hex_Color'].loc[squirrels['Primary Fur Color'] == 'Cinnamon'] = '#D27D2D'
squirrels['Hex_Color'].loc[(squirrels['Primary Fur Color'] != 'Gray') &
                           (squirrels['Primary Fur Color'] != 'Black') &
                           (squirrels['Primary Fur Color'] != 'Cinnamon')] = '#FFFFFF'

#Map squirrels using MapBox API
st.map(squirrels,
       latitude='Y',
       longitude='X', 
       use_container_width=True,
       size=10,
       color='Hex_Color')
#squirrels.drop(columns='Hex_Color', inplace=True)


#--- Filterable Map ---#
st.markdown("""
            This is a lot nicer to look at and more descriptive too, and probably a good place for us to start building some visualization from.
            Let's start by giving some basic filters and see if maybe these give some insight into our sightings and maybe raise new questions.
            We'll actually move back to the basic map we had earlier for this, as there is currently an ongoing bug with the functionality used
            to create our far nicer map earlier. We don't really need the street view to illustrate the point.
            """)

#Map Filters
filters = pd.Series(index=['Apathetic', "Young'un", 'Noisy', 'Foodie' ], 
                    data=[st.checkbox("Apathetic"), st.checkbox("Young'un"), st.checkbox("Noisy"),st.checkbox("Foodie")])

#Helper Function to take filters
@st.cache
def squirrel_filterer(filters=filters):
    #determine apathy of the squirrel
    if filters['Apathetic']:
        filtered_squirrels = squirrels.loc[squirrels['Indifferent'] == True]
    else:
        filtered_squirrels = squirrels.loc[squirrels['Indifferent'] == False]
    #Determine if a juvenile
    if filters["Young'un"]:
        filtered_squirrels_depth2 = filtered_squirrels.loc[filtered_squirrels["Age"] == 'Juvenile']
    else:
        filtered_squirrels_depth2 = filtered_squirrels
    #Determine if it's a noisy squirrel
    if filters['Noisy']:
        filtered_squirrels_depth3 = filtered_squirrels_depth2.loc[(filtered_squirrels_depth2['Kuks'] == True) |
                                                                (filtered_squirrels_depth2['Quaas'] == True) |
                                                                (filtered_squirrels_depth2['Moans'] == True)]
    else:
        filtered_squirrels_depth3 = filtered_squirrels_depth2
    #Determine if it's a food motivated squirrel
    if filters['Foodie']:
        filtered_squirrels_depth4 = filtered_squirrels_depth3.loc[(filtered_squirrels_depth3['Eating'] == True) |
                                                                  (filtered_squirrels_depth3['Foraging'] == True)]
    else:
        filtered_squirrels_depth4 = filtered_squirrels_depth3
    return filtered_squirrels_depth4

#Call function and show map
filtered_squirrels = squirrel_filterer()
filtered_squirrel_map = alt.Chart(filtered_squirrels).mark_circle(size=30).encode(
    y=alt.X('X', title="Longitude", scale=alt.Scale(domain=[-73.9814, -73.9484])),
    x=alt.Y('Y', title="Latitude", scale=alt.Scale(domain=[40.7640, 40.801]), axis=alt.Axis(labelBound=True)),
    color=alt.Color('Primary Fur Color', scale=alt.Scale(domain=[None, 'Black', 'Cinnamon', 'Gray'], range=['#FFFFFF', '#000000', '#D27D2D', '#919191'])),
    tooltip=['Unique Squirrel ID', 'Age', 'Hectare']
).interactive()

st.altair_chart(filtered_squirrel_map, use_container_width=True, theme=None)



#Explain meaning behind mapping and introduce further investigation
st.write('##')
st.markdown("""
            Looking at this mapping, it's still pretty difficult to make broad claims about any squirrels based on age, moans, food foraging, or their
            bias against census takers. Reading into this map we might be of the impression that squirrels who are juveniles are less noisy, but we should recall that 
            we have far less juvenile sightings overall. Proportionally, the noisy ones aren't that far apart:
            """)
with st.container():
    st.write("Adults: \t", 
             str(round(100*len(squirrels.loc[((squirrels['Kuks'] == True) |(squirrels['Quaas'] == True) |(squirrels['Moans'] == True)) & (squirrels['Age']=='Adult')]) / len(squirrels.loc[squirrels['Age']=='Adult']), 2)),
             "%")
    st.write("Juveniles: \t", 
             str(round(100*len(squirrels.loc[((squirrels['Kuks'] == True) |(squirrels['Quaas'] == True) |(squirrels['Moans'] == True)) & (squirrels['Age']=='Juvenile')]) / len(squirrels.loc[squirrels['Age']=='Juvenile']), 2)),
             "%")


#--- Machine Learning to Predict the noisy Squirrels ---#
st.markdown("""
            Regardless, perhaps we can still make predictions about how noisy our little squirrels can be, and whether this has a pattern with longitude, latitude, hectares, etc.
            We want to know why these squirrels are so dang noisy or so dang quiet. Our census takers won't be satisfied otherwise!
            And that's where we'll bring in some machine learning concepts to help us along...is what I would say, but first let's see if the noisy ones tend to have
            any special commentary. It could be that we just have more dedicated census takers, and that a lot of census takers just didn't bother to record noises.
            Let's see the general proportion here of additional notes vs noises records.
            """)

#Create a response column for Noise
squirrels['Noise'] = 0
squirrels.loc[(squirrels['Kuks'] != False) | (squirrels['Moans'] != False) | (squirrels['Quaas'] != False), 'Noise'] = 1
squirrel_noise = squirrels.drop(columns=['Kuks', 'Moans', 'Quaas'])
st.write("Total Proportion of Noise recordings including other interactions:",
         len(squirrel_noise[squirrel_noise['Noise'] > 0].dropna(subset=['Other Interactions']))/len(squirrel_noise[squirrel_noise['Noise'] > 0]))

st.write("Not bad! Looks like we have a solid distribution of commentary on squirrel interactions instead of a few zealous census takers. Now for some general data transformation and reformatting and cleaning to get this dataset ready for action.")

#Dataset alterations
with st.container():
    #Conform Booleans to Ints
    squirrel_noise[['Running', 'Chasing', 'Climbing', 'Eating', 'Foraging', 'Tail flags', 'Tail twitches', 'Approaches', 'Indifferent', 'Runs from']] = squirrel_noise[['Running', 'Chasing', 'Climbing', 'Eating', 'Foraging', 'Tail flags', 'Tail twitches', 'Approaches', 'Indifferent', 'Runs from']].astype(int)
    squirrel_noise.loc[squirrel_noise['Shift'] == 'AM', 'Shift'] = 0
    squirrel_noise.loc[squirrel_noise['Shift'] == 'PM', 'Shift'] = 1
    #Impute some columns and drop Frequently NA columns
    squirrel_noise['Age'].fillna('Adult', inplace=True)
    squirrel_noise.loc[squirrel_noise['Age'] == 'Juvenile', 'Age'] = 0
    squirrel_noise.loc[squirrel_noise['Age'] == 'Adult', 'Age'] = 1
    squirrel_noise.loc[squirrel_noise['Age'] == '?', 'Age'] = 1
    squirrel_noise['Primary Fur Color'].fillna('Gray', inplace=True)
    squirrel_noise = squirrel_noise.drop(columns=['Unique Squirrel ID', 'Highlight Fur Color', 'Combination of Primary and Highlight Color', 'Color notes', 'Location', 'Above Ground Sighter Measurement', 'Specific Location', 'Other Activities', 'Other Interactions', 'Lat/Long', 'Hex_Color', 'Date', 'Hectare'])
    dummy_fur = pd.get_dummies(squirrel_noise['Primary Fur Color'])
    squirrel_noise = pd.concat([squirrel_noise, dummy_fur], axis=1)
    squirrel_noise = squirrel_noise.drop(columns=['Primary Fur Color'])
    squirrel_noise = squirrel_noise[["X","Y","Shift","Hectare Squirrel Number","Age","Black","Cinnamon",
                                     "Gray","Running","Chasing","Climbing","Eating","Foraging","Tail flags","Tail twitches","Approaches","Indifferent","Runs from","Noise"]]
    st.dataframe(squirrel_noise)
    
#Address Categorical vs numerical
st.markdown("""
            The shrewd or experienced among you may have noticed that we have quite a bit of categorical and numerical data mixed together,
            even some date objects. We aren't left with any null values in our remaining data, and just have to worry about preprocessing for these.
            After that we can talk about methods we might use to predict if a squirrel will be noisy. Importantly, our data is very unbalanced, with
            their being far more silent squirrels recorded than noisy ones. We'll talk about the problems that could cause in a moment.
            #
            First lets talk about the kind of model that we want to use. Here, we're dealing with a classification problem where we want to identify
            whether a squirrel is noisy or not. Popular (and straightforward) classification model include algorithms such as Support Vector Machines
            and Naive Bayes, but we're actually going to go with a popular ensemble method: Random forests. Why this choice? Partly because they are
            good classifiers and partly because we aren't as interested in **what** predicts a squirrel will be noisy, but rather if they are noisy.
            """)
st.write('---')
st.title("Come on Feel the Noize")

with st.container():
    code = """
    #Instantiate model and its hyperparameters and split data
    Xtrain, Xtest, ytrain, ytest = train_test_split(squirrel_noise.iloc[:, :-1], squirrel_noise.iloc[:, -1], random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    #Fit model to data
    model.fit(Xtrain, ytrain)
    #Predict on new data
    y_model = model.predict(Xtest)
    """
    st.code(code, language='python')
    
#Instantiate model and its hyperparameters and split data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(squirrel_noise.iloc[:, :-1], squirrel_noise.iloc[:, -1], random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
#Fit model to data
model.fit(Xtrain, ytrain)
#Predict on new data
y_model = model.predict(Xtest)

#Display accuracy score
from sklearn.metrics import accuracy_score
st.write(f'A quick accuracy score shows: {round(accuracy_score(ytest, y_model)*100, 2)}% of the testing data was correctly predicted.')

#Creating a confusion matrix
st.markdown(":red[Wow]. That seems like a really solid level of accuracy on this basic classification with no extra frills. But let's make a confusion matrix and see what's going on...")
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, y_model)
#Seaborn plotting confusion matrix
fig = plt.figure()
sns.heatmap(mat, square=True, annot=True, cbar=False, fmt='g')
plt.xlabel('predicted value')
plt.ylabel('true value')
st.pyplot(fig, use_container_width=False)

st.markdown("""
            That looks much worse now. We predicted almost all of our values would be 0, i.e. not a noisy squirrel.
            Of our 46 Noisy squirrels in the test set, we misclassified 43 of them. **:red[Atrocious]**. What happened?
            
            Let's recall earlier we mentioned the percentage of Juvenile and Adult squirrels who were noisy were 5.15 and 4.13%, respectively.
            Well, all squirrels are Adults or Juveniles, so our max proportion of the dataset which represents noisy squirrels is in the interval (.0413, .0515).
            That's probably less than 5% of the entire dataset...our predictor is a good predictor simply because the overwhelming majority of squirrels
            aren't noisy. However, this doesn't help us, because we specifically want to find the squirrels that *are* noisy.

            So, what do we do? Here is where techniques such as undersampling and oversampling come in handy, i.e. resampling our data to underrepresent a majority class
            or overrepresent a minority class. In our case, given the overwhelming amount of data, we will use undersampling techniques.
            We'll start with a Random Resampling targeting subsets of our data selected randomly. Let's see how that adjusts our model and matrix.
            """)


#--- Adjust ML Model ---#
#Use a RandomUnderSampler
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(Xtrain, ytrain)

#Fit model to data
model.fit(X_resampled, y_resampled)
#Predict on new data
y_model = model.predict(Xtest)

#Display accuracy score
st.write(f'A quick accuracy score shows: {round(accuracy_score(ytest, y_model)*100, 2)}% of the testing data was correctly predicted.')

#Creating a confusion matrix
st.markdown("The initial reaction might be, 'But that's a worse accuracy than before'. You would be right, but before we write it off, let's see our confusion matrix.")
mat = confusion_matrix(ytest, y_model)
#Seaborn plotting confusion matrix
fig = plt.figure()
sns.heatmap(mat, square=True, annot=True, cbar=False, fmt='g')
plt.xlabel('predicted value')
plt.ylabel('true value')
st.pyplot(fig, use_container_width=False)


#--- Closing Statements ---#
st.markdown("""
            Well, what's the lesson here? Is this a better model? The answer, as with all things, is 'depends, I guess.' 
            In my hypothetical world, I care a lot more about finding a squirrel who is noisy than I do just predicting that none of them are.
            So while the accuracy is far worse and our number of false positive is far greater, this fits the criteria I set forward, and serves as a far
            better predictor of whether a squirrel will be noisy.
            
            Now, that doesn't mean I would deploy this model in a professional scenario, as it would require far more fine-tuning and the potential
            use of a different algorithm if those involved had more questions regarding things such as the cause of a noisy squirrel, but as an illustration
            of the risks of blindly trusting an algorithm just because it works on our dataset, it's a great example. Our accuracy was superb in our untweaked model
            because our data was so heavily biased. This is the truth with quite a bit of data. Most of the most popular ML algorithms will favor a majority class.
            If we just trusted the original iteration of the algorithm, we would never even answer our core question: "What squirrels are noisy little dudes?"
            """)
