import sqlite3 as sql
from os.path import dirname, join

from datetime import date, datetime

import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Div, Select, DateRangeSlider, DatePicker, TextInput
from bokeh.plotting import figure

def normalise_z(df):
    '''Performs Z-score normalisation on specified dataframe'''
    mean = df.mean()
    std = df.std()
    return (df-mean)/std

def prepare_features(df_features):
    '''
    1. Converts df_features into a 2d numpy array
    2. Adds a column of ones to df_features
    3. Returns df_features
    '''
    ones = np.ones((df_features.shape[0], 1))

    X = np.concatenate((ones, df_features.to_numpy()), axis=1)
    
    return X

# Get predicted values, y, based on current beta
def predict(df_features, beta):
    '''Obtains predicted target values via linear regression algorithm'''
    df_features_norm = normalise_z(df_features)
    
    array_2d = prepare_features(df_features_norm)
    
    return np.matmul(array_2d, beta)

def select_covid_data():
    '''Returns a df'''

    global cur_model_map_val
    global l
    global location
    global covid_data
    global pred_data
    global x_axis
    global y_axis

    # Check if input title == prev input's title
    if model.value != model_map[cur_model_map_val]["title"]:
        # If different, change cur_model_map_val to key of model.value
        cur_model_map_val = list(model_map.keys())[titles.index(model.value)]
        
        if cur_model_map_val == 1:

            covid_data = pd.read_sql("select * from owid_one", conn)
            pred_data = pd.read_sql("select * from pred_one", conn)
        else:
            covid_data = pd.read_sql("select * from owid_two", conn)
            pred_data = pd.read_sql("select * from pred_two", conn)

        # covid_data["color"] = np.where(covid_data["new_deaths_smoothed_per_million"] > 20, "red", "grey")
        # covid_data["alpha"] = np.where(covid_data["new_deaths_smoothed_per_million"] > 0, 0.9, 0.25)
        
        covid_data["color"] = "grey"
        covid_data["alpha"] = 0.9
        pred_data["color"] = "orange"
        pred_data["alpha"] = 0.9


        all_var = covid_data.drop(["index", "alpha", "color", "location", "date"], axis=1).columns.tolist()
        locations = ["All"] + covid_data["location"].unique().tolist()


        # Dynamically change options and values for location, x_axis, y_axis
        location.options=locations
        location.value = "All"
        
        x_axis.options = model_map[cur_model_map_val]["features"]
        x_axis.value = model_map[cur_model_map_val]["features"][0]

        y_axis.options = model_map[cur_model_map_val]["target"]
        y_axis.value = model_map[cur_model_map_val]["target"][0]

        predicted_value.title = model_map[cur_model_map_val]["target"][0]


        
        

    location_val = location.value
    start, end = date_range_slider.value
    d = date_picker.value
   
    
    selected = covid_data
    predicted = pred_data

    query = pred_data.loc[pd.to_datetime(pred_data.date) == d]

    val = query.loc[:, y_axis.value].iloc[0]

    if isinstance(val, float):
        predicted_value.value = f"{val}"
    else:
        predicted_value.value = "Date does not exist"


   


    if (isinstance(start, int) and isinstance(end, int)):
        start = datetime.fromtimestamp(start / 1000)
        end = datetime.fromtimestamp(end / 1000)

    
        selected_date = pd.to_datetime(covid_data.date)
        predicted_date = pd.to_datetime(pred_data.date)


        selected = covid_data[
            (selected_date >= start) &
            (selected_date <= end) 
        ]

        predicted = pred_data[
            (predicted_date >= start) &
            (predicted_date <= end) 
        ]

         

    if (location_val != "All"):
        selected = selected[selected.location.str.contains(location_val)==True]
        predicted = predicted[predicted.location.str.contains(location_val)==True]


    # Selected is of type df
    return selected, predicted


def update():

    df, df_pred = select_covid_data()

    x_name = x_axis.value
    y_name = y_axis.value


    p.xaxis.axis_label = x_axis.value
    p.yaxis.axis_label = y_axis.value
    p.title.text = f"{len(df)} / {len(covid_data)} data points selected"

    source.data = dict(
        x=df[x_name],
        y=df[y_name],
        color=df["color"],
        location=df["location"],
        date=df["date"],
        vaccination=df["people_vaccinated_per_hundred"],
        deaths=df["new_deaths_smoothed_per_million"],
        alpha=df["alpha"],
    )

    source_pred.data = dict(
        x=df_pred[x_name],
        y=df_pred[y_name],
        location=df_pred["location"],
        date=df_pred["date"],
        vaccination=df_pred["people_vaccinated_per_hundred"],
        deaths=df_pred["new_deaths_smoothed_per_million"],
    )


# Initialise GLOBAL variables
conn = sql.connect("owid/covid19.db")


covid_data = pd.read_sql("select * from owid_one", conn)
pred_data = pd.read_sql("select * from pred_one", conn)

# covid_data["color"] = np.where(covid_data["people_vaccinated_per_hundred"] < 10, "red", "grey")
# covid_data["alpha"] = np.where(covid_data["new_deaths_smoothed_per_million"] > 0, 0.9, 0.25)
covid_data["color"] = "grey"
covid_data["alpha"] = 0.9

cur_model_map_val = 1

model_map = {
    1: {
        "title": "Task 1 - Predict new_deaths_smoothed_per_million",
        "beta": [[ 2.74209905],
            [ 1.36816698],
            [-0.20810756],
            [ 0.2183362 ],
            [ 1.03668009],
            [-0.715664  ],
            [ 0.47310631]],
        "features": ['icu_patients_per_million',
            'people_vaccinated_per_hundred',
            'stringency_index',
            'icu_patients_per_million_squared',
            'people_vaccinated_per_hundred_square_root',
            'people_vaccinated_per_hundred_squared'],
        "target": ["new_deaths_smoothed_per_million"],
    },
    2: {
        "title": "Task 2 - Predict hosp_patients_per_million",
        "beta": [[153.12707804],
       [ 26.34445848],
       [-36.36395753],
       [-28.78560888],
       [ 12.35167325],
       [  2.83655482],
       [ 19.74953453]],
        "features": ['new_deaths_smoothed_per_million',
            'people_vaccinated_per_hundred',
            'human_development_index',
            'hospital_beds_per_million',
            'people_vaccinated_per_hundred_squared',
            'people_vaccinated_per_hundred_cubed'],
        "target": ["hosp_patients_per_million"],
    }
}


locations = ["All"] + covid_data["location"].unique().tolist()

titles = [model_map[k]["title"] for k, v in model_map.items()]
model_options = [model_map[k]["title"] for k, v in model_map.items()]

# Read HTML
desc = Div(text=open(join(dirname(__file__), "description.html")).read(), sizing_mode="stretch_width")



# Create Input controls
model = Select(title="Model", options=sorted(model_options), value=model_map[cur_model_map_val]["title"])
location = Select(title="Location", value="All",
            options=locations)
x_axis = Select(title="X Axis", options=model_map[cur_model_map_val]["features"], value=model_map[cur_model_map_val]["features"][0])
y_axis = Select(title="Y Axis", options=model_map[cur_model_map_val]["target"], value=model_map[cur_model_map_val]["target"][0], disabled=True)
date_range_slider = DateRangeSlider(value=(date(2020, 1, 1), date(2021, 11, 10)),
                                    start=date(2020, 1, 1), end=date(2021, 11, 10))
date_picker = DatePicker(title='Select date', value="2021-10-10", min_date="2020-01-01", max_date="2021-11-10")
predicted_value = TextInput(title=model_map[cur_model_map_val]["target"][0], value="Hello", disabled=True)



controls = [model, location, x_axis, y_axis, date_range_slider, date_picker, predicted_value]

# Update if control values change
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

# Displays controls
inputs = column(*controls, width=320)

# Create Column Data Source that will be used by the plot
source = ColumnDataSource(data=dict(x=[], y=[], color=[], location=[], date=[], vaccination=[], deaths=[], alpha=[]))
source_pred = ColumnDataSource(data=dict(x=[], y=[], location=[], date=[], vaccination=[], deaths=[]))

TOOLTIPS=[
    ("Location", "@location"),
    ("Date", "@date"),
    ("Vaccination rate", "@vaccination"),
    ("Daily deaths", "@deaths"),
]

p = figure(tools="tap", height=600, width=700, title="", toolbar_location=None, tooltips=TOOLTIPS, sizing_mode="scale_both")
p.circle(x="x", y="y", source=source, size=7, color="color", line_color=None, fill_alpha="alpha", legend="Actual")

p.circle(x="x", y="y", color="orange", source=source_pred, legend="Predicted")


l = column(desc, row(inputs, p), sizing_mode="scale_both")



curdoc().add_root(l)
curdoc().title = "OWID COVID-19 Linear Regression"


# Initial load of data
update()
    


