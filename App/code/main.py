# Import packages
from dash import Dash, html, callback, Output, Input, State, dcc
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import plotly.express as px
import dash_bootstrap_components as dbc
import pickle
import warnings
warnings.simplefilter("ignore")

# Import csv file
df = pd.read_csv("/root/code/Cars.csv")

# Split mileage, max_power into value and number
df[["mileage_value","mileage_unit"]] = df["mileage"].str.split(pat=' ', expand = True)
df[["max_power_value","max_power_unit"]] = df["max_power"].str.split(pat=' ', expand = True)
df.drop(["mileage","max_power"], axis=1, inplace=True)

# Filter dataframe not to include LPG and CNG in fuel column
df = df.loc[(df["fuel"] != 'LPG') & (df["fuel"] != 'CNG')]

# convert mileage, max_power from string to float64
df[["mileage" ,"max_power"]] = df[["mileage_value","max_power_value"]].astype('float64')
df.drop(["mileage_value","max_power_value",
        "mileage_unit","max_power_unit"], axis=1, inplace = True)

# Dicard dataframe containing test drive car in owner column
df = df[df["owner"] != 'Test Drive Car']

# Prepare chosen features
df["log_km_driven"] = np.log(df["km_driven"])
features = df[["max_power","mileage","log_km_driven"]]

# Fill in missing data in features
features_max_power_median = features['max_power'].median()
features_mileage_mean = features['mileage'].mean()
features['max_power'].fillna(features_max_power_median, inplace=True)
features['mileage'].fillna(features_mileage_mean, inplace=True)

# Prapare pattern of transformation for dataset
scaler = StandardScaler()
parabola = PolynomialFeatures(degree = 2, include_bias=True)
load_model = pickle.load(open("/root/code/model.pkl", 'rb')) # Import new model


# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.JOURNAL]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Create app layout
app.layout = html.Div([
        html.H1("Selling car price prediction by new linear regression"),
        html.Br(),
        html.H3("Welcome to second car prediction website."),
        html.Br(),
        html.H6("This is where you can estimate car price by putting numbers in parameters. Car price prediction depends on three features, including to"),
        html.H6("maximum power, mileage and kilometers driven. Firstly, you have to fill at least one input boxes, and then click submit to get result (like previous model)."),
        html.H6("below the submit button. Please make sure that filled number are not negative (like previous model)."),
        html.Br(),
        html.H4("Definition"),
        html.Br(),
        html.H6("Maximum power: Maximum power of a car in bhp"),
        html.H6("Mileage: The fuel efficieny of a car or ratio of distance which a car could move per unit of fuel consumption measuring in km/l"),
        html.H6("Kilometers driven: Total distance driven in a car by previous owner in km"),
        html.Br(),
        html.Div(["Maximum power",dbc.Input(id = "max_power", type = 'number', min = 0, placeholder="please insert"),
        dbc.FormText("Please do not put nagative numbers.",color="secondary"), html.Br()]),
        html.Div(["Mileage", dbc.Input(id = "mileage", min = 0, type = 'number', placeholder ="please insert"),
        dbc.FormText("Please do not put nagative numbers.",color="secondary"), html.Br()]),
        html.Div(["Kilometers driven", dbc.Input(id = "km_driven", type = 'number', min = 0, placeholder="please insert"),
        dbc.FormText("Please do not put nagative numbers.",color="secondary"), html.Br()]),
        dbc.Button(id="submit", children="submit", color="success", className="me-1"),
        html.Div(id="output")
])
# Callback input and output
@callback(
    Output(component_id = "output", component_property = "children"),
    State(component_id = "max_power", component_property = "value"),
    State(component_id = "mileage", component_property = "value"),
    State(component_id = "km_driven", component_property = "value"),
    Input(component_id = "submit", component_property = "n_clicks"),
    prevent_initial_call=True
)

# Function for finding estimated car price
def prediction (max_power, mileage, km_driven, submit):
    if max_power == None:
        max_power = features_max_power_median # Fill in maximum power if dosen't been inserted
    if mileage == None:
        mileage = features_mileage_mean # Fill in mileage if dosen't been inserted
    if km_driven == None:
        km_driven = math.exp(features["km_driven"].median()) # Fill in kilometers driven if doesn't been inserted
    sample = [[max_power, mileage, math.log(km_driven)]]
    scaler.fit(features) #make standard scale for dataset
    sample = scaler.transform(sample) # transform standard scal for samples
    sample = parabola.fit_transform(sample)
    result = np.exp(load_model.predict(sample)) #Predict price
    return f"Your prediction car price is {np.round(result, decimals = 0)[0]}"

if __name__ == '__main__':
    app.run(debug = True)