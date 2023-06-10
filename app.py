import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
# Import relevant libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import socket
import paho.mqtt.client as mqtt

# MQTT broker details
broker = "agbc-fe.pdn.ac.lk"
port = 1883
username = "e18-team"
password = "pera@e18"


base_url = "http://agbc-fe.pdn.ac.lk/api/v1/data/?sensor=10008&date="

start_date = pd.to_datetime("2020-10-22")
end_date = pd.to_datetime("2020-12-30")

date_range = pd.date_range(start=start_date, end=end_date, freq="D")

all_data = []

def fetch_data(date):
    date_str = date.strftime("%Y-%m-%d")
    url = base_url + date_str

    try:
        response = requests.get(url)
        data = response.json()
        return data['data']
    except:
        print(f"Error: Could not retrieve data for date {date_str}")
        return []

start_time = time.time()  # Get the current time before starting the execution


# Create a ThreadPoolExecutor with the maximum number of workers
executor = ThreadPoolExecutor(max_workers=None)

# Use tqdm to track the progress
with tqdm(total=len(date_range), desc="Progress", unit="day") as pbar:
    # Submit the fetch_data task to the executor for each date in parallel
    futures = [executor.submit(fetch_data, date) for date in date_range]

    # Retrieve the results from the completed futures
    for future in futures:
        all_data.extend(future.result())
        pbar.update(1)

end_time = time.time()  # Get the current time after finishing the execution
execution_time = end_time - start_time

print("Execution Time:", execution_time, "seconds")

# Create the DataFrame from the collected data
df = pd.DataFrame(all_data, dtype=str)
import numpy as np
# Replace '?' with NaN

df.replace(' ?', np.nan, inplace=True)
# Convert temperature columns to numeric
df['temp1'] = pd.to_numeric(df['temp1'], errors='coerce')
df['temp2'] = pd.to_numeric(df['temp2'], errors='coerce')
df['temp3'] = pd.to_numeric(df['temp3'], errors='coerce')

# Convert temperature columns to numeric
df['humidity1'] = pd.to_numeric(df['humidity1'], errors='coerce')
df['humidity2'] = pd.to_numeric(df['humidity2'], errors='coerce')
df['humidity3'] = pd.to_numeric(df['humidity3'], errors='coerce')

df['seqNo'] = pd.to_numeric(df['seqNo'], errors='coerce')

# Calculate the average temperature
df['average_internal_temp'] = df[['temp1', 'temp2', 'temp3']].mean(axis=1,skipna=True)

# Calculate the average humidity
df['average_internal_humidity'] = df[['humidity1', 'humidity2', 'humidity3']].mean(axis=1,skipna=True)

# Create a new DataFrame with only the desired columns
new_df = df[['seqNo','date','time','average_internal_temp', 'average_internal_humidity', 'light']]

# Combine the 'date' and 'time' columns into a single datetime column
new_df['datetime'] = pd.to_datetime(new_df['date'] + ' ' + new_df['time'])
# Set the 'time' column as the DataFrame index
new_df.set_index('datetime', inplace=True)
new_df.drop(['date', 'time','seqNo'], axis=1, inplace=True)
# Resample the DataFrame using 'H' offset alias and select the first entry from each hour
new_df_hourly = new_df.resample('H').first()
# Load the CSV file into a DataFrame
external_weather = pd.read_csv('ML Part/weather_data.csv')

# Combine the 'Date' and 'Time' columns into a single datetime column
external_weather['datetime'] = pd.to_datetime(external_weather['Date'] + ' ' + external_weather['Time'])

external_weather.drop(["Time","Date"],axis=1,inplace=True)

external_weather.set_index('datetime', inplace=True)

merged_df = pd.merge(external_weather, new_df_hourly, on='datetime')

# Drop rows with any null values
merged_df.dropna(inplace=True)


from sklearn.model_selection import train_test_split

columns_to_drop = ['average_internal_temp', 'average_internal_humidity', 'light', 'Clouds', 'Wind Speed','Description']
X = merged_df.drop(columns_to_drop, axis=1)
print(X.dtypes)
y = merged_df[['average_internal_temp', 'average_internal_humidity', 'light']]

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Splitting dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, )
model = LinearRegression()
# Training the linear regression model
model.fit(X_train, y_train)
model.score(X_test,y_test)
print(model.score(X_test,y_test))

# Create the Dash app
app = dash.Dash(__name__)
server = app.server

# Callback functions
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    # Subscribe to topics after connection is successful
    client.subscribe("v0/controller/1000/blower")
    client.subscribe("v0/controller/1000/mist")
    print("kaarna")

def on_message(client, userdata, msg):
    print("Received message: " + msg.topic + " " + str(msg.payload))

    # Process the received message and take action accordingly
    if msg.topic == "v0/controller/1000/blower":
        if msg.payload == b'1':
            # Code for turning on the blower
            print("Blower turned on")
        elif msg.payload == b'0':
            # Code for turning off the blower
            print("Blower turned off")

    elif msg.topic == "v0/controller/1000/mist":
        if msg.payload == b'1':
            # Code for turning on the mist
            print("Mist turned on")
        elif msg.payload == b'0':
            # Code for turning off the mist
            print("Mist turned off")

# Create MQTT client and set callback functions
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Set username and password for MQTT broker
client.username_pw_set(username, password)

# Connect to MQTT broker
client.connect(broker, port, 60)

# Start the MQTT client loop
client.loop_start()
# Define the layout of the dashboard
app.layout = html.Div(
    children=[
    
    html.H1('Interior temp /humidity predictor'),
    
    html.Div([
        html.H3('Exploratory Data Analysis'),
        html.Label('Feature 1 (X-axis)'),
        dcc.Dropdown(
            id='x_feature',
            options=[{'label': col, 'value': col} for col in merged_df.columns],
            value=merged_df.columns[0]
        )
    ], style={'width': '30%', 'display': 'inline-block'}),
    html.Br(),
    html.Br(),
    html.Div([
        html.Label('Feature 2 (Y-axis)'),
        dcc.Dropdown(
            id='y_feature',
            options=[{'label': col, 'value': col} for col in merged_df.columns],
            value=merged_df.columns[1]
        )
    ], style={'width': '30%', 'display': 'inline-block'}),
    
    dcc.Graph(id='correlation_plot'),
    
    # Wine quality prediction based on input feature values
    html.H3("Internal Temperature Prediction"),
    html.Div(className="form-group", children=[
    html.Label("External Temperature : "),
    dcc.Input(
        id='externalTemperature',
        type='number',
        required=True,
        className="form-control"
    )
]),
html.Br(),
html.Div(className="form-group", children=[
    html.Label("Feels Like : ",style={'margin-bottom': '5px'}),
    dcc.Input(
        id='feelsLike',
        type='number',
        required=True,
        className="form-control",style={'margin-bottom': '5px'}
    )
]),
html.Br(),
html.Div(className="form-group", children=[
    html.Label("Pressure : "),
    dcc.Input(
        id='pressure',
        type='number',
        required=True,
        className="form-control"
    )
]),
html.Br(),
html.Div(className="form-group", children=[
    html.Label("External Humidity : "),
    dcc.Input(
        id='externalHumidity',
        type='number',
        required=True,
        className="form-control"
    )
]),
html.Br(),
html.Div(className="form-group", children=[
    html.Label("Dew Point : "),
    dcc.Input(
        id='dewPoint',
        type='number',
        required=True,
        className="form-control"
    )
]),
html.Br(),


    html.Div([
        html.Button('Predict', id='predict-button', n_clicks=0),
    ]),

    html.Div([
        html.H4("Predicted Quality"),
        html.Div(id='prediction-output')
    ])
])

# Define the callback to update the correlation plot
@app.callback(
    dash.dependencies.Output('correlation_plot', 'figure'),
    [dash.dependencies.Input('x_feature', 'value'),
     dash.dependencies.Input('y_feature', 'value')]
)
def update_correlation_plot(x_feature, y_feature):
    fig = px.scatter(merged_df, x=x_feature, y=y_feature, color='average_internal_temp')
    fig.update_layout(title=f"Correlation between {x_feature} and {y_feature}")
    return fig


# Define the callback function to predict wine quality
@app.callback(
    Output(component_id='prediction-output', component_property='children'),
    [Input('predict-button', 'n_clicks')],
    [State('externalTemperature', 'value'),
     State('feelsLike', 'value'),
     State('pressure', 'value'),
     State('externalHumidity', 'value'),
     State('dewPoint', 'value'),
     ]
)
def predict_quality(n_clicks, externalTemperature, feelsLike, pressure, externalHumidity,
                     dewPoint):
    # Create input features array for prediction
    input_features = np.array([externalTemperature, feelsLike, pressure, externalHumidity, dewPoint, 
                               ]).reshape(1, -1)

    # Predict the wine quality (0 = bad, 1 = good)
    prediction = model.predict(input_features)

    # Return the prediction
    # Return the prediction
    temp_prediction = prediction[0, 0]
    humidity_prediction = prediction[0, 1]
    light_prediction = prediction[0, 2]
    print(temp_prediction)
    
    if temp_prediction > 30:
        client.publish("v0/controller/1000/blower", "1")
        return 'Turn on the fan. Prediction temperature : ' + str(temp_prediction)

    else:
        client.publish("v0/controller/1000/blower", "0")
        return 'Turn off the fan. Prediction temperature: ' + str(temp_prediction)


if __name__ == '__main__':
    app.run_server(debug=False)

