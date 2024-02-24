import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import calendar
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler 
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objs as go
import plotly.offline as pyo

# load read data
df_hepatitis = pd.read_csv('/Users/harish/Documents/under-the-weather/public/api/Data_Sources/hepatitis.csv')
df_measles = pd.read_csv('/Users/harish/Documents/under-the-weather/public/api/Data_Sources/measles.csv')
df_mumps = pd.read_csv('/Users/harish/Documents/under-the-weather/public/api/Data_Sources/mumps.csv')
df_pertussis = pd.read_csv('/Users/harish/Documents/under-the-weather/public/api/Data_Sources/pertussis.csv')
df_rubella = pd.read_csv('/Users/harish/Documents/under-the-weather/public/api/Data_Sources/rubella.csv')
df_smallpox = pd.read_csv('/Users/harish/Documents/under-the-weather/public/api/Data_Sources/smallpox.csv')

# Taking care of data discrepancies 
dfs = [df_hepatitis, df_measles, df_mumps, df_pertussis, df_rubella, df_smallpox]

for i, df in enumerate(dfs):
    max_cases = df['cases'].max()  
    dfs[i] = df[df['cases'] != max_cases]  

class DiseasePredictor(nn.Module):
    def __init__(self, input_dim):
        super(DiseasePredictor, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Preprocessing function to normalize the features
def preprocess_data(df):
    scaler = MinMaxScaler()
    df[['week', 'incidence_per_capita', 'cases']] = scaler.fit_transform(df[['week', 'incidence_per_capita', 'cases']])
    return df

# Loop through each DataFrame
disease_dfs = {
    'Hepatitis': df_hepatitis,
    'Measles': df_measles,
    'Mumps': df_mumps,
    'Pertussis': df_pertussis,
    'Rubella': df_rubella,
    'Smallpox': df_smallpox
}

disease_models = {}
for disease in disease_dfs.keys():
    model = DiseasePredictor(input_dim=3)  # Change input_dim to 3 since there are now 3 features
    model.load_state_dict(torch.load("/Users/harish/Documents/under-the-weather/public/api/" + f'{disease.lower()}_model.pth'))
    model.eval()
    disease_models[disease] = model

# Create tables for each disease
predicted_tables = {}
for disease, model in disease_models.items():
    # Create a DataFrame to store predicted cases
    predicted_df = pd.DataFrame(columns=['state', 'predicted_cases'])
    states = disease_dfs[disease]['state'].unique()  # Get unique states
    for state in states:
        # Prepare input tensor for prediction
        
        state_df = disease_dfs[disease][disease_dfs[disease]['state'] == state] 
        
        if disease == 'Hepatitis':
            state_df *= 100
        if disease == 'Measles':
            state_df *= 100
        if disease == 'Mumps':
            state_df *= 10
        if disease == 'Pertussis':
            state_df *= 100
        features_tensor = torch.tensor(state_df[['week', 'incidence_per_capita', 'cases']].values, dtype=torch.float32)
        
        # Make predictions
        with torch.no_grad():
            predictions = model(features_tensor).numpy()
            
        predictions = np.abs(predictions)
        
        # Append predicted cases to DataFrame
        predicted_df = predicted_df._append({'state': state, 'predicted_cases': predictions.mean()}, ignore_index=True)
    
    # Sort DataFrame by state for better readability
    predicted_df.sort_values(by='state', inplace=True)
    
    # Store the predicted table for the disease
    predicted_tables[disease] = predicted_df

# Display tables
for disease, table in predicted_tables.items():
    print(f"\n{disease} Predicted Cases Table:")
    print(table)
    

    
def create_heatmap_for_state(df, disease_name, state):
    # Filter DataFrame for the specified state
    state_df = df[df['state'] == state]
    
    # Create the heatmap
    fig = go.Figure(data=go.Choropleth(
        locations=state_df['state'],  # Spatial coordinates (should be just one state)
        z=state_df['predicted_cases'].astype(float),  # Data to be color-coded
        locationmode='USA-states',  # set of locations match entries in `locations`
        colorscale='Reds',
        colorbar_title="Predicted Cases",
    ))

    fig.update_layout(
        title_text=f'Predicted {disease_name} Cases in {state}',
        geo_scope='usa',  # limit map scope to USA
    )

    # Save the plot as an HTML file
    filename = f'heatmap_{disease_name.lower()}_{state.lower()}.html'
    pyo.plot(fig, filename=filename)


def generate_heatmaps_for_state(selectedState):
    for disease_name, df in predicted_tables.items():
        create_heatmap_for_state(df, disease_name, selectedState)
        
    
