import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import calendar
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objs as go
import plotly.offline as pyo

# # load read data
# df_hepatitis = pd.read_csv('Data_Sources/hepatitis.csv')
# df_measles = pd.read_csv('Data_Sources/measles.csv')
# df_mumps = pd.read_csv('Data_Sources/mumps.csv')
# df_pertussis = pd.read_csv('Data_Sources/pertussis.csv')
# df_rubella = pd.read_csv('Data_Sources/rubella.csv')
# df_smallpox = pd.read_csv('Data_Sources/smallpox.csv')
# df_hepatitis

# # taking care of data discrepencies 
# dfs = [df_hepatitis, df_measles, df_mumps, df_pertussis, df_rubella, df_smallpox]

# for i, df in enumerate(dfs):
#     max_cases = df['cases'].max()  
#     dfs[i] = df[df['cases'] != max_cases]  
    
class DiseasePredictor(nn.Module):
    def __init__(self):
        super(DiseasePredictor, self).__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=64)  # Assuming 3 features for simplicity
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)  # Output 1 value: the number of cases
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# # Define a function to convert DataFrames to tensors
# def df_to_tensors(df_features, df_target):
#     features_tensor = torch.tensor(df_features.values, dtype=torch.float32)
#     target_tensor = torch.tensor(df_target.values, dtype=torch.float32).view(-1, 1)
#     return features_tensor, target_tensor

# # Loop through each DataFrame
# disease_dfs = {
#     'Hepatitis': df_hepatitis,
#     'Measles': df_measles,
#     'Mumps': df_mumps,
#     'Pertussis': df_pertussis,
#     'Rubella': df_rubella,
#     'Smallpox': df_smallpox
# }

# for disease, df in disease_dfs.items():
#     print(f"\nProcessing {disease}")
    
#     # Assume X and y are defined; you'll need to adapt this part to actually prepare X and y for each df
#     X = df[['week', 'incidence_per_capita']]  # Placeholder: replace with actual features
#     y = df['cases']
    
#     # Split into training and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Convert to tensors
#     X_train_tensor, y_train_tensor = df_to_tensors(X_train, y_train)
#     X_val_tensor, y_val_tensor = df_to_tensors(X_val, y_val)
    
#     # Initialize model and other components for each disease to avoid knowledge retention
#     model = DiseasePredictor()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = torch.nn.MSELoss()
    
#     # Training loop
#     epochs = 5
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         outputs = model(X_train_tensor)
#         loss = criterion(outputs, y_train_tensor)
#         loss.backward()
#         optimizer.step()
    
#     # Evaluation
#     model.eval()
#     with torch.no_grad():
#         predictions = model(X_val_tensor)
#         val_loss = criterion(predictions, y_val_tensor)
#         print(f"{disease} Validation Loss: {val_loss.item()}")

#     # Optionally, save each model with a disease-specific name
#     torch.save(model.state_dict(), f'{disease.lower()}_model.pth')
    
# def predict_and_create_table(model, X_features, states):
#     # Convert features to tensor
#     features_tensor = torch.tensor(X_features.values, dtype=torch.float32)
    
#     # Predict cases
#     model.eval()  # Set the model to evaluation mode
#     with torch.no_grad():
#         predicted_cases_tensor = model(features_tensor)
    
#     # Convert predictions to numpy array
#     predicted_cases = predicted_cases_tensor.numpy().flatten()  # Adjust shape as necessary
    
#     # Create DataFrame with state and predicted cases
#     predicted_df = pd.DataFrame({
#         'State': states,
#         'Predicted Cases': predicted_cases
#     })
    
#     return predicted_df

# # Assuming disease_dfs dictionary is already defined and filled with DataFrames for each disease

# for disease, df in disease_dfs.items():
#     print(f"\nProcessing {disease}")
    
#     # Extract state information
#     states = df['state']
    
#     # Prepare features - ensure these match your model's expected input
#     X = df[['week', 'incidence_per_capita']]
    
#     # Initialize model - assuming a single model architecture for all diseases
#     model = DiseasePredictor()
#     # Load the trained model weights - replace 'your_model_path.pth' with the actual path
#     model_path = f'{disease.lower()}_model.pth'
#     model.load_state_dict(torch.load(model_path))
    
#     # Predict cases and create table
#     predicted_table = predict_and_create_table(model, X, states)
    
#     # Print the table
#     print(f"{disease} Predicted Cases by State:")
#     print(predicted_table)
    
#     # # Convert X_train and y_train to PyTorch tensors
# # X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
# # y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

# # # If y_train is a series, ensure it's reshaped into a 2D tensor for consistency
# # y_train_tensor = y_train_tensor.view(-1, 1)

# # epochs = 5  # Example epoch count
# # for epoch in range(epochs):
# #     optimizer.zero_grad()
# #     outputs = model(X_train_tensor)  # Use the tensor version
# #     loss = criterion(outputs, y_train_tensor)  # Use the tensor version
# #     loss.backward()
# #     optimizer.step()
# #     print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# # torch.save(model.state_dict(), 'model.pth')

# # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
# # X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# # X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
# # y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
# # y_val_tensor = y_val_tensor.view(-1, 1)

# # model.eval()  # Set the model to evaluation mode
# # with torch.no_grad():
# #     predictions = model(X_val_tensor)  # Use the tensor version
# #     val_loss = criterion(predictions, y_val_tensor)  # Use the tensor version
# #     print(f"Validation Loss: {val_loss.item()}")

# model_files = ['hepatitis_model.pth', 'measles_model.pth', 'mumps_model.pth', 
#                'pertussis_model.pth', 'rubella_model.pth', 'smallpox_model.pth']
# models = {}

# for file_name in model_files:
#     model = DiseasePredictor()
#     model.load_state_dict(torch.load(file_name))
#     model.eval()
#     models[file_name] = model
    
# def calculate_accuracy(model, X_val_tensor, y_val_tensor):
#     with torch.no_grad():
#         outputs = model(X_val_tensor)
#         _, predicted_classes = torch.max(outputs, 1)
#         correct_predictions = (predicted_classes == y_val_tensor).sum().item()
#         accuracy = correct_predictions / y_val_tensor.size(0)
#     return accuracy

# datasets = {
#     'hepatitis': df_hepatitis,
#     'measles': df_measles,
#     'mumps': df_mumps,
#     'pertussis': df_pertussis,
#     'rubella': df_rubella,
#     'smallpox': df_smallpox
# }

# # validation_data = {
# #     'hepatitis': (X_val_tensor_hepatitis, y_val_tensor_hepatitis),
# #     'measles': (X_val_tensor_measles, y_val_tensor_measles),
# #     'mumps': (X_val_tensor_mumps, y_val_tensor_mumps),
# #     'pertussis': (X_val_tensor_pertussis, y_val_tensor_pertussis),
# #     'rubella': (X_val_tensor_rubella, y_val_tensor_rubella),
# #     'smallpox': (X_val_tensor_smallpox, y_val_tensor_smallpox)
# # }

# validation_tensors = {}

model_files = ['public/api/hepatitis_model.pth', 'public/api/measles_model.pth', 'public/api/mumps_model.pth', 
               'public/api/pertussis_model.pth', 'public/api/rubella_model.pth', 'public/api/smallpox_model.pth']
models = {}

for file_name in model_files:
    disease_name = file_name.split("_")[0]
    model = DiseasePredictor()
    model.load_state_dict(torch.load(file_name))
    model.eval()
    models[disease_name] = model

# Function to predict cases for a given disease and set of features
def predict_cases(model, X_features):
    features_tensor = torch.tensor(X_features.values, dtype=torch.float32)
    with torch.no_grad():
        predicted_cases_tensor = model(features_tensor)
    predicted_cases = predicted_cases_tensor.numpy().flatten()
    return predicted_cases

# Now you can use the loaded models to predict cases for each disease
# For example, to predict cases for hepatitis
df_hepatitis = pd.read_csv('Data_Sources/hepatitis.csv')  # Load the hepatitis dataset
X_hepatitis = df_hepatitis[['week', 'incidence_per_capita']]  # Extract features
predicted_cases_hepatitis = predict_cases(models['hepatitis'], X_hepatitis)
df_hepatitis['predicted_cases'] = predicted_cases_hepatitis

all_states = [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']

df_hepatitis = pd.DataFrame({'state': ['PA'], 'predicted_cases': [100]})
df_measles = pd.DataFrame({'state': [all_states], 'predicted_cases': [150]})
df_mumps = pd.DataFrame({'state': [all_states], 'predicted_cases': [20]})
df_pertussis = pd.DataFrame({'state': [all_states], 'predicted_cases': [250]})
df_rubella = pd.DataFrame({'state': [all_states], 'predicted_cases': [300]})
df_smallpox = pd.DataFrame({'state': [all_states], 'predicted_cases': [350]})

disease_dataframes = {
    'Hepatitis': df_hepatitis,
    'Measles': df_measles,
    'Mumps': df_mumps,
    'Pertussis': df_pertussis,
    'Rubella': df_rubella,
    'Smallpox': df_smallpox
}

# # Ensure all DataFrames have a 'predicted_cases' column; add it with default values if missing
# for disease_name, df in disease_dataframes.items():
#     if 'predicted_cases' not in df.columns:
#         df['predicted_cases'] = 0  # Assign a default value

# Function to create a heatmap for a given disease DataFrame
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
    for disease_name, df in disease_dataframes.items():
        create_heatmap_for_state(df, disease_name, selectedState)
    
# df_hepatitis['year'] = df_hepatitis['week'].apply(lambda x: int(str(x)[:4]))
# df_hepatitis['week_of_year'] = df_hepatitis['week'].apply(lambda x: int(str(x)[4:]))

# # Use the 'year' and 'week_of_year' as features for now
# X = df_hepatitis[['year', 'week_of_year', 'incidence_per_capita']]
# y = df_hepatitis['cases']

# # Splitting the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Predict on the testing set
# y_pred = model.predict(X_test)

# # Evaluate the model
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# # Calculate R-squared on the training set
# r_squared_train = model.score(X_train, y_train)
# print(f"R-squared on the training set: {r_squared_train}")

# # Calculate R-squared on the testing set
# r_squared_test = model.score(X_test, y_test)
# print(f"R-squared on the testing set: {r_squared_test}")

# print(f"RMSE: {rmse}")