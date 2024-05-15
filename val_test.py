import statsmodels.stats.api as sms
import torch
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, recall_score, roc_curve
import torch.optim as optim
import numpy as np
import pandas as pd

# Load selected feature names
selected_features_path = 'selected_features.csv'

selected_features = pd.read_csv(selected_features_path, header=None).squeeze().tolist()
if selected_features[0] == '0':
    selected_features = selected_features[1:]
adjusted_selected_features = [feature + '_x' for feature in selected_features]
# Read Excel file
excel_file_path = r'output_processed.xlsx'
df = pd.read_excel(excel_file_path)

# Extract label column and feature columns
labels = df.iloc[:, 0].values  # The first column is labels

# Only select features listed in selected_features
features = df[selected_features].values

# Handle missing values
imputer = SimpleImputer(strategy='mean')
features = imputer.fit_transform(features)

# Data normalization
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Select the best k features
k_best = 19  # Adjust k based on actual situation
selector = SelectKBest(f_classif, k=k_best)
features = selector.fit_transform(features, labels)

# Convert data to tensors
features_tensor = torch.tensor(features, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)

# Create a TensorDataset, combining features and labels
dataset = TensorDataset(features_tensor, labels_tensor)

# Create DataLoader
batch_size = 32

# Define the deep learning model
class MyModel(nn.Module):
    def __init__(self, n_features, n_classes=2, n_hidden=256):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define validation function
def validate_model(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_preds_prob = []
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            preds_prob = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_preds_prob.extend(preds_prob)
            all_labels.extend(batch_labels.cpu().numpy())
    acc = accuracy_score(all_labels, (np.array(all_preds_prob) > 0.5).astype(int))
    auc = roc_auc_score(all_labels, all_preds_prob)
    return acc, auc

# Initialize variables to track the best model
best_val_auc = -np.inf
best_model_weights = None

# Create model instance and move to GPU
model = MyModel(n_features=k_best)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Split data into five folds and start training
num_folds = 5
stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
num_epochs = 10

for fold, (train_index, val_index) in enumerate(stratified_kfold.split(features, labels)):
    print(f"Fold {fold + 1}/{num_folds}")
    train_dataset = TensorDataset(features_tensor[train_index], labels_tensor[train_index])
    val_dataset = TensorDataset(features_tensor[val_index], labels_tensor[val_index])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    val_acc, val_auc = validate_model(model, val_loader)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model_weights = model.state_dict().copy()
        print(best_val_auc)

# Save the best model weights
torch.save(best_model_weights, 'auc_improvement/model_best.pth')

# Load a new test set
test_df_path = 'integrated_validation_set_2024-03-18.xlsx'  # Update to actual test set file path
test_df = pd.read_excel(test_df_path)

# Apply the same data preprocessing steps
# Note: Use the previously fit imputer and scaler
test_features = test_df[adjusted_selected_features].values
test_features = scaler.transform(test_features)
test_features = selector.transform(test_features)  # Use the same feature selection as training
test_labels = test_df['label'].values  # Adjust based on actual label column name

# Convert processed data to tensors
test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.load_state_dict(torch.load('auc_improvement/model_best.pth'))
model.eval()
model.to(device)

all_preds_prob = []
all_preds = []
all_actuals = []
with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        outputs = model(batch_features)

        # Calculate probabilities and predictions
        probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_preds_prob.extend(probs)
        all_preds.extend(preds)
        all_actuals.extend(batch_labels.cpu().numpy())

# Calculate performance metrics on the test set
test_auc = roc_auc_score(all_actuals, all_preds_prob)
test_accuracy = accuracy_score(all_actuals, all_preds)
print(f"Test AUC: {test_auc:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save test set predictions
test_df['Probability'] = all_preds_prob  # Add prediction probability column
test_df['Prediction'] = all_preds  # Add prediction result column

# Update the list of output columns, only including necessary columns
output_columns = ['编号', '患者姓名', 'label', 'Probability', 'Prediction']

# Create output DataFrame
output_df = test_df.loc[:, output_columns]

# Save to Excel file
output_df.to_excel('final_test_predictions.xlsx', index=False)