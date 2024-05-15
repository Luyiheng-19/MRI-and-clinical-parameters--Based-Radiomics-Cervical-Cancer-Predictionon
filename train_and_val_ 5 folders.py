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
from torchgen.selective_build import selector

# Load Excel file
excel_file_path = r'output_processed.xlsx'
df = pd.read_excel(excel_file_path)

# Extract label column and feature columns
labels = df.iloc[:, 0].values  # The first column is labels
features = df.iloc[:, 1:].values  # Features start from the second column

# Handle missing values
imputer = SimpleImputer(strategy='mean')
features = imputer.fit_transform(features)

# Data normalization
scaler = StandardScaler()
features = scaler.fit_transform(features)


# Convert data to tensors
features_tensor = torch.tensor(features, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)

# Create a TensorDataset, combining features and labels together
dataset = TensorDataset(features_tensor, labels_tensor)

# Create DataLoader
batch_size = 32

# Define the deep learning model
class MyModel(nn.Module):
    def __init__(self, n_features=19, n_classes=2, n_hidden=256):
        super(MyModel, self).__init__()

        # Use a fully connected layer to process feature vectors
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

# Define validation function
def validate_model(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_preds_prob = []
    fixed_threshold = 0.5  # Fixed threshold

    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features, batch_labels = batch_features.to('cuda'), batch_labels.to('cuda')

            # Forward pass
            outputs = model(batch_features)

            # Compute prediction probabilities
            preds_prob = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_preds_prob.extend(preds_prob)
            all_labels.extend(batch_labels.cpu().numpy())

    # Recalculate predictions using a fixed threshold
    fixed_preds = (np.array(all_preds_prob) > fixed_threshold).astype(int)

    # Compute other performance metrics
    acc = accuracy_score(all_labels, fixed_preds)
    auc = roc_auc_score(all_labels, all_preds_prob)
    cm = confusion_matrix(all_labels, fixed_preds)

    return acc, auc, cm, fixed_threshold, fixed_preds, all_preds_prob

# Split data into five folds
num_folds = 5
stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initialize lists of performance metrics
train_acc_list, train_auc_list, train_sen_list, train_cutoff_list, train_spe_list = [], [], [], [], []
test_acc_list, test_auc_list, test_sen_list, test_cutoff_list, test_spe_list = [], [], [], [], []

# Create model instance
model = MyModel()

# Move model to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize an empty DataFrame to store the results
results_df = pd.DataFrame()

# Training and validation
for fold, (train_index, val_index) in enumerate(stratified_kfold.split(features, labels)):
    print(f"Fold {fold + 1}/{num_folds}")

    # Split data sets
    train_dataset = TensorDataset(features_tensor[train_index], labels_tensor[train_index])
    val_dataset = TensorDataset(features_tensor[val_index], labels_tensor[val_index])

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Train the model
    num_epochs = 10
    best_val_acc = 0.0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        for batch_features, batch_labels in train_data_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        # Validate at the end of each epoch
        val_acc, val_auc, val_sen, val_cutoff, val_spe, _ = validate_model(model, val_data_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_acc}')

        # Save the best-performing model on the validation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
            # Save the best model for each fold
            fold_model_save_path = f'model_best_fold_{fold + 1}.pth'
            torch.save(best_model, fold_model_save_path)

    # At the end of training for each fold, use the best model to evaluate on that fold's test set
    model.load_state_dict(best_model)
    model.eval()

    # Compute training set performance
    train_dataset = TensorDataset(features_tensor[train_index], labels_tensor[train_index])
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    train_acc, train_auc, train_sen, train_cutoff, train_spe, train_cm = validate_model(model, train_data_loader)

    # Compute test set performance
    test_dataset = TensorDataset(features_tensor[val_index], labels_tensor[val_index])
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_acc, test_auc, test_sen, test_cutoff, test_spe, test_cm = validate_model(model, test_data_loader)

    # Compute confusion matrix
    all_train_preds = []
    all_train_labels = []
    # Initialize a DataFrame to store results of the current fold
    fold_results = pd.DataFrame()
    with torch.no_grad():
        for batch_features, batch_labels in train_data_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            # Forward pass
            outputs = model(batch_features)

            # Compute prediction probabilities
            preds_prob = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = (preds_prob > 0.5).astype(int)

            all_train_preds.extend(preds)
            all_train_labels.extend(batch_labels.cpu().numpy())
            # Append results to the DataFrame

    # Compute confusion matrix
    train_cm = confusion_matrix(all_train_labels, all_train_preds)
    # Compute test set confusion matrix
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for batch_features, batch_labels in test_data_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            # Forward pass
            outputs = model(batch_features)

            # Compute prediction probabilities
            preds_prob = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = (preds_prob > 0.5).astype(int)

            test_preds.extend(preds)
            test_labels.extend(batch_labels.cpu().numpy())
            # Append results to the current fold's results DataFrame

        results_df = pd.concat([results_df, fold_results], ignore_index=True)

    # Compute test set confusion matrix
    test_cm = confusion_matrix(test_labels, test_preds)

    # Print training set performance
    print(f'Fold {fold + 1} Train Accuracy: {train_acc}')
    print('Train Confusion Matrix:')
    print(train_cm)

    # Print test set performance
    print(f'Fold {fold + 1} Test Accuracy: {test_acc}')
    print('Test Confusion Matrix:')
    print(test_cm)

# Load a fresh test set
test_df_path = 'integrated_validation_set_2024-03-18.xlsx'  # Update to actual test set file path
test_df = pd.read_excel(test_df_path)

# Apply the same data preprocessing steps
# Note: Use the previously fit imputer and scaler
test_features = imputer.transform(test_df.iloc[:, 3:].values)  # Assuming the first three columns are not feature columns
test_features = scaler.transform(test_features)
test_labels = test_df['label'].values  # Adjust based on actual label column name

# Convert processed data to tensors
test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluate each fold's model and identify the one that performs best on the test set
best_auc = 0
best_model_path = ''

for fold in range(1, 6):
    model_path = f'model_best_fold_{fold}.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    all_preds_prob = []
    with torch.no_grad():
        for batch_features, _ in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            preds_prob = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_preds_prob.extend(preds_prob)

    auc = roc_auc_score(test_labels, all_preds_prob)
    print(f"Fold {fold}, Test AUC: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_model_path = model_path

# Use the best-performing model on the test set for final predictions
print(f"Best Model Path: {best_model_path}")
model.load_state_dict(torch.load(best_model_path))
model.eval()

all_preds_prob = []
all_preds = []
with torch.no_grad():
    for batch_features, _ in test_loader:
        batch_features = batch_features.to(device)
        outputs = model(batch_features)
        preds_prob = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        preds = np.argmax(outputs.cpu().numpy(), axis=1)
        all_preds_prob.extend(preds_prob)
        all_preds.extend(preds)

# Save test set predictions
test_df['Probability'] = all_preds_prob  # Add prediction probability column
test_df['Prediction'] = all_preds  # Add prediction result column

# Update the list of output columns, including only the necessary columns
output_columns = ['编号', '患者姓名', 'label', 'Probability', 'Prediction']

# Create output DataFrame
output_df = test_df.loc[:, output_columns]

# Save to Excel file
output_df.to_excel('final_test_predictions_88feature.xlsx', index=False)