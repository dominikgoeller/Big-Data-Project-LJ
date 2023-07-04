from sklearn.linear_model import SGDClassifier  
from dask.distributed import Client, LocalCluster
import xgboost as xgb
import os
import dask.dataframe as dd
import dask_ml.model_selection as dms
import numpy as np
from dask_ml.wrappers import Incremental
from sklearn.metrics import accuracy_score,log_loss, precision_score, recall_score
import torch.nn as nn
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
import multiprocessing as mp
from xgboost import XGBClassifier
from memory_profiler import LogFile, profile
import logging
from joblib import dump
import time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import pandas as pd
from dask_ml.preprocessing import StandardScaler
from dask.diagnostics import ProgressBar
import psutil

class FileHandlerStream:
    def __init__(self, file_handler):
        self.file_handler = file_handler

    def write(self, s):
        self.file_handler.emit(logging.LogRecord('', logging.DEBUG, '', 0, s, (), None))

    def flush(self):
        self.file_handler.flush()


def create_func_logger(func_name):
    # create logger with function name
    logger = logging.getLogger(func_name)
    logger.setLevel(logging.DEBUG)

    # create file handler which logs debug messages
    fh = logging.FileHandler(f"../../reports/logs/{func_name}_memory_profile.log")
    fh.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger, FileHandlerStream(fh)

xgb_logger, xgb_fh = create_func_logger('train_xgboost_model')
dask_ml_logger, dask_ml_fh = create_func_logger('train_dask_ml_model')
b_ml_logger, b_ml_fh = create_func_logger('train_b_ml_model')



def get_data2(parquet_data):
    ddf = dd.read_parquet(path=parquet_data)
    ddf = ddf.fillna(0)
    #print(ddf.head(5))
    # Calculate threshold, defined as the 75th percentile of the 'summons_number' column
    threshold = ddf['summons_number'].quantile(0.75).compute()
    #print("THRESHOLD", threshold)
    #print(ddf[ddf['summons_number'] > threshold].compute())

    # Filter rows where 'summons_number' is higher than the threshold
    high_summons = ddf[ddf['summons_number'] > threshold].compute()
    
    # Add a new column 'is_high_ticket_day' which is 1 if 'summons_number' is greater than threshold, 0 otherwise
    ddf['is_high_ticket_day'] = (ddf['summons_number'] > threshold).astype(int)
    #print(ddf.head(5))

    y_ddf = ddf['is_high_ticket_day']
    X_ddf = ddf.drop('is_high_ticket_day', axis=1)
    
    X_temp, X_test_ddf, y_temp, y_test_ddf = dms.train_test_split(X_ddf, y_ddf, test_size=0.2, shuffle=True, random_state=111)
    X_train_ddf, X_val_ddf, y_train_ddf, y_val_ddf = dms.train_test_split(X_temp, y_temp, test_size=0.25, shuffle=True, random_state=111)

    #print("Distribution of 'is_high_ticket_day' in training set:\n", y_train_ddf.value_counts(normalize=True).compute())
    #print("Distribution of 'is_high_ticket_day' in validation set:\n", y_val_ddf.value_counts(normalize=True).compute())
    #print("Distribution of 'is_high_ticket_day' in test set:\n", y_test_ddf.value_counts(normalize=True).compute())

    X_train_dda = X_train_ddf.to_dask_array(lengths=True)
    X_val_dda = X_val_ddf.to_dask_array(lengths=True)
    X_test_dda = X_test_ddf.to_dask_array(lengths=True)
    y_train_dda = y_train_ddf.to_dask_array(lengths=True)
    y_val_dda = y_val_ddf.to_dask_array(lengths=True)
    y_test_dda = y_test_ddf.to_dask_array(lengths=True)

    
    # Make sure issue_date is in datetime format
    high_summons = high_summons.reset_index()
    high_summons['issue_date'] = pd.to_datetime(high_summons['issue_date'])
    
    # Aggregate by month
    high_summons = high_summons.resample('M', on='issue_date')['summons_number'].sum()

    # Create a bar chart of the issue_dates with high summons_number
    high_summons_dates = high_summons.index
    high_summons_values = high_summons.values
    
    plt.figure(figsize=(10, 5))
    plt.bar(high_summons_dates, high_summons_values)
    plt.xlabel('Issue Dates')
    plt.ylabel('Summons Number')
    plt.title('Issue Dates with Summons Number higher than the threshold (aggregated by month)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('../../reports/figures/high_summons_dates.png')
    plt.close()

    return X_train_dda, X_val_dda, X_test_dda, y_train_dda, y_val_dda, y_test_dda


# a) distributed algorithms from DASK-ML(XGBoost, LightGBM)
    
@profile(stream=xgb_fh)
def train_and_test_model(X_train, X_val, X_test, y_train, y_val, y_test, client):
    start_time = time.time()

    clf = XGBClassifier(objective='binary:logistic', random_state=111, eval_metric=['error', 'logloss'])
    clf.client = client

    # Specify our validation set and early stopping rounds
    eval_set = [(X_train, y_train), (X_val, y_val)]
    clf.fit(X=X_train, y=y_train, eval_set=eval_set, early_stopping_rounds=4)

    # Extract the performance metrics
    results = clf.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)

    # Save the performance metrics plots
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Validation')
    ax.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')
    plt.savefig('../../reports/figures/xgb_log_loss.png')

    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Validation')
    ax.legend()
    plt.ylabel('Classification Error')
    plt.title('XGBoost Classification Error')
    plt.savefig('../../reports/figures/xgb_classification_error.png')

    clf.save_model('../../models/XGBClassifie.bin')

    # Begin testing phase
    y_pred = clf.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_log_loss = log_loss(y_test, clf.predict_proba(X_test))

    # Print the metrics
    print("Test Accuracy: ", test_accuracy)
    print("Test Precision: ", test_precision)
    print("Test Recall: ", test_recall)
    print("Test Log Loss: ", test_log_loss)

    # Log the metrics
    logger = logging.getLogger('metrics_log')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("../../reports/logs/xgb_metrics.log")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info(f"Test Accuracy: {test_accuracy}")
    logger.info(f"Test Precision: {test_precision}")
    logger.info(f"Test Recall: {test_recall}")
    logger.info(f"Test Log Loss: {test_log_loss}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken for XGBoost-ML model training: {total_time} seconds")
    logger.info(f"Total time taken for XGBoost-ML model training: {total_time} seconds")
    # Get the memory usage
    process = psutil.Process(os.getpid())
    total_memory = process.memory_info().rss  # in bytes

    # You might want to convert bytes to MB or GB for better readability
    total_memory_MB = total_memory / (1024 ** 2)  # Convert to MBs
    print(f"Total memory used for PartialFit model training: {total_memory_MB} MB")
    logger.info(f"Total memory used for PartialFit model training: {total_memory_MB} MB")
    logger.removeHandler(fh)

    # creating figures for the metrics and saving them
    metrics = [test_accuracy, test_precision, test_recall, test_log_loss]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'Log Loss']

    plt.figure(figsize=(10, 5))
    plt.bar(metric_names, metrics, color=['blue', 'orange', 'green', 'red'])
    for i in range(len(metrics)):
        plt.text(i, metrics[i], round(metrics[i], 2), ha='center')
    plt.savefig('../../reports/figures/XGBoost_metrics_bar_chart.png')


# b) common scikit-learn algorithms utilizing partial_fit()

@profile(stream=b_ml_fh)
def train_ml_model(X_train_dda, y_train_dda, X_val_dda, y_val_dda, X_test_dda, y_test_dda, classes):
    """
    Trains a ML model using batch processing. 

    Parameters:
    X_train_dda: Training features
    y_train_dda: Training target
    X_val_dda: Validation features
    y_val_dda: Validation target
    X_test_dda: Test features
    y_test_dda: Test target
    """
    start_time = time.time()
    # Feature scaling
    scaler = StandardScaler()
    X_train_dda = scaler.fit_transform(X_train_dda)
    X_val_dda = scaler.transform(X_val_dda)
    X_test_dda = scaler.transform(X_test_dda)

    # Initialize the SGDClassifier
    clf = SGDClassifier(loss='log', max_iter=1000, random_state=111)

    # The size of each batch (Ensure that this size will fit into the memory of your worker nodes)
    batch_size = 500
    num_batches = X_train_dda.shape[0] // batch_size

    with ProgressBar():
        # The partial_fit method updates the model's parameters with the data of each batch
        for i in range(num_batches):
            clf.partial_fit(X_train_dda[i*batch_size:(i+1)*batch_size], 
                            y_train_dda[i*batch_size:(i+1)*batch_size],
                            classes=classes)
        remaining_size = X_train_dda.shape[0] % batch_size
        if remaining_size != 0:
            clf.partial_fit(X_train_dda[-remaining_size:],
                    y_train_dda[-remaining_size:],
                    classes=classes)

    # Use the validation set to tune your model hyperparameters
    val_predictions = clf.predict(X_val_dda)
    val_accuracy = accuracy_score(y_val_dda, val_predictions)
    val_loss = log_loss(y_val_dda, clf.predict_proba(X_val_dda))

    # Perform prediction on the test set and calculate the metrics
    y_pred = clf.predict(X_test_dda)
    test_accuracy = accuracy_score(y_test_dda, y_pred)
    test_precision = precision_score(y_test_dda, y_pred)
    test_recall = recall_score(y_test_dda, y_pred)
    test_loss = log_loss(y_test_dda, clf.predict_proba(X_test_dda))

    print(f"Validation Accuracy: {val_accuracy}")
    print(f"Test Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken for PartialFit-ML model training: {total_time} seconds")

    dump(clf, '../../models/sgd_model.joblib')

    # creating figures for the metrics and saving them
    metrics = [test_accuracy, test_precision, test_recall, test_loss]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'Log Loss']

    plt.figure(figsize=(10, 5))
    plt.bar(metric_names, metrics, color=['blue', 'orange', 'green', 'red'])
    for i in range(len(metrics)):
        plt.text(i, metrics[i], round(metrics[i], 2), ha='center')
    plt.savefig('../../reports/figures/SGD_metrics_bar_chart.png')

    # Log the metrics
    logger = logging.getLogger('metrics_log')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("../../reports/logs/dask_ml_metrics.log")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info(f"Validation Accuracy: {val_accuracy}")
    logger.info(f"Validation Log Loss: {val_loss}")
    logger.info(f"Test Accuracy: {test_accuracy}")
    logger.info(f"Test Precision: {test_precision}")
    logger.info(f"Test Recall: {test_recall}")
    logger.info(f"Test Log Loss: {test_loss}")
    logger.info(f"Total time taken for PartialFit model training: {total_time} seconds")
    # Get the memory usage
    process = psutil.Process(os.getpid())
    total_memory = process.memory_info().rss  # in bytes

    # You might want to convert bytes to MB or GB for better readability
    total_memory_MB = total_memory / (1024 ** 2)  # Convert to MBs
    print(f"Total memory used for PartialFit model training: {total_memory_MB} MB")
    logger.info(f"Total memory used for PartialFit model training: {total_memory_MB} MB")
    logger.removeHandler(fh)


# T6 TODO still missing parquet file to analyse (in trial run we got some interesting anomalies but couldnt find events that matched exactly)
'''
street_day_tickets = df.groupby(['street_name', 'issue_date']).size().reset_index(name='num_tickets')
scaler = MinMaxScaler(feature_range=(0, 100))

# Fit the scaler and transform the 'num_tickets' column
street_day_tickets['num_tickets'] = scaler.fit_transform(street_day_tickets[['num_tickets']])
'''
def detect_anomalies(df, column='num_counts', contamination=0.01):
    # assuming df is your original DataFrame and 'street_name' is your column of interest
    street_day_tickets = df.groupby(['street_name', 'issue_date']).size().reset_index(name='num_tickets')
    scaler = MinMaxScaler(feature_range=(0, 100))

    # Fit the scaler and transform the 'num_tickets' column
    street_day_tickets['num_tickets'] = scaler.fit_transform(street_day_tickets[['num_tickets']])

    clf = IsolationForest(contamination=contamination)
    clf.fit(df[[column]])
    preds = clf.predict(df[[column]])
    df['anomaly'] = preds
    anomaly_dates = df[df['anomaly'] == -1].index.tolist()
    
    return anomaly_dates






# c) TODO dask with PyTorch
class FFNN(nn.Module):
    def __init__(self, n_features):
        super(FFNN, self).__init__()
        
        self.layer1 = nn.Linear(n_features, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x

def train_and_validate_model(model, X_train, y_train, X_test, y_test, device, n_epochs=100):
    # Initialize the distributed environment.
    dist.init_process_group('nccl')

    # Move the model to the device and make it DistributedDataParallel
    model = model.to(device)
    model = DistributedDataParallel(model)

    # Create the DataLoader for our training set
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_sampler = DistributedSampler(train_data)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=(train_sampler is None), sampler=train_sampler, multiprocessing_context=mp.get_context('fork'))

    # Create the DataLoader for our testing set
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_sampler = DistributedSampler(test_data)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, sampler=test_sampler, multiprocessing_context=mp.get_context('fork'))

    # Define a loss function and an optimizer
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

    return model

def main(client):
    path = '../../data/processed/aggregated_data.parquet'
    X_train_dda, X_val_dda, X_test_dda, y_train_dda, y_val_dda, y_test_dda = get_data2(parquet_data= path)
    train_and_test_model(X_train= X_train_dda, X_val= X_val_dda, X_test= X_test_dda, y_train= y_train_dda, y_val= y_val_dda, y_test= y_test_dda, client=client)
    classes = np.unique(y_train_dda.compute())
    #train_ml_model(X_train_dda=X_train_dda, y_train_dda=y_train_dda, X_val_dda= X_val_dda, y_val_dda=y_val_dda, X_test_dda=X_test_dda, y_test_dda=y_test_dda, classes=classes)


if __name__ == '__main__':
    cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='2GB')
    client = Client(cluster)
    main(client=client)