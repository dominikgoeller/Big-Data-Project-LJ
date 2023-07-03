from sklearn.linear_model import SGDClassifier  
from dask.distributed import Client, LocalCluster
import xgboost as xgb
import dask.array as da
import dask.dataframe as dd
import dask_ml.model_selection as dms
import numpy as np
from dask_ml.wrappers import Incremental
from sklearn.metrics import accuracy_score,log_loss
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


def get_data(parquet_data):
    ddf = dd.read_parquet(path=parquet_data)
    print(ddf.info())
     # Calculate threshold, defined as the 75th percentile of the 'summons_number' column
    threshold = ddf['summons_number'].quantile(0.75).compute()
    ddf = ddf.fillna(0)
    # Add a new column 'is_high_ticket_day' which is 1 if 'summons_number' is greater than threshold, 0 otherwise
    ddf['is_high_ticket_day'] = (ddf['summons_number'] > threshold).astype(int)

    y_ddf = ddf['is_high_ticket_day']
    X_ddf = ddf.drop('is_high_ticket_day', axis=1)
    
    X_train_ddf, X_test_ddf, y_train_ddf, y_test_ddf = dms.train_test_split(X_ddf, y_ddf, test_size=0.2, shuffle=True, random_state=42)

    X_train_dda = X_train_ddf.to_dask_array(lengths=True)
    X_test_dda = X_test_ddf.to_dask_array(lengths=True)
    y_train_dda = y_train_ddf.to_dask_array(lengths=True)
    y_test_dda = y_test_ddf.to_dask_array(lengths=True)


    return X_train_dda, X_test_dda, y_train_dda, y_test_dda
    #return X_train_ddf, X_test_ddf, y_train_ddf, y_test_ddf

# a) distributed algorithms from DASK-ML(XGBoost, LightGBM)
@profile(stream=xgb_fh)
def train_xgboost_model(X_train, X_test,y_train, y_test, client):
    start_time = time.time()

    clf = XGBClassifier(objective='binary:logistic', random_state=111)
    clf.client = client
    clf.fit(X=X_train, y=y_train)
    preds = clf.predict(X=X_test)
    accuracy = accuracy_score(y_test, preds)
    print("XGBoost Model Accuracy: ", accuracy)

    preds_proba = clf.predict_proba(X_test)
    loss = log_loss(y_test, preds_proba)
    print("XGBoost Model Log Loss: ", loss)

    clf.save_model('../../models/XGBClassifie.bin')
    client.close()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken for Dask-ML model training: {total_time} seconds")

    logger = logging.getLogger('metrics_log')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("../../reports/logs/xgb_metrics.log")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info(f"Dask-ML Model Accuracy: {accuracy}")
    logger.info(f"Dask-ML Model Log Loss: {loss}")
    logger.info(f"Total time taken for Dask-ML model training: {total_time} seconds")
    logger.removeHandler(fh)
    


# b) common scikit-learn algorithms utilizing partial_fit()
def data_chunker(X, y, chunk_size=1000):
    """Generator function to yield chunks of data."""
    X_aligned = da.rechunk(X, chunks=(chunk_size, -1))
    y_aligned = da.rechunk(y, chunks=(chunk_size, -1))

    num_chunks = X_aligned.shape[0]
    for i in range(num_chunks):
        X_chunk = X_aligned[i]
        y_chunk = y_aligned[i]
        #X_chunk_reshaped = np.reshape(X_chunk, (X_chunk.shape[0], -1))
        X_chunk_reshaped = np.reshape(X_chunk, (-1,))
        print("X_chunk shape:", X_chunk_reshaped.shape)
        print("y_chunk shape:", y_chunk.shape)
        yield X_chunk_reshaped, y_chunk



@profile(stream=dask_ml_fh)
def train_dask_ml_model2(X_train_dda, X_test_dda, y_train_dda, y_test_dda, classes):
    start_time = time.time()
    
    sgd = SGDClassifier(loss='log_loss')

    model = Incremental(sgd)

    #for X_batch, y_batch in data_chunker(X_train_dda, y_train_dda, chunk_size=1000):
        #model.fit(X_batch, y_batch, classes=classes)
    print(type(X_train_dda))
    print(type(y_train_dda))
    model.fit(X=X_train_dda, y=y_train_dda, classes=classes)
    pred = model.predict(X_test_dda)

    accuracy = accuracy_score(y_test_dda.compute(), pred.compute())
    print("Dask-ML Model Accuracy: ", accuracy)

    pred_proba = model.predict_proba(X_test_dda)
    loss = log_loss(y_test_dda.compute(), pred_proba.compute())
    print("Dask-ML Model Log Loss: ", loss)

    sgd_model = model.estimator
    dump(sgd_model, '../../models/sgd_model.joblib')

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken for Dask-ML model training: {total_time} seconds")

    logger = logging.getLogger('metrics_log')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("../../reports/logs/dask_ml_metrics.log")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info(f"Dask-ML Model Accuracy: {accuracy}")
    logger.info(f"Dask-ML Model Log Loss: {loss}")
    logger.info(f"Total time taken for Dask-ML model training: {total_time} seconds")
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
    X_train_dda, X_test_dda, y_train_dda, y_test_dda = get_data(parquet_data=path)
    print(np.unique(y_train_dda.compute()))
    train_xgboost_model(X_train_dda, X_test_dda, y_train_dda, y_test_dda, client)
    classes = np.unique(y_train_dda.compute())
    #train_dask_ml_model2(X_train_dda, X_test_dda, y_train_dda, y_test_dda, classes=classes)

if __name__ == '__main__':
    cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='2GB')
    client = Client(cluster)
    main(client=client)