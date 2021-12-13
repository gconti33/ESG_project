'''
Created on 9 déc. 2021

Code to predict Risk premium using historical data + Worldbank 

@author: geraldineconti
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.random.set_seed(0)
from numpy.random import seed
seed(1)

################################################################################
# Parameters

MAX_EPOCHS = 100

mypath = '/Users/geraldineconti/Desktop/projet_Pictet/'

mycountry = 'BRA'

################################################################################
# Helper functions and class (from tensorflow tutorials webpage) 

def compile_and_fit(model, window, patience=2):
    #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
    #                                                  patience=patience,
    #                                                  mode='min')
    
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])
    
    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val)#,
                        #callbacks=[early_stopping])
    return history


# generate data windows
class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
               train_df, val_df, test_df,
               label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
    
        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}
    
        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
    
        self.total_window_size = input_width + shift
    
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
    
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

# to visualize the split window 
# blue input line : input target at each time step 
# green label dots : target prediction value, shown at prediction time (shift of 1) 
# orange prediction crosses : model predictions for each output time step
def plot(self, model=None, plot_col='BRA', max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(max_n, 1, n+1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)
    
        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index
    
        if label_col_index is None:
            continue
    
        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                      marker='X', edgecolors='k', label='Predictions',
                      c='#ff7f0e', s=64)
    
        if n == 0:
            plt.legend()
    
    plt.xlabel('Time [h]')

WindowGenerator.plot = plot

# split windows into a window of inputs and a window of labels 
def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)
    
    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])
    
    return inputs, labels

WindowGenerator.split_window = split_window

# talkes a time series DataFrame and convert it to a tf.data.Dataset 
# (input_window, label_window) pairs 

def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
          data=data,
          targets=None,
          sequence_length=self.total_window_size,
          sequence_stride=1,
          shuffle=True,
          batch_size=32,)
    
    ds = ds.map(self.split_window)
    
    return ds

# holds training, validation and test data
WindowGenerator.make_dataset = make_dataset


# Add properties for accessing them as tf.data.Dataset 
@property
def train(self):
    return self.make_dataset(self.train_df)
#
@property
def val(self):
    return self.make_dataset(self.val_df)
#
@property
def test(self):
    return self.make_dataset(self.test_df)
#
@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
    return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

################################################################################
# Data preparation

# read input file 
df = pd.read_excel(mypath+'input/risk_premium_BRA.xlsx')
df = pd.DataFrame(df)
df.index = df['Date']
df = df.drop(['Date'],axis=1)

# missing values
df = df.interpolate()

# rolling average of target - to smoothen label 
df[mycountry] = df[mycountry].rolling(window=3).mean()
df[mycountry].iloc[0] = 277
df[mycountry].iloc[1] = 277

# remove column that is not strongly correlated 
#print(df.corr())
fig,ax = plt.subplots()
df = df.drop(columns=['bonds'])


# inspect input data
print(df.describe().transpose())
_ = df.plot(subplots=True)
plt.savefig(mypath+'/figures/risk_premium_input_variables.jpeg')

# split data - train/test (not enough data to have three samples) 
# Time series : DO NOT shuffle data ! 
column_indices = {name: i for i, name in enumerate(df.columns)}
n = len(df) 
train_df = df[0:int(n*0.85)]
val_df = df[int(n*0.85):int(n*0.95)]
test_df = df[int(n*0.95):]
num_features = df.shape[1]

# data normalization 
# TODO : change for moving average instead
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

################################################################################
# windows 

wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1,
    train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=[mycountry])

CONV_WIDTH = 3
conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=[mycountry])

LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=[mycountry])

################################################################################
## CNN model : can be run on inputs of any length 

print('Start CNN model #######################################################')

conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

history = compile_and_fit(conv_model, conv_window)

val_performance = conv_model.evaluate(conv_window.val)
performance = conv_model.evaluate(conv_window.test, verbose=0)

# loss function
fig,ax = plt.subplots()
plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'],label='validation loss')
plt.legend()
plt.savefig(mypath+'figures/risk_premium_CNN_loss.jpeg')

fig,ax = plt.subplots()
wide_conv_window.plot(conv_model)
plt.savefig(mypath+'figures/risk_premium_CNN_results.jpeg')

############################################################################
# RNN model 
print('Start LSTM model ######################################################')

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

history = compile_and_fit(lstm_model, wide_window)

fig,ax = plt.subplots()
wide_window.plot(lstm_model)
plt.savefig(mypath+'figures/risk_premium_LSTM_results.jpeg')

# loss function 
fig,ax = plt.subplots()
plt.plot(history.history['loss'],label='training loss')
#plt.plot(history.history['val_loss'])
plt.legend()
plt.savefig(mypath+'figures/risk_premium_LSTM_loss.jpeg')

#val_performance = lstm_model.evaluate(wide_window.val)
performance = lstm_model.evaluate(wide_window.test, verbose=0)

