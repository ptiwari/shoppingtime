
# Reading an excel file using Python # Reading
import xlrd 
import datetime
import time
import numpy as np
import math
import os
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn import metrics
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt

def getIthCol(first_sheet,i):
	'''
		Get the ith column of the sheet
	'''
	result = first_sheet.col_values(i)
	label = result[0];
	result = result[1:]
	if type(result[0])==float:
		result = list(map(int,result))
	#if type(result[0]==String):
	#	result = label(result);
	return label,result;	

		
def getUnixTime(result):
	'''
		Get time in the unix format
	'''
	l = [];
	for s in result:
		v = datetime.datetime.strptime(s,'%Y-%m-%d %H:%M:%S');
		u = time.mktime(v.timetuple())
		l.append(u);
	return l


def getExcelSheet(filename):
	'''
		Open excel file and return the sheet
	'''
	# Give the location of the file 
	loc = (filename)   
	# To open Workbook 
	wb = xlrd.open_workbook(loc) 
	 # get the first worksheet
	first_sheet = wb.sheet_by_index(0)
	return first_sheet


def readData(first_sheet,ncols):
	'''
		Read data from the excel
		Args:
			first_sheet: The excel sheet
			ncols: Number of columns to read
		Returns:
			Dictionary of data, columns header name and shopping
			started time in unix format
	'''
	
	columns = [];
	result_dict = {};
	for i in range(1,ncols):
		label,result = getIthCol(first_sheet,i);
		result_dict[label] = result
		print("Reading:",label)
		columns.append(label)		
	label,result = getIthCol(first_sheet,4);
	uxtime = getUnixTime(result)
	columns.append(label)
	result_dict[label] = uxtime
	return result_dict,columns,uxtime

def getUniqueCol(input_features,key):
	'''
		Get the unique values in the list corresponding to key
		presented in input_features dictionary
		Args:
			input_features: Dictionary of features
			key: Key in the dictionary
		Return:
			Return unique elements in the list
	'''
	return list(set(input_features[key]))

def getEmbeddedVocabCol(k,input_features):
	unqcol = getUniqueCol(input_features,k)
	col =  tf.feature_column.categorical_column_with_vocabulary_list(key=k,vocabulary_list=unqcol);
	dim = int(len(unqcol)*0.25)
	if k == "fulfillment_model":
		dim = 2;
	return tf.feature_column.embedding_column(col,dim)
def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """ 
  my_features=[];
  #Store id,shopper_id,fulfillment_mode is the categorical feature. 
  # There could be relatively 
  # few number of stored is, but they may not start from 0. So, we
  # need to normalize the data,The function getEmbeddedVocabCol()
  # normalizes the feature and return embedded column  
  my_features.append(getEmbeddedVocabCol("store_id",input_features))
  my_features.append(getEmbeddedVocabCol("shopper_id",input_features))
  my_features.append(getEmbeddedVocabCol("fulfillment_model",input_features))
  #We used the unix time
  my_features.append(tf.feature_column.numeric_column(key="shopping_started_at"))
  return my_features

def train_nn_regression_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    model_type):
  """Trains a neural network regression model.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and loss over time.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    training_examples: A `DataFrame` containing one or more columns from
      train_trips.xlsx to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      train_trips.xlsx to use as target for training.
    model_type: Type of model Linear or Neural Network
      
  Returns:
    A `DNNRegressor` object trained on the training data.
  """

  periods = 10
  steps_per_period = steps / periods
  
  # Create a DNNRegressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  #my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
  #my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
  if(model_type == "NN"):
  	hidden_units=[10,2]
        #periods = 100;
  	dnn_regressor = tf.estimator.DNNRegressor(
				      		feature_columns=construct_feature_columns(training_examples),
				      		hidden_units=hidden_units,
						optimizer= my_optimizer
  						)
  else:
  	dnn_regressor = tf.estimator.LinearRegressor(feature_columns=construct_feature_columns(training_examples))
  
  # Create input functions.
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["shopping_time"], 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["shopping_time"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    dnn_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions]) 
    #for p,t in zip(training_predictions,training_targets["shopping_time"]):
    #	print(p,t)
    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    training_rmse.append(training_root_mean_squared_error)
  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  prd = list(range(0, periods+1))
  print(prd)
  plt.xticks(prd)
  plt.plot(training_rmse )
  plt.legend()
  plt.show(block=False)
  print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
  return dnn_regressor

def preprocess_targets(stime):
  """Prepares target features (i.e., labels) from California housing data set.

  Args:
    stime: Shoppint time
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  output_targets["shopping_time"] = (stime["shopping_time"])
  return output_targets

def test_fn(features, batch_size):
    """Trains a neural net regression model.
  
    Args:
      features: pandas DataFrame of features
      batch_size: Size of batches to be passed to the model
    Returns:
      Dataset
    """
    features=dict(features) 
    dataset = tf.data.Dataset.from_tensor_slices(features)
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    return dataset   

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a neural net regression model.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                             
    
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    #with tf.Session() as session:
    #	print(session.run(features),session.run(labels))

    return features, labels
def readTestData(test_file):
	"""
		This function read the test data from the excel sheet. 
		We used xlrd module to read the excel sheet. The first_sheet
		corresponds to first sheet in the excel. After opening the excel sheet,
		we call readData to read the data from excel. The readData() doesn't
		read the trip id. So, we needed extra step to read the
		shopping completed time.
		Args
			test_file: Path of the test file
		Returns:
			Excel sheet data, and trip id
	"""
	first_sheet = getExcelSheet(test_file)
	#Read tirp id
	label,tripId = getIthCol(first_sheet,0)
	result_dict,col,uxtime = readData(first_sheet,first_sheet.ncols-1)
	return result_dict,tripId
	
def readTrainData(train_file):
	"""
		This function read the training data from the excel sheet. 
		We used xlrd module to read the excel sheet. The first_sheet
		corresponds to first sheet in the excel. After opening the excel sheet,
		we call readData to read the data from excel. The readData() doesn't
		read the shopping completed time. So, we needed extra step to read the
		shopping completed time.
		Args
			train_file: Path of the training file
		Returns:
			Excel sheet data, name of the column header, shopping time in Unix format
	"""
	#Open Excel sheet
	first_sheet = getExcelSheet(train_file)
	#Read data
	result_dict,col,uxtime = readData(first_sheet,first_sheet.ncols-2)
	#Reda shopping completed time
	label,result = getIthCol(first_sheet,5);
	uxtime1 = getUnixTime(result)
	shoppingtime = [m - n for m,n in zip(uxtime1,uxtime)]
	return result_dict,col,shoppingtime

def trainLinearTheModel(training_file):
	"""
		This function read the training data and train the linear model
		Args:
			training_file: Path of the training file
		Returns:
			Classifier
	"""
	#Read training data	
	result,columns,target = readTrainData(training_file)
	training_targets = {}
	training_targets["shopping_time"] = target
	#Perform Training
	dnn_regressor = train_nn_regression_model(learning_rate=0.01,
						  steps=500,
						  batch_size=10,
						  training_examples=result,
						  training_targets=preprocess_targets(training_targets),model_type="linear")
	return dnn_regressor	

def trainNNTheModel(training_file):
	"""
		This function read the training data and train the Neural Network model
		Args:
			training_file: Path of the training file
		Returns:
			Classifier
	"""
	result,columns,target = readTrainData(training_file)
	training_targets = {}
	training_targets["shopping_time"] = target
	dnn_regressor = train_nn_regression_model(learning_rate=0.01,
						  steps=500,
						  batch_size=10,
						  training_examples=result,
						  training_targets=preprocess_targets(training_targets),model_type="NN")
	return dnn_regressor
def testTheModel(dnn_regressor,test_file,output_file):
	"""
		This function read the test data and test the model
		Args:
			training_file: Path of the training file
		Returns:
			None
	"""
	testing_set,tripId = readTestData(test_file)
	f = open(output_file, "w")
	predictions = dnn_regressor.predict(input_fn=lambda: test_fn(testing_set,batch_size=10))
	f.write("trip_id,shopping_time\n")
	for id,pred_dict in zip(tripId,predictions):
		arr = pred_dict['predictions']
		f.write("%d,%d\n" %(id,arr[0]))
	

		
def main():
	
	dnn_regressor = trainLinearTheModel('train.xlsx')
	#dnn_regressor = trainNNTheModel('train.xlsx')
	print("####Testing#####")
	#testTheModel(dnn_regressor,'test_trips.xlsx',"nnpredict.txt")
	testTheModel(dnn_regressor,'test_trips.xlsx',"linearpredict.txt")
	print("Testing Finished. File linearpredict.txt is generated in the current directory");
	plt.show()

if __name__== "__main__":
  main()


	
