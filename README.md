# Toronto-weather-average-temperature-prediction-with-RNN
Regression problem addressed by using LSTM (Long Short Term Memory) over Toronto's average temperature time series

Long Short Term Memory - LSTM
For the LSTM model, the entire process was broken down into 3 basic steps which are:, Building the Recurrent Neural Network, Making predictions and visualizing the results, and Tunning the Neural Network
Long Short Term Memory networks rely on a gated cell to track information throughout many time steps, in other words deep sequence modeling. LSTM modules contain computational blocks that control information flow and its cells are able to track information throughout many timesteps. Information is added or removed through structures called gates, and gates optionally let information through, for example via a sigmoid neural net layer and pointwise multiplication.

<p align="center">
  <img src="https://user-images.githubusercontent.com/90570944/162834605-0f7bc829-876d-4acb-a0a4-d18c2623fccd.png">
</p>

This Recurrent Neural Network basically works this way: First forgetting irrelevant parts of the previous state, secondly by storing the most relevant new information into the cell state, thirdly by updating their internal cell state values and then generating an output. It is widely used to model audio files, text since it can be splitted up into a sequence of characters or also words that can also be interpreted as timesteps. There are several cases depending on the problem we want to address, for instance: many to one for sentiment classification,  one to many for image captioning, and many to many for machine translation.

Building the Recurrent Neural Network

The architecture for this neural network started with a baseline consisting of one hidden layer with 30 neurons, tanh as the activation function and return_sequence as True when stacking LSTM layers, and the output layer with one neuron which is the result of the regression process. After several architecture prototypes and trial and error attempts, adding and removing dropout layers to keep under control the overfitting, the following architecture was the most suitable for this project:

<p align="center">
  <img src="https://user-images.githubusercontent.com/90570944/162837608-3cbe1218-0ea6-45a6-ac06-276ede9e3b2c.png">
</p>

After defining the architecture the subsequent steps were: compile the model, where the chosen optimizer was 'adam' and the loss function was 'mean_squared_error'. Then the model was fitted and special mention at this stage are the following parameters: batch_size = 10, epochs = 100 and validation_split = 0.2 as was mentioned previously. The result was depicted as follows:

<p align="center">
  <img src="https://user-images.githubusercontent.com/90570944/162838301-153cd78a-8394-4ac6-8ea7-e5c5a9f8bd97.png">
</p>

The generalization gap was kept under control throughout the entire process of validation by using a dropout layer and the loss function for the training data was progressively reduced going from 0.04 on the first epoch up to 0.005 on the last epoch, similarly, the validation loss function had a resembling performance and it went from 0.0193 on the first epoch up to 0.0038 on the last epoch.

Making predictions and visualizing the results

The model has been trained and it is the moment to put in contrast the model predictions against real data, for that purpose we used the test data to make comparisons about the model performance when it runs into unseen data. And basically, we repeated the preprocessing part for the test set and we concatenated trained and test data to run the model, plot it and see the differences between real data and the results of our regressions. A key aspect here is that the outcomes needed to be transformed back into real temperature range values so the scaling process was applied but this time inversely. this is what the plot looks like:

<p align="center">
  <img src="https://user-images.githubusercontent.com/90570944/162838610-00905e01-894b-4241-9dc3-9a4c4c3d321f.png">
</p>

In general, this model generalizes pretty well the patterns found on the data, however, the peaks are in most of the cases unreached meaning that the model is accurate but it is still subject to enhancements.

Tunning the Neural Network

To find out what are the best parameters and their scores we applied a technique called Grid Search where several parameters are assigned a set of values in order to optimize a certain process. To be more precise, for this model the parameters we wanted to fine-tune were the number of epochs (a list with the following values:50, 100, 150) and optimizers (a list with two optimizers: 'adam' and 'rmsprop), the scoring measure utilized was negative mean squared error and the number of cross-validation was 3. The Grid Search was carried out on the Train data and the process shows these results:

best parameters:
 {'epochs': 100, 'optimizer': 'adam'}

best accuracy:
 -0.0034007596930891028
