# BiLSTM-Signal-Prediction
ECG signal prediction with the support of SCG frequency domain features using BiLSTM

# 1-1 Introduction
SCG is the measure of the precordial vibrations produced at every heartbeat by the cardiac contraction and relaxation. The typical SCG waveform, illustrated in Fig. 1, is characterized by a number of peaks and troughs. In particular, some of these displacements are associated with the opening and closure of the aortic valve, AO and AC, and the opening and closure of the mitral valve, MO and MC.

In this project we are going to work on SCG signals obtained from the body through the use of machine learning techniques to predict some of parameters like Heartrate (H), Respiratory rate (R), Systolic (S) and Diasytic (D). The goal of machine learning, after observing a set of examples, is to learn regularities from these examples through the use of computational methods in order to make predictions on a future set of data. In this project we use Supervised deep learning Regression algorithms which is concerned with accurately modeling the mapping of inputs to outputs. After training with a set of inputs labeled with their actual outputs, a supervised learning algorithm can then produce predictions of the output for a new set of inputs. I used LSTM network for deep learning and I could reach to MAE below 5 for four paramters and a score morethan 95.
This report consists of multiple parts to accomplish this project including:
1-	The raw data first goes through an initial preprocessing to remove noise and abnormal events in signal
2-	The data then undergoes what is called feature extraction where the input goes through a transformation into a new representation of inputs. 
3-	Finally, the input is fed to the machine deep learning algorithm to make a model and then use this for prediction. Also before using parameters, I checked labels to discard labels with R=0.

# 1-2 Preprocessing section
SCG signal is sampling with sampling rate 100Hz. At first I tied to remove out of band noise by a band pass Butterworth with order 5. In Fig.2 the figure of frequency response of signal before and after filtering is represented.

After filtering, I tried to remove abnormal signals by a technique like supervision moving average. By splitting data to segments with 100 samples (which is approximately equal to the number of samples in one SCG signal period) and using moving average technique with order 8, I removed abnormal events by calculation variance of every segmented moved average signal and normalizing signal by calculated variance. 

At the end for gaining high accuracy, before feature extractions ( to have more features for one signal duration), I used adaptive segmentations technique to divide each SCG cycle into bins with various sample data. However, binning of the signal was performed discriminately, where areas of the signal corresponding to higher variation received a higher concentration of bins. In this project I used of 4 bins length for every SCG cycle.

# 1-3 Deep learning 
Traditionally, time series forecasting has been dominated by linear methods because they are well understood and effective on many simpler forecasting problems. Deep learning neural networks are able to automatically learn arbitrary complex mappings from inputs to outputs and support multiple inputs and outputs. 
Deep Learning Algorithms For Time Series Forecasting including:

1-	MLPs, the classical neural network architecture including how to grid search model hyperparameters.
2-	CNNs, Simple CNN models as well as multi-channel models and advanced multi-headed and multi-output models.
3-	Hybrids, Hybrids of MLP, CNN and LSTM models such as CNN-LSTMs, ConvLSTMs and more.

A powerful type of neural network designed to handle sequence dependence is called recurrent neural networks. 
The Long Short-Term Memory network or LSTM network is a type of recurrent neural network used in deep learning because very large architectures can be successfully trained.
Unlike traditional recurrent neural networks, LSTM supports time steps of arbitrary sizes and without the vanishing gradient problem. I considered bidirectional LSTM for this problem because it learns how and when to forget and when not to using gates in its architecture. In previous RNN architectures, vanishing gradients was a big problem and caused those nets not to learn so much. Using Bidirectional LSTM, I feed the learning algorithm with the training data once from beginning to the end and once from end to beginning. It usually learns faster than one-directional approach although it depends on the task. In this particular dataset it yielded lower MAE compared to one-directional LSTM.
By setting drop-out rate to 0.5, it also aimed to prevent overfitting. The resutl of traing data is showed in Fig.5. I used this approcah (70%,20%,10%) for dividing training data to traing data, test data and predicting data respectively to make our model and evaluate it. After that I tried new data(next 7min) to predict parameters. By making a model with 20min data as traing I got a score more than 95 (it would be 98 if we increase the size of train data).


# 1-4 References
1-	Thomas Roczink & Robert Bosch.2018. Prediction of electrocardiography features points using seismocardiography data: a machine learning approach
2-	Brian Solar – 2018. A Machine Learning Approach to Assess the Separation of Seismocardiographic Signals by Respiration - University of Central Florida
3-	Xiao Jiang , Gui-Bin Bian and Zean Tian. 2019. Removal of Artifacts from EEG Signals: A Review
4-	Dan Li, Xiaoyuan Ma, 2016, Use Moving Average Filter to Reduce Noises in Wearable PPG During Continuous Monitoring
5-	Siddharth Kohli, Alex Casson, 2015, Removal of Transcranial a.c. Current Stimulation artifact from simultaneous EEG recordings by superposition of moving averages

6-	Marco Di Rienzo, Emanuele Vaini & Prospero Lombardi,2017, An algorithm for the beat-tobeat
              assessment of cardiac mechanics during sleep on Earth and in microgravity from the 
              seismocardiogram

7-	Siddharth Kohli and Alexander J. Casson, 2015. Alternating Current Stimulation in Simultaneous
              EEG Monitoring

8-	Kouhyr Tavakolian, 2013, Characterization and analysis of seismocardiography for estimation of hemodynamic parameters
9-	https://tsfresh.readthedocs.io/en/latest
