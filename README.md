# BiLSTM-Signal-Prediction
ECG signal prediction with the support of SCG frequency domain features using BiLSTM

# 1-1 Introduction
SCG is the measure of the precordial vibrations produced at every heartbeat by the cardiac contraction and relaxation. The typical SCG waveform, illustrated in Fig. 1, is characterized by a number of peaks and troughs. In particular, some of these displacements are associated with the opening and closure of the aortic valve, AO and AC, and the opening and closure of the mitral valve, MO and MC.

In this project we are going to work on SCG signals obtained from the body through the use of machine learning techniques to predict some of parameters like Heartrate (H), Respiratory rate (R), Systolic (S) and Diasytic (D). The goal of machine learning, after observing a set of examples, is to learn regularities from these examples through the use of computational methods in order to make predictions on a future set of data. In this project we use Supervised deep learning Regression algorithms which is concerned with accurately modeling the mapping of inputs to outputs. After training with a set of inputs labeled with their actual outputs, a supervised learning algorithm can then produce predictions of the output for a new set of inputs. I used LSTM network for deep learning and I could reach to MAE below 5 for four paramters and a score morethan 95.
This report consists of multiple parts to accomplish this project including:
1-	The raw data first goes through an initial preprocessing to remove noise and abnormal events in signal
2-	The data then undergoes what is called feature extraction where the input goes through a transformation into a new representation of inputs. 
3-	Finally, the input is fed to the machine deep learning algorithm to make a model and then use this for prediction. Also before using parameters, I checked labels to discard labels with R=0.

#1-2 Preprocessing section
SCG signal is sampling with sampling rate 100Hz. At first I tied to remove out of band noise by a band pass Butterworth with order 5. In Fig.2 the figure of frequency response of signal before and after filtering is represented.

After filtering, I tried to remove abnormal signals by a technique like supervision moving average. By splitting data to segments with 100 samples (which is approximately equal to the number of samples in one SCG signal period) and using moving average technique with order 8, I removed abnormal events by calculation variance of every segmented moved average signal and normalizing signal by calculated variance. 

At the end for gaining high accuracy, before feature extractions ( to have more features for one signal duration), I used adaptive segmentations technique to divide each SCG cycle into bins with various sample data. However, binning of the signal was performed discriminately, where areas of the signal corresponding to higher variation received a higher concentration of bins. In this project I used of 4 bins length for every SCG cycle.
