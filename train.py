import pickle

from tensorflow import keras
import EEGModels
import tcn
from sklearn.model_selection import train_test_split

label_file = open('./preprocessed/label', 'rb')
epoch_file = open('./preprocessed/epoch', 'rb')

label = pickle.load(label_file)
epoch = pickle.load(epoch_file)

epoch_train, epoch_test, label_train, label_test = train_test_split(epoch, label, test_size=0.2, random_state=19260817)
epoch_train, epoch_val, label_train, label_val = train_test_split(epoch_train, label_train, test_size=0.1, random_state=19260817)

model = EEGModels.EEGNet()