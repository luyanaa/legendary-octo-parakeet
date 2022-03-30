from tabnanny import verbose
import mne
import os
import pickle
import numpy

tmin = -0.2  # start of each epoch (200ms before the trigger)
tmax = 0.8  # end of each epoch (600ms after the trigger) - longest beat is 0.57s long
detrend = 0 # remove dc

fname_raw = ['P01-raw.fif', 'P04-raw.fif', 'P05-raw.fif', 'P06-raw.fif', \
    'P07-raw.fif', 'P09-raw.fif', 'P11-raw.fif', 'P12-raw.fif', 'P13-raw.fif',
    'P14-raw.fif']
mne_path = "~/repo/openmiir/eeg/mne"

epoch = numpy.zeros([0, 52, 513])
label = numpy.zeros(0)

channel = numpy.linspace(0, 63, 64)
channel = channel.astype(int)
print(channel)


for path in fname_raw:
    raw = mne.io.read_raw_fif(os.path.join(mne_path, path), preload=False, verbose=False)
    eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,  exclude='bads')
    eeg_picks = numpy.delete(eeg_picks, numpy.argwhere(eeg_picks>63))
    channel = numpy.intersect1d(channel, eeg_picks)

for path in fname_raw: 
    raw = mne.io.read_raw_fif(os.path.join(mne_path, path), preload=True)
    events = mne.find_events(raw, stim_channel='STI 014', shortest_event=0)
    raw.filter(0.5, 30, picks=channel, filter_length='10s', l_trans_bandwidth=0.4, h_trans_bandwidth=0.5,\
         method='fft', verbose=False)
    event_id = None # any
    this_epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                    picks=channel, baseline=None, preload=True, verbose=False)
    y = this_epochs.events[:, -1]
    for i in range(0, this_epochs.events.shape[0]):
        print(this_epochs.events[i])

    X = this_epochs.get_data() * 1000.0
#     print(y.shape)
#     print(X.shape)
#     if label.shape == (0,):
#        label = y
#     else :
#         label = numpy.hstack((label, y))
#     if epoch.shape == (0, 52, 513):
#         epoch = X
#     else:
#         epoch = numpy.concatenate((epoch, X), axis=0)

# print(label.shape)
# print(epoch.shape)

# label_file = open('./preprocessed/label', 'wb')
# pickle.dump(label, label_file, 0)
# label_file.close()

# epoch_file = open('./preprocessed/epoch', 'wb')
# pickle.dump(epoch, epoch_file, 0)
# epoch_file.close()

