import librosa
import mne

# Resampling all input into 22050Hz. 
def MusicSTFT(filename):
    chroma = []
    for file in filename:
        y, sr = librosa.load(file)
        # This configuration refers to Large-scale brain networks emerge from dynamic processing of musical timbre, key and rhythm, NeuroImage(2012), Alluri et al. 
        # n_fft = 2048 causes window length of 93ms, so that we can use small n_fft to build new stimulus
        chroma.append(librosa.feature.chroma_cqt(y=y, sr=sr))
    return chroma

def eegSTFT(filename):
    for file in filename:
        raw = mne.io.read_raw_fif(file, preload=False, verbose=False)
        
        # 
        eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,  exclude='bads')
        eeg_picks = numpy.delete(eeg_picks, numpy.argwhere(eeg_picks>63))
        channel = numpy.intersect1d(channel, eeg_picks)
    
    events = mne.find_events(raw, stim_channel='STI 014', shortest_event=0)
    event_id = None # any
    this_epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                    picks=channel, baseline=None, preload=True, verbose=False)
    y = this_epochs.events[:, -1]
    mne.time_frequency.stft()