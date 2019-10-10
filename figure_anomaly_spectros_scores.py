import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.mixture import GaussianMixture
from plot_libs import gmm_plot_ellipses
from scipy.io import wavfile
from scipy import signal
import os
from tqdm import tqdm
import pickle
from sklearn import preprocessing
import matplotlib.colors as colors

matplotlib.rcParams.update({'font.size': 20})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


fig = plt.figure(figsize=(10,8))

site_gmm_model_f = 'B10_gmm-5.0days-10comps-diag_audio_moths_sorted_june2019-raw_audioset_feats_1s.pickle'
with open(os.path.join('anomaly_gmms', site_gmm_model_f),'rb') as gmm_savef:
    gmm_model, af_data, labs, dts, uq_ids, clss, mins_per_feat, mins_per_rec_hr = pickle.load(gmm_savef)

anom_exp_dir = 'anomaly_playback_exps'

data_dir = 'anomaly_demo_sounds'
all_wav_fs = ['just_chainsaws.wav','B10-1m-chainsaws.wav','B10-10m-chainsaws.wav','B10-50m-chainsaws.wav']
dists = ['Original', '1m','10m','50m']

splot_idx = 1

tot_spec = []

for wav_f,dist in zip(all_wav_fs,dists):
    fig.add_subplot(len(all_wav_fs),1,splot_idx)
    splot_idx += 1

    t = plt.text(25,5,'{}'.format(dist))
    t.set_bbox(dict(facecolor='white', edgecolor='white'))

    sr, wav_data = wavfile.read(os.path.join(data_dir,wav_f))
    dur = len(wav_data) / sr
    f, t, Sxx = signal.spectrogram(wav_data, sr)

    secs_scale = dur / Sxx.shape[1]

    plt.imshow(Sxx, aspect='auto', origin='lower', cmap='winter', norm=colors.LogNorm(vmin=Sxx.min(), vmax=Sxx.max()))
    #plt.pcolormesh(t, f, tot_spec)
    if splot_idx != len(all_wav_fs)+1:
        plt.gca().axes.get_xaxis().set_ticks([])
    else:
        ticks = plt.gca().get_xticks() * secs_scale
        ticks = [np.round(t,2) for t in ticks]
        plt.gca().set_xticklabels(ticks)

    if wav_f != 'just_chainsaws.wav':
        pickle_f = wav_f.replace('.wav','.pickle')
        with open(os.path.join(anom_exp_dir,pickle_f), 'rb') as savef:
            anom_exp_results = pickle.load(savef,encoding='latin1')
        feat_data = anom_exp_results['raw_audioset_feats_1s']
        feat_data = feat_data.reshape((feat_data.shape[1],feat_data.shape[2]))
        anom_scores = -1 * gmm_model.score_samples(feat_data)

        offset = 510
        anom_scores = np.log(anom_scores + offset)
        anom_scores = (anom_scores - 0.34) * ((Sxx.shape[0]-2) / 11)

        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, Sxx.shape[1]-1))
        xs = min_max_scaler.fit_transform(np.asarray(range(len(anom_scores))).reshape(-1, 1))
        xs = xs[:-1]
        anom_scores = anom_scores[:-1]
        plt.plot(xs+(0.49/secs_scale),anom_scores,c='darkred',linewidth=3)

    ax = plt.gca()
    ax2 = plt.gca().twinx()

    if splot_idx == 2:
        ax.set_yticks(ax.get_ylim())
        f_khz = [int(np.round(f_hz/1000)) for f_hz in f]
        ax.set_yticklabels([f_khz[0],f_khz[-1]])
        ax.set_ylabel('Frequency (kHz)')
    else:
        ax.set_yticks([])

    ax2.set_yticks([])
    if splot_idx == 3:
        ax2.set_ylabel('Anomaly score')

ax.set_xlabel('Time (secs)')

plt.tight_layout()
plt.savefig('figs/figure_anom_spectros_scores.pdf')
plt.show()
