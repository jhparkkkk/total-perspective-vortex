dataset: EEG Motor Movement/Imagery Database de PhysioNet
[[https://physionet.org/content/eegmmidb/1.0.0/]]
EEG collectes lors d'experience ou des sujets realisaient ou imaginaient des mouvements de la main u du pied.

goal: implementer un algorithme de reduction de dimension pour permette un classifier de trouver une reponse dans des donnees qui se retrouvent dans un espace dimensionnel.

-> créer une interface cerveau-ordinateur basée sur des données EEG en utilisant des algo de ml.
-> deviner l'action ou la pensée d'un sujet à partir de la lecture d'un EEG sur une période donnée

---

.edf (European Data Format): format standardisé pour stocker des recordings de signaux biologiques comme EEG, ECG

.edf.event: description des évènements
If you really need to use the original data files, note that the EDF files already contain events in annotation channels (accessible by raw.annotations) - you do _not_ need to load the .event files because they contain the same events that are already present in the EDF files.

channels : position des électrodes dans le cuir chevelu (système international 10-20)

- F, C, P, O, T, Fp
- 1, 3, 5, 7 à gauche
- 2, 4, 6, 8 à droite
- z au centre

Ici, 64 canaux EEG sont inclus

fréquence d'échantillonage: par ex si la fréquence est de 160 Hz alors ça veut dire que les données EEG sont enregistrées 160 fois par seconde pour chaque canal -> 160 points de données.

L'activité cérébrale typique se situe autour de 80 Hz

Hz = nb oscillations / secondes

lowpass, highpass = fréquences limites de la capture des signaux

event[start(int), previous_event_id(int), event_id(int)]
par exemple event[0, 0, 1]
l'event a demarré à l'échantillon 0, l'event est le 1er il n'y en a pas eu avant, id de l'event actuel.

event id :

- T0 : repose
- T1 :
  - si run

onset(sample): position in the data stream, can be converted to seconds by dividing it by the sampling frequency

viz:

- signaux eeg en fonction du temps. Un tracé repré l'amplitude du signal EEG mesuré en microvolt (μV) -> variation du voltage en fonction du temps
- events sont marqués par les plages chromatiques (T0, T1, T2)
  -> comparer activité eeg de tous les canaux
  -> un motif associé à une task
  -> séquences temporelles de l'expérience

fourier transform : pour obtenir le spectre de fréquence

N_FFT = résolution fréquentielle.

Power Spectral Density (PSD)

Independant Component Analysis (ICA)

- technique for estimating independent source signals from a set of recordings in which the source signals were mixed together in unknown ratios.

Principal Component Analysis (PCA) : simplify EEG data, reduce noises, reduce dimensionality

I. Preprocessing, parsing, and formatting:

- [x] load dataset
  - [x] exclude bads ? to test
- [x] visualize raw data (before)
- [x] filter by useful frequencies
- [x] visualize preprocessed data (after)

the power of the signal by frequency and by channel
-> energu
-> **transformee de Fourier** (FFT): convertir un signal du domaine temporel en domaine frequentiel
-> permet d'identifier quelles frequences dominent dans un signal

II. Implement a dimensionality reduction algorithm

- [ ] implement my own PCA to transform the data into meaningful features

III. Use the pipeline object from scikit-learn

- [ ] create pipeline using dimensionality reduction (my own PCA)
- [ ] add classifier for motion detection task
- [ ] simulate streaming with playback mechanism

IV. Train, Validation and Test

- [ ] cross-validation to evaluate pipeline
- [ ] split data into training/validation/test sets
- [ ] minimum accuracy of 60% on the test set

A l'aide je ne comprends rien:

[[https://www.youtube.com/watch?v=B3u57yF2JSc]]

[[https://towardsdatascience.com/a-step-by-step-implementation-of-principal-component-analysis-5520cc6cd598]]

[[https://neuraldatascience.io/7-eeg/time_freq.html]]

[[https://arxiv.org/html/2405.01269v1]]

about channels selection:
[[https://arxiv.org/pdf/1312.2877]]

FC3, FCZ, FC4, C3, C1, CZ,
C2, and C4

```
python mybci.py plot --subject 4 --run 14
python mybci.py train --subject 1 --run 4
python mybci.py train
python mybci.py predict --subject 2 --run 6
```

BaseEstimator
standardiser l'interface des objets de scikit-learn
méhode .fit() .get_params()

fit: Prepares or "fits" the step (e.g., learns parameters, initializes state).
transform: Applies a transformation to the input data and passes the transformed data to the next step.
Donc si je veux utiliser BaseEstimator et TransformerMixin je dois implémenter fit et transform dans ma class custom EEGPreprocessor

#### mne

reject_criteria : exclude epochs that contain excessive noise or artifact
reject_criteria = dict(eeg=150e-6)

because EEG signals are prone to artifacts (eye blinks, muscles mvt, external noises)
low treshold : more strict rejection -> clean data with minimum noise
high threshold : less strict rejection -> noisier dataset but retain more epochs for training

_tmin tmax_ in seconds
tmin = starting time of the epoch relative to the event
tmax = ending time of the epoch relative to event
-> when analyzing motor movements the common window is -0.2 and 0.5 or -1.0 and 4.00 ??
[[https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html]]

24.11.2024
problèmes:
- le modele fonction que pour 1 run OU le dataset entier. 
    - liste de run []
- pas du tout opti pour charger tout le dataset.
    - revoir EEGDataLoader.load_data -> au lieu de retourner un objet EEGData, envoyer raw


