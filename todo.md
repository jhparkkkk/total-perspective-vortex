dataset: EEG Motor Movement/Imagery Database de PhysioNet 


EEG collectes lors d'experience ou des sujets realisaient ou imaginaient des mouvements de la main u du pied. 


goal: implementer un algorithme de reduction de dimension pour permette un classifier de trouver une reponse dans des donnees qui se retrouvent dans un espace dimensionnel.

Hz = nb oscillations / secondes

I. Process EEG datas (parsing and filtering)
- [ ] load dataset
- [ ] visualize raw data
- [ ] filter frequencies
- [ ] visualize preprocessed data


the power of the signal by frequency and by channel
-> energu
-> **transformee de Fourier** (FFT): convertir un signal du domaine temporel en domaine frequentiel
    -> permet d'identifier quelles frequences dominent dans un signal

II. Implement a dimensionality reduction algoithm

III. Use the pipeline object from scikit-learn

IV. Classify a data stream in 'real time'


[[https://www.youtube.com/watch?v=B3u57yF2JSc]]

