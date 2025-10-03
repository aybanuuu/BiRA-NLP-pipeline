# BiRA-NLP-pipeline

This repository contains model pipelines and other relevant files from our capstone project "BiRA: BiLSTM-powered AI Reading Aid as an Instructional Tool for Teaching Reading in Filipino to Grade 1 Pupils".
## Table of Contents

- [Overview](#Overview)
- [Dataset](#Dataset)
- [Pipelines](#center)
- [Results](#Results)
- [Acknowledgements](#Acknowledgements)
## Overview
BiRA is a device that aims to aid Grade 1 learners in refining their reading skills in Filipino, particularly pronunciation/phonics. The device bridges the gap between teachers and learners, allowing for a more interactive learning experience.

The BiRA device is a fully functional system in a tablet form factor that consists of a Raspberry Pi 5 (8gb version) equipped with a: 
* Camera Module
* Microphone
* USB Sound Card (for plugging the microphone and either headsets or speakers)
* Touch Screen

Software-wise, it uses a BiLSTM model with CTC loss optimization for analyzing pronunciation in conjunction with a pronunciation lexicon and a [Levenshtein](https://pypi.org/project/python-Levenshtein/) alignment algorithm that evaluates the learner's pronunciation. The whole ML pipeline is contained inside a Tkinter GUI, which also contains functions such as OCR to scan words being studied for pronunciation, a gamified UI that classifies words based on difficulty, and opportunities to study the pronunciation of a word or utterance first before trying out the AI evaluation model.

This repository will only contain the project's NLP pipeline, specifically dataset augmentation, [MFA](https://montreal-forced-aligner.readthedocs.io/en/latest/index.html) acoustic model (for phoneme transcriptions), phoneme lexicon, and the full ML pipeline. Datasets and model metric results will also be described, but the dataset and model themselves will not be made publicly available due to the agreement made by the researchers and the speaker who lent their voices for this project regarding privacy concerns.


## Dataset
The main dataset consists of 18,632 files, with one (1) female adult speaker lending her voice and three (3) pseudo speakers simulating child voices, all based on the female adult speaker's voice. This was done instead of obtaining actual child voices due to the lack of datasets publicly available and time constraints.

The original recording was recorded in a moderately quiet environment using a microphone similar to what the BiRA device uses. To add augmented pseudo-child speakers, the original audio files were augmented by doing a combination of pitch and formant shifting in Python's [PyWorld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) library. Resources for the voice recording come from DepEd's Marungko Booklets being used for teaching reading inside the classroom (used with permission).

Speaker breakdown and duration of each are as follows:
| Speaker ID  | Description | Duration | Pitch Shift Ratio | Formant Shift Ratio
| ----------- | ----------- | ----------- | ----------- | ----------- |
| BR01      | Female adult voice (original) | 6 hrs 18 mins 24 secs | ------- | -------
| BR02      | Main child voice (male) | 7 hrs 55 mins 22 secs | 1.315 | 1.21
| BR03      | Slightly older child voice | 7 hrs 55 mins 22 secs  | 1.2 | 1.1
| BR04      | Female child voice | 7 hrs 55 mins 22 secs  | 1.425 | 1.305
| **TOTAL** |                     | **30 hrs 04 mins 30 secs**

**Augmentation**  
To improve dataset diversity and model robustness in real-world conditions, a series of augmentations were introduced using Python's [PyWorld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder), [SoX](https://pysox.readthedocs.io/en/latest/index.html), [Pyroomacoustics](https://github.com/LCAV/pyroomacoustics?tab=readme-ov-file), and [NumPy](https://numpy.org/devdocs/) (noise injection). For BR01 (female adult):
| Augment No. | Augment Type | Description |
| ----------- | -----------  | ----------- |
| 1   | None | ORIGINAL |
| 2   | Pitch Shift | +1 semitone |
| 3   | Time Stretch + Noise | 1.1x + 20dB |
| 4   | Time Stretch | 0.9X |
| 5   | Noise | 15dB |
| 6   | Reverberation | Reverb 1 (well-treated room) |
| 7   | Noise + Reverberation | 25dB + Reverb 2 (room with thin walls) |

For BR02 to BR04 (pseudo-child speakers):
| Augment No. | Augment Type | Description |
| ----------- | -----------  | ----------- |
| 1   | None | ORIGINAL |
| 2   | Pitch Shift | +2.4 semitones |
| 3   | Time Stretch + Noise | 1.1x + 20dB |
| 4   | Time Stretch | 1.2x |
| 5   | Time Stretch + Pitch Shift | 0.95x + +2.4 semitones |
| 6   | Noise | 15dB |
| 7   | Reverberation | Reverb 1 (well-treated room) |
| 8   | Noise + Reverberation | 20dB + Reverb 2 (room with thin walls) |
| 9   | Noise + Reverberation | 20dB + Reverb 3 (old room with strong echo/reverb) |

Another dataset was used for training the MFA Acoustic Model to obtain phoneme transcriptions. This dataset uses only the original speaker's audio files with less aggressive augmentation (+1 and -1 semitones, 0.95x to 1.05x time stretch, 15dB noise), totalling 5 hrs 19 mins 58 secs.

**Pronunciation lexicon**  
The lexicon for the dataset was created by using a modified subset of the FlipVox pronunciation lexicon version 1.0 ([F7Xdict](https://github.com/flipvox/F7Xdict) repository), which is based on the format of the CMU pronunciation dictionary. The lexicon contains 1,332 words.
## ML Pipeline
The BiRA project implements a full ML pipeline for speech recognition and pronunciation evaluation:

**Data Loading**  
   - Loads audio files and phoneme transcriptions (TextGrid format).

**Preprocessing & Feature Extraction**  
   - Applies resampling (16kHz, 16-bit mono), pre-emphasis, and normalization. 
   - Performs VTLN-based warping for speaker variability.  
   - Extracts CMVN-normalized MFCCs, Δ MFCCs, ΔΔ MFCCs, pitch, and energy
   - Feature Extraction Configuration:

| Parameter | Value |
| ----------- | ----------- |
| VTLN Warp Factor      | 0.85 |
| Pivot Frequency   | 1500 Hz |
| # of MFCCs      | 39 (13 MFCCs, 13 Δ, 13 ΔΔ) |
| FFT Windows      | 512 |
| Window Length      | 400 |
| Window Type     | Hamming |
| Hop Length      | 160 |
| # of Mel Filterbanks  | 26 |

**Model Training & Testing**  
   - **Model Architecture**: BiLSTM-CTC (41 input layers, 128 hidden layers, 44 output layers) with dropout regularization.  
   - **Training Loop**: CTC loss with entropy regularization, gradient clipping, warmup scheduling.  
   - **Validation & Testing**: Evaluates with Levenshtein distance and compute Phoneme Error Rate (PER).  
   - A **70-15-15 split** was done for training, validation, and testing.

**Pronunciation Evaluation**  
   - Implements stress-aware Levenshtein alignment.  
   - Scores predictions with phoneme-level feedback (0 for errors, 0.5 for stress mismatches, 1 for correct).  

**Testing on Lone Audio Samples**  
   - Run inference on unseen audio.  
   - Provide phoneme-level alignment visualization and pronunciation feedback.

## Results

Three BiLSTM-CTC models with varying hyperparameters (dropout, learning rate, batch size, weight decay, regularization) were evaluated. Further testing of more models was not possible due to time constraints in the completion of this project.  

Key results (test set) are summarized below:

| Model | Dropout | Learning Rate | Batch Size | PER   | Accuracy | Precision | Recall | F1 Score |
|-------|---------|---------------|------------|-------|----------|-----------|--------|----------|
| M1    | 0.3     | 1e-4          | 8         | 21.40% | 78.60%    | 75.90%      | 65.26%   | 67.49%     |
| M2    | 0.3     | 1e-4          | 8         | 17.66% | 82.34%    | 70.31%      | 63.94%  | 65.86%     |
| M3    | 0.4     | 1e-3          | 16         | 16.37% | 83.63%    | 74.78%      | 66.69%   | 68.81%    |

The hyperparameters used for each model were:

| Model  | LR Warmup | LR Decay | Weight Decay (AdamW) | Entropy | No. of Epochs |
|---------|-----------|----------|--------------|---------|---------------|
| M1 | No        | No       | -            | -       | 20            |
| M2 | Yes       | No       | -            | 0.0010  | 23            |
| M3 | Yes       | Yes      | 0.0001       | 0.0001  | 11            |

*Note: Early stopping triggered on M2 and M3 (patience at 5).*

Additional visualizations (see [`results/`](results/)):
- **Confusion Matrix** 
- **Phoneme-level F1 Scores** 
- **Training/Validation Loss Curves** 
*Note: For the accuracy metric, the Levenshtein edit distance accuracy (100 - PER) was considered instead of classification accuracy (sklearn) since the researchers were focused more on actual use case accuracy (finding the distance between the reference and predicted phoneme) than classification accuracy. Kindly disregard sklearn accuracy in the **phoneme_classification_report** CSV files.*

Overall, the best-performing model (M3) achieved:  
- **Phoneme Error Rate (PER)**: 16.37%  
- **Accuracy**: 83.63%

**Things to consider:**
- The relatively large accuracy-precision-recall-f1 gap is to be expected since the custom dataset has limited variability, having only 1 real speaker and three artificial child speakers.
- The Marungko booklets contain common Filipino words that are easy to pronounce for Grade 1 learners. Therefore, it has limited phoneme variability on phonemes commonly found in more complex Filipino words (as evidenced by the phoneme-level f1 scores).
- Since phoneme AA (IPA: a) is the most common phoneme in the Filipino language, differentiation between stress symbols can be potentially challenging (as evidenced by the confusion matrix).
- Despite these limitations, the model and device performed well on real-world scenarios as assessed inside a classroom setting by the researchers and users.

**Points of Improvement**  
With these results and considerations, the model can be improved by:
- Introducing more natural speakers, preferably actual child speech.
- Introducing audio speech for less evident phonemes, oversampling the rare phonemes and making phoneme distribution more balanced.
- Further hyperparameter experimentation (hidden layers, learning rate, batch size, regularization).
- Further experimentation on training with stress symbols, possibly adding stress as an additional feature.

## Acknowledgements

This project was developed as part of the Capstone Project at Colegio de Muntinlupa.  
We would like to thank our adviser, Engr. Tristan Jay Calaguas, as well as Dr. Cyd Laurence Santos and Engr. Ricrey Marquez, for guidance throughout the project.  

Special thanks to my teammates:  
- Lord Brix Pimentel for dataset collection, noise reduction, and paper documentation
- Michael John Luis Ramirez for hardware integration and system prototyping

We also thank Federico Ang for making his FlipVox pronunciation lexicon repository (F7Xdict) publicly available.

For additional inquiries regarding this project or if you want to point out any issues, kindly reach out to me via email found in my profile.