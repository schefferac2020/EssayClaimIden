# Automated Discourse Identification in Argumentative Essays
Nikki Ratanapanichkich and Andrew Scheffer

[Final Paper](./487FinalPaper.pdf)

<img width="1357" alt="Qualitative" src="https://user-images.githubusercontent.com/54146662/232963053-59b8385e-c60e-49e1-b8a5-4a5e09f3a6d0.png">

## Description
This github page contains the code for our EECS 487 -- Introduction to Natural Language Processing final project. 
Our team focused on developing a transformer based architecture (using BigBird as a backbone) to do discourse segmentation
in student essays -- a critical component of Automated Essay Scoring (AES). Throughout this project, our team developed and finetuned this 
transformer based model alongside a custom Hidden Markov Model (HMM) as a benchmark. 

## Datasets Required 
The code in this repo made heavy use of the [PERSUADE Corpus](https://www.sciencedirect.com/science/article/pii/S1075293522000630) collected by the GSU Learning Agency Lab. 
Exactly 15,594 student essays (at the 6th-12th grade level) were used for training and evaluation in this project. 
Each essay in the PERSUADE corpus was annotated by human raters for argumentative and discourse elements using a double-blind rating process.
To download this dataset please visit [this link](https://www.kaggle.com/competitions/feedback-prize-2021/data)

## Usage
The main files in this repo are [FinalProj.ipynb](./FinalProj.ipynb) and [HMMBenchmark.ipynb](./HMMBenchmark.ipynb).

### FinalProj.ipynb
This jupyter notebook does the main processes of our report including: preprocessing data, simple data visualization, transformer architecture definition, 
model training, and results visualization and testing.

### HMMBenchmark.ipynb
This jupyter notebook simply reads in the generated preprocessed dataframes from the previous notebook to avoid code duplication. Additionally, this notebook does
data visualizations and graphing. This notebook is mainly responsible for the definition, training, and testing of the HMM model.

## Requirements
- [Student Essays Dataset](https://www.kaggle.com/competitions/feedback-prize-2021/data)
- numpy
- pandas
- tqdm
- transformers
- pytorch 
- sklearn