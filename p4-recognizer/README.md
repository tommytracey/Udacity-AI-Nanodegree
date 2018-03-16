#### Udacity Artificial Intelligence Engineer Nanodegree
## Probabilistic Models
# Term 1 / Project 4: Sign Language Recognition System

##### &nbsp;

## Goal
The overall goal of this project is to build a word recognizer for American Sign Language video sequences, demonstrating the power of probabalistic models.  In particular, this project employs  [hidden Markov models (HMM's)](https://en.wikipedia.org/wiki/Hidden_Markov_model) to analyze a series of measurements taken from videos of American Sign Language (ASL) collected for research (see the [RWTH-BOSTON-104 Database](http://www-i6.informatik.rwth-aachen.de/~dreuw/database-rwth-boston-104.php)).  

In this video, the right-hand x and y locations are plotted as the speaker signs the sentence.
[![ASLR demo](http://www-i6.informatik.rwth-aachen.de/~dreuw/images/demosample.png)](https://drive.google.com/open?id=0B_5qGuFe-wbhUXRuVnNZVnMtam8)

The raw data, train, and test sets are pre-defined. Students must derive a variety of feature sets, as well as implement three different model selection criterion to determine the optimal number of hidden states for each word model. Finally, students must implement the recognizer and compare the effects the different combinations of feature sets and model selection criteria.  
## Results
- All of the work for this project is outlined in [this Jupyter notebook (asl_recognizer.ipynb)](https://github.com/tommytracey/Udacity-AI-Nanodegree/blob/master/p4-recognizer/asl_recognizer.ipynb)
- The code can be found in these two files: [my_model_selectors.py](https://github.com/tommytracey/Udacity-AI-Nanodegree/blob/master/p4-recognizer/my_model_selectors.py), [my_recognizer.py](https://github.com/tommytracey/Udacity-AI-Nanodegree/blob/master/p4-recognizer/my_recognizer.py)


##### &nbsp;

---

# Project Starter Code
In case you want to run this project yourself, below is the project starter code. 


### Install

This project requires **Python 3** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/0.17/install.html)
- [pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [jupyter](http://ipython.org/notebook.html)
- [hmmlearn](http://hmmlearn.readthedocs.io/en/latest/)

Notes: 
1. It is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python and load the environment included in the "Your conda env for AI ND" lesson.
2. The most recent development version of hmmlearn, 0.2.1, contains a bugfix related to the log function, which is used in this project.  In order to install this version of hmmearn, install it directly from its repo with the following command from within your activated Anaconda environment:
```sh
pip install git+https://github.com/hmmlearn/hmmlearn.git
```

### Code

A template notebook is provided as `asl_recognizer.ipynb`. The notebook is a combination tutorial and submission document.  Some of the codebase and some of your implementation will be external to the notebook. For submission, complete the **Submission** sections of each part.  This will include running your implementations in code notebook cells, answering analysis questions, and passing provided unit tests provided in the codebase and called out in the notebook. 

### Run

In a terminal or command window, navigate to the top-level project directory `AIND_recognizer/` (that contains this README) and run one of the following command:

`jupyter notebook asl_recognizer.ipynb`

This will open the Jupyter Notebook software and notebook in your browser which is where you will directly edit and run your code. Follow the instructions in the notebook for completing the project.


### Additional Information
##### Provided Raw Data

The data in the `asl_recognizer/data/` directory was derived from 
the [RWTH-BOSTON-104 Database](http://www-i6.informatik.rwth-aachen.de/~dreuw/database-rwth-boston-104.php). 
The handpositions (`hand_condensed.csv`) are pulled directly from 
the database [boston104.handpositions.rybach-forster-dreuw-2009-09-25.full.xml](boston104.handpositions.rybach-forster-dreuw-2009-09-25.full.xml). The three markers are:

*   0  speaker's left hand
*   1  speaker's right hand
*   2  speaker's nose
*   X and Y values of the video frame increase left to right and top to bottom.

Take a look at the sample [ASL recognizer video](http://www-i6.informatik.rwth-aachen.de/~dreuw/download/021.avi)
to see how the hand locations are tracked.

The videos are sentences with translations provided in the database.  
For purposes of this project, the sentences have been pre-segmented into words 
based on slow motion examination of the files.  
These segments are provided in the `train_words.csv` and `test_words.csv` files
in the form of start and end frames (inclusive).

The videos in the corpus include recordings from three different ASL speakers.
The mappings for the three speakers to video are included in the `speaker.csv` 
file.
