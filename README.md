<!-- PROJECT SHIELDS -->
<div align="center">
	<a href =https://github.com/dvdimitrov13/Predict-recipe-cuisine-using-NLPLogo_Detection/blob/master/LICENSE><img src =https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge>
	</a>
	<a href =https://www.linkedin.com/in/dimitarvalentindimitrov/><img src =https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555>
	</a>
</div>

  <h1 align="center">Where is this recipe from?</h1>

  <p align="center">
   In the restaurant business, trends change constantly and n. Not only that, but there are several regional differences. Understanding the market’s preferences and aligning your menu accordingly can be powerful! In this repository, I have developed a pipeline consisting of two models that isare able to read a recipe and predict its cuisine.
  </p>
</div>


<!-- TABLE OF CONTENTS -->
## Table of Contents
<details>
  <ol>
    <li>
      <a href="#main-idea">Main dea/a>
      <ul>
        <li><a href="#environment">Environment</a></li>
        <li><a href="#development">Development</a></li>
        <li><a href="#results">Results</a></li>
      </ul>
    </li>
    <li>
      <a href="#user-manual">User manual</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#prediction">Prediction</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## Main idea
This project was developed as part of my Data Science MSc curriculum, with a goal of exploring two different problems in the realm of Natural Language Processing - structured prediction and classification. 

The idea is to train a Named Entity Recognition model to extract ingredients from raw recipe instructions ehn, thate output will be fed into a classification model that  able to predict the cuisine (currently trained on 20 different cuisines).

* Structured prediction is performed using a Bi-LSTM Recurrent Neural Net 
* Classification is performed using a non-linear Support Vector Machine

### Environment

The development environment for this project was a Ubuntu 20.04 machine with a GTX1060 GPU. This was helpful when training the RNN as tensorflow is able to take advantage of the GPU. However, this is not strictly necessary as the model runs well on any modern computer!

Additionally, in order to make the reproducibility and improvement of this repository as straightforward as possible, I used Git Large File Storage. This allowed for a simple cloning that includes all relevant training data, as well as model weights.

Now I am going to give a high level overview of the projector. For more detail and interesting analysis/visualisations please go through the [jupyter notebook](https://github.com/dvdimitrov13/Predict-recipe-cuisine-using-NLP/blob/master/Non-Attending_Project.ipynb) in the repo! 

<div align="center" style="position:relative;">
    </a>
    <a href="https://git-lfs.github.com/">
        <img src="https://github.com/dvdimitrov13/Logo_Detection/blob/master/images/git_lfs.png" width="60" height="60"/>
    </a>
    <a href="https://www.tensorflow.org/">
        <img src="https://github.com/dvdimitrov13/Predict-recipe-cuisine-using-NLP/blob/master/.github/images/TF.jpg" width="120" height="60"/>
    </a>
    <a href="https://spacy.io/">
        <img src="https://github.com/dvdimitrov13/Predict-recipe-cuisine-using-NLP/blob/master/.github/images/spacy.png" width="120" height="60"/>
    </a>
    <a href="https://flask.palletsprojects.com/en/2.0.x/">
        <img src="https://github.com/dvdimitrov13/Predict-recipe-cuisine-using-NLP/blob/master/.github/images/flask.png" width="120" height="60"/>
    </a>
 
</div>

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
### Development

#### 1. Dataset

Three different datasets were used for this project - 2 for training,and one for the final analysis:

-   A dataset that has recipe ingredients and the cuisine of the recipe - used in the Simple Prediction
-   A new york times dataset with recipes and a list of ingredients used to create the annotated data for Structured Prediction
-   All recipes on allrecepies.com scraped using an API provided by [Apify](https://apify.com/?fpr=x4r3l). This data contains the recipes, as well as their rating which we will use to identify the current "trendy" recipes.

#### 2. Training 
 Training each of the two models consisted of two steps: model selection and hyper-parameter optimization.

For the classification problem a baseline was established by a Regularized Logistic regression using 2-6 ngrams. To improve this method I had a choice between a Convolutional Neural Net and a Support Vector Machine. I opted for a non-linear SVM as it is less computationally intensive and delivers similar performance. In order to pick the best set of hyper-parameters Baysian Optimization was used.

For the structured prediction problem the baseline was established by a structured perceptron based on Collins 2002, and modified for the prediction at task. After that I tested LSTM vs Bi-LSTM and since the Bi-LSTM gave better performance I optimised the hyper-parameters using grid search.

### Results
 
For evaluation I used F1 score, implementing an f1_multiclass function for the structured prediction.  The final results of the best models were as follows:

 Model| Accuracy | Weighted Avg F1 score | Macro Avg F1 score
| :-----: | :-: | :-: | :-: 
| Adidas 		| 0.79 | 0.79 | 0.73 
| Apple Inc. 	| 0.97 | 0.97 | 0.88 


### Installation

Running the models requires a Python 3.9 environment with Tensorflow >= 2.4. Additionally, nltk, spacy and gensim were used for text analysis. Finally, scikit-learn was used for classification, dimensionality reduction, clustering and other implementations.

All required dependencies are located under `reqirements.txt` The cleanest way to test or improve this project would be to set up a virtual environment. Follow this steps to do so:

* First you need Git LFS in order to be able to properly clone the repository
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```
* Next you clone the repository including submodules
 ```
git clone --recurse-submodules https://github.com/dvdimitrov13/Predict-recipe-cuisine-using-NLP nlp
  ```
  * Create a conda environment and activate it
  ```
  cd yolo
  conda env create -n "nlp" python=3.9
  conda activate nlp
  ```
  * Install the requirements using pip
    
  ```
  conda install pip
  pip install -r requirements.txt 
  ```

### Prediction

There are two ways you can go about testing the models on your own recipe. The first and simpler one is to use the web app interface I have provided. To do run the 'flask_app.py' file in your terminal and follow the address that you see in console!
* I am currently working on deploying this web app, after running into RAM issues with heroku I am exploring other deployment options -- stay tuned!

The second option is to use the `predict.py` script as follows. To predict the cuisine of your original recipe you need to follow the insatlation steps after which you can paste your recipe in a custom .txt file or `recipe.txt`  and run the follwing command: 
```
python predict.py --recipe recipe.txt
```


<!-- LICENSE -->
## License

Distributed under the GNU License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Dimitar Dimitrov 
	<br>
Email - dvdimitrov13@gmail.com
	<br>
LinkedIn - https://www.linkedin.com/in/dimitarvalentindimitrov/

<p align="right">(<a href="#top">back to top</a>)</p>
