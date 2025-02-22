# Content Based Recommendation-System for Movies  
## Lumaa Spring Internship Challenge
#### by: Devin Vliet



## Overview  

This content-based recommendation system suggests five movies based on a user's input, by leveraging movie overviews and genres,
to determine the most applicable movies based on the context gathered from the user's input. Given a user's input, the system
embeds the input into a 768-dimensional vector using the "all-mpnet-base-v2" model from Hugging Face's Sentence Transformers library.
The system compares this input vector in a two fold manner, first to the centroids of clusters generated over the embeddings of the
movies' genres, and secondly to the embeddings of each movie associated with the genre's within each cluster. The cluster whose centroid
has the greatest "semantic similarity" with the user's input is selected, and k-genres from that cluster are selected respectively.
Then the embedding of the overview of every movie that belongs to each of those k-genres, is compared to the user's input and the
model selects k most-similar movies as it's output. These movies are then re-ranked by their similarity scores and returned to the user. 


#### Dataset 
This recommendation system uses a subset of Kaggle's "The Movies Dataset", which contains metadata for
over 45,000 movies listed in the "Full MovieLens Dataset". In order to keep the datasets dimensions in line with 
the expectations for this challenge, the data was prepared by first dropping any missing rows, and repeatedly
truncating the dataset by a couple of features, here are the steps should you want to replicate the dataset yourself:  

- The top 1,000 entries ranked by popularity were selected
- The top 750 entries ranked by total revenue were selected
- The top 650 entries ranked by the average viewer score were selected
- The remaining 650 entries were randomly downsampled to 500 entries.

After downsizing the dataset sufficeintly, all features were dropped aside from "title", "overview", "vote average" and "genre".
The genre feature was originally formatted as a list of dictionaries, I unpacked the genre feature into a single string of each
genre seperated by spaces. I have attached a link to the original dataset below.  

- Link: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

#### Embedding Model
The embedding model used for this system is "all-mpnet-base-v2" model from Hugging Face's Sentence Transformers library, additionally  
the util.community_detection and cos_sim methods are used for the clustering of the genre embeddings.

- Link: https://huggingface.co/sentence-transformers/all-mpnet-base-v2

#### Displaying Outputs
Personally I feel as if interfaces that are located in a terminal are not very classy, and leave something to be desired. Since the goal 
of this challenge was to build a simple recommender system, the user interface options are fairly limited. I wanted something I could be proud of
so I opted to use pythons 'Rich' framework. Rich allows you to present terminal outputs in interesting and fun ways using their "Console" methods.
I have attached a link to the docs:  
- Link: https://rich.readthedocs.io/en/stable/introduction.html 


## Setup & Instructions

To run this system yourself, you need a virtual environment running Python 3.12.4 and the following modules installed:  
- numpy
- pandas
- torch
- sentence_transformers
- rich

### Step 1: Clone the Forked Repository
First clone the forked repository to your local machine:  
``` bash
git clone https://github.com/DevinV-sudo/lumaa-spring-2025-ai-ml.git
cd lumaa-spring-2025-ai-ml
```

### Step 2: Create Virtual Environment
To create a virtual environment to run the system in enter the following:  

```bash
# Create a virtual environment
python3.12 -m venv myenv

# Activate the virtual environment
source myenv/bin/activate  # macOS/Linux
myenv\Scripts\activate     # Windows
```

### Step 3: Install Dependencies
Now that the prior steps are completed, simply install the required modules with the following: 

```bash
# Install requirements
pip install -r requirements.txt
```  

### Step 4: Run the Model!
Once you have a virtual environment up and running that meets the prior specifications, all you need to do is enter the following command:

```bash
python model.py run
```

If you have completed the previous steps successfully, you should see the following outputs in your terminal:
![Local Image](images/Screenshot%202025-02-21%20at%207.03.12%20PM.png)  

The first time you run the code, it may take a minute before the above response appears, this is simply due to the fact that the model must
be loaded from hugging face, after the first time it will load much quicker. Additionally the "Embeddings Loaded" response simply means that the
two JSON files which contain the movie-metadata embeddings are already present in the 'data' folder, if they are not the model will load them for you.

After the model is loaded the foloowing will appear:  
![Local Image](images/Screenshot%202025-02-21%20at%207.06.35%20PM.png)  

All you need to do now is type in a description of the sort of movie you would like to see!  
Heres an example:  

![Local Image](images/Screenshot%202025-02-21%20at%207.08.50%20PM.png)  

If you would like to enter another prompt simply type: 'y' if you are done using the model simply type: 'n'.

## Tutorial & Example Usage:

Here is a simple tutorial detailing the outputs of the model and how it works as well as potenial short comings:

[![Watch the Demo](https://img.youtube.com/vi/gOU8CCw6_GQ/0.jpg)](https://youtu.be/gOU8CCw6_GQ)


### Salary Expectations
---
I thoroughly enjoyed the internship challenge, and I hope you enjoy my response!
As far as the monthly salary expectations, I wasn’t sure where I was expected to state them. I personally believe that as a senior studying data science, math and computer science, experience is incredibly important. I have spent some significant time developing my own AI integrations and systems, as well as studying deep learning theory at university and frankly there is no aspect of data science I love more.
 
This internship would allow me to continue to grow in the field as well as work on problems I am already attempting to solve in a space that I am passionate about. I think a fair monthly salary considering all of those things, would be somewhere in the $2,000-2,700 range, but I cannot emphasize enough that the opportunity to dig my teeth into more problems in this domain is what excites me about this potential role, and should I be selected I am more than happy to negotiate!
 
Thank you for considering me for the role, I hope to hear from you soon.


