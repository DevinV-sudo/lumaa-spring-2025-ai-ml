#dont show warnings from torch import
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

#imports
import pandas as pd
import numpy
import json
from sentence_transformers import SentenceTransformer
import os

#this is a one-time processing script that generates the embeddings for the model to use.
def data_embedding(file_path, model_name):
    '''
    This function takes a file_path and a model_name, it loads the model and dataset
    extracts the two features for semantic comparision from the dataset and embeds them.
    The embeddings are then passed to the save function to be saved as JSON
    '''
    
    #load in the csv file
    movies_dataset = pd.read_csv(file_path).drop(columns="Unnamed: 0")

    #load the embedding model
    print(f"Loading {model_name}\n")
    model = SentenceTransformer(model_name)

    #select the two search features
    genres = [text.strip() for text in movies_dataset["genre"].unique()]
    overviews = [text.strip() for text in movies_dataset["overview"].to_list()]

    #metadata for vectors
    titles = movies_dataset["title"].to_list()
    ratings = movies_dataset["vote_average"].to_list()

    #generate embeddings
    genre_embeddings = model.encode(genres)
    overview_embeddings = model.encode(overviews)
    print(f"generated embeddings for 'genre' and 'overview'\n")
    
    #store the vector with associated meta_data
    genre_list = []
    for idx, genre in enumerate(genres):
        genre_dict = {}
        genre_dict["genre"] = genre
    
        #converting the embedding vectors to lists for JSON compatibility
        genre_dict["movies"] = movies_dataset[movies_dataset["genre"] == genre]["title"].to_list()
        genre_dict["embedding"] = genre_embeddings[idx].tolist()
    
        #append the stored vectors and metadata to the list
        genre_list.append(genre_dict)
        
    #store the vectors with associated meta_data
    overview_list = []
    for idx in range(len(movies_dataset["title"])):
        overview_dict = {}
        overview_dict["title"] = titles[idx]
        overview_dict["rating"] = ratings[idx]
    
        #converting the embedding vectors to lists for JSON compatibility
        overview_dict["embedding"] = overview_embeddings[idx].tolist()
    
        #append the stored vectors and metadata to the list
        overview_list.append(overview_dict)

    return genre_list, overview_list

def save_embeddings(genre_path, overview_path, genre_list, overview_list):
    '''
    This function simply saves the vector embeddings and associated metadata as
    a JSON file to be retreived and used in the model python file.
    '''
    
    #load embeddings as json files
    with open(genre_path, "w") as f:
        json.dump(genre_list, f)
    
    with open(overview_path, "w") as f:
        json.dump(overview_list, f)

    print(f"Successfully saved embeddings as JSON\n")

    return None
    
if __name__ == "__main__":
    #function arguements
    model_name = "all-mpnet-base-v2"
    file_path = "data/movies_dataset.csv"
    genre_path = "data/genre_embeddings.json"
    overview_path = "data/overview_embeddings.json"
    
    # Check if embeddings already exist
    if os.path.exists(genre_path) and os.path.exists(overview_path):
        print("Embeddings already exist. Skipping generation.")

    #If embeddings dont exist - save them
    else:
        print("Embeddings not found. Generating...")
        genre_list, overview_list = data_embedding(file_path, model_name)
        save_embeddings(genre_path, overview_path, genre_list, overview_list)
    
