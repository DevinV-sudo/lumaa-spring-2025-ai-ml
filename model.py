#dont show warnings from torch import
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

#other imports
import numpy as np
import json
import torch
import sys
import os
from sentence_transformers import SentenceTransformer, util
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.table import Table
import subprocess


def load_embeddings(genre_path, overview_path):
    ''' load in embeddings generated in embedding.py'''
    with open(genre_path, "r") as f:
        genre_list = json.load(f)

    with open(overview_path, "r") as f:
        overview_list = json.load(f)

    return genre_list, overview_list

def embed_input(input_string, model):
    ''' Embed user input for recommender model'''
    input_string = input_string.strip()
    input_embedding = model.encode([input_string])[0].astype(np.float32)

    return input_embedding

def genre_search(genre_list, input_embedding, n_genres):
    '''
    This function performs a clustering of the embedded genres from the training data,
    then selects the centroid vector of the cluster compared to the input embedding.
    The selected cluster is unpacked, and the n-most similar genres are selected,
    and the titles of the movies contained within are returned.
    '''
    #extract the embeddings from the genre_list
    genre_embeddings = np.array([item["embedding"] for item in genre_list], dtype=np.float32)

    #convert input and genre embeddings to tensors
    input_tensor = torch.tensor(input_embedding, dtype=torch.float32)
    genre_tensor = torch.tensor(genre_embeddings, dtype=torch.float32)

    #cluster the genre tensor for concise search
    kmeans = util.community_detection(genre_tensor, threshold = 0.75, min_community_size = 10, batch_size=500)

    #in the util clustering, the first item of each list is the centroid vector
    centroids = np.array([genre_tensor[item[0]].numpy() for item in kmeans], dtype=np.float32)
    centroid_tensor = torch.tensor(centroids, dtype=torch.float32)

    #get similarities of input embedding to cluster centroids
    centroids_similarities = util.cos_sim(input_embedding, centroid_tensor)[0]

    #get index of top centroids with respect to 'kmeans'
    top_centroid_idx = torch.argmax(centroids_similarities).item()

    #get embedding indices from selected cluster with respect to 'genre_list'
    selected_cluster_indices = kmeans[top_centroid_idx]

    #get genre embeddings from selected cluster
    selected_embeddings = genre_tensor[selected_cluster_indices]
    selected_embeddings_tensor = selected_embeddings.clone().detach()

    #get genre similarity, and select the n-most similar genres 
    genre_similarity = util.cos_sim(input_tensor, selected_embeddings_tensor)[0]
    top_k_similar = torch.argsort(genre_similarity, descending=True)[:n_genres]

    #extract the titles associated with each genre
    retreived_titles = []
    for index in top_k_similar:
        #select the vector from index
        index = selected_cluster_indices[index.item()]
        vector = genre_list[index]

        #store the movie titles associated with said genre
        retreived_titles.extend(vector["movies"])
        
    return retreived_titles

def overview_search(overview_list, input_embedding, titles, n_movies):
    '''
    This function takes the selected titles, and searches each titles
    overview of the movie with the input embedding and returns the most 
    similar n-movies and their similarity scores.
    '''
    #turn the title list into a set
    titles = set(titles)
    
    # Pre-index overview_list into a dictionary for faster look-up
    overview_dict = {item["title"]: item for item in overview_list}
    search_space = [overview_dict[title] for title in titles if title in overview_dict]
    
    #flip embeddings from list to array for tensor transformation
    overview_embeddings = np.array([item["embedding"] for item in search_space], dtype=np.float32)

    #convert the input and overview embeddings to tensor
    input_tensor = torch.tensor(input_embedding, dtype=torch.float32)
    overview_tensor = torch.tensor(overview_embeddings, dtype=torch.float32)

    #get n-most similar titles by overview similarity
    similarities = util.cos_sim(input_embedding, overview_embeddings)[0]
    sorted_similarities = np.argsort(-similarities)
    top_k_similar = sorted_similarities[:n_movies]

    #extract each of the similar vectors
    selected_vectors = [search_space[idx] for idx in top_k_similar.tolist()]

    #extract the associated similarities
    selected_similarities = [similarities[idx] for idx in top_k_similar.tolist()]

    return selected_vectors, selected_similarities

def display_results(selected_vectors, selected_similarities):
    """Display the selected movies and their similarity to prompt"""
    table = Table(title="Recommended Movies", show_header=True, header_style="bold magenta")
    table.add_column("Title", style="cyan", justify="left")
    table.add_column("Similarity to Prompt", style="green", justify="right")

    for idx, vector in enumerate(selected_vectors):
        table.add_row(vector["title"], f"{selected_similarities[idx]:.1f}")

    console.print(table)

def main():
    '''Driver function for the model'''
    #load in rich console
    global console
    console = Console()
    
    #paths to embeddings
    genre_path = "data/genre_embeddings.json"
    overview_path = "data/overview_embeddings.json"
    
    #check if embeddings exist if not run embedding.py
    if not os.path.exists(genre_path) or not os.path.exists(overview_path):
        console.print(Panel("[bold yellow]Embeddings not found. Running embedding.py...[/bold yellow]"))
        subprocess.run(["python", "embedding.py"])

    console.print(Panel("[bold green]Embeddings are ready![/bold green]"))
    
    #Load the model
    model_name = "all-mpnet-base-v2"
    console.print(Panel("Loading Recommender Model....\n"))
    model = SentenceTransformer(model_name)
    console.print(Panel("Done.\n\n"))
    
    
    while True:

        #load in search data embeddings
        genre_list, overview_list = load_embeddings(genre_path, overview_path)

        #welcome message
        console.print(Panel("[bold cyan]Welcome to the Movie Recommender![/bold cyan]\n"
                            "Type a movie description to find similar movies.", expand=False))

        #prompt user for input
        user_query = Prompt.ask("[bold green]Enter a movie description:[/bold green]")

        #embed input
        input_embedding = embed_input(user_query, model)
        
        #genre search
        titles = genre_search(genre_list, input_embedding, n_genres=5)

        #overview search
        selected_vectors, similarities = overview_search(overview_list, input_embedding, titles, n_movies=5)

        #display
        display_results(selected_vectors, similarities)
        
        continue_search = Prompt.ask("[bold yellow]Would you like to search again? (y/n)[/bold yellow]", choices=["y", "n"])

        if continue_search.lower() != "y":
            console.print("[bold magenta]Thank you for using the recommender! Goodbye![/bold magenta]")
            break

    
    
if __name__ == "__main__":
    main()
    


        