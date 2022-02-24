# AR-BERT: Aspect-relation enhanced Aspect-level Sentiment Classification with Multi-modal Explanations
Codebase for our WWW'22 paper: "AR-BERT: Aspect-relation enhanced Aspect-level Sentiment Classification with Multi-modal Explanations"

## Requirement:
* item Python 3.6.8
* item Tensorflow-gpu 1.15.0 (See suitable tensorflow 1.x GPU version based on your system CUDA: https://www.tensorflow.org/install/source#gpu)
* item Spacy 2.3.2
* item Snappy 1.1.7 (http://snap.stanford.edu/snappy/index.html)

## Infrastructure:
* item Quadro P5000 Single core GPU (16278MiB)
* item CUDA version: 10.0

## Usage:
Please follow the follwing steps sequentially (as mentioned in teh paper also)

## 1. Aspect to entity mapping: 
Go to folder: /src/aspect_to_entity_mapping/
  ### 1.1. To prepare raw data for wikification:
  ```
  python data_preprocessing_for_wikification_input.py
  ```
  ### 1.2 To extract entities (wikification):
  ```
  python wikification_of_input_data.py wikifier_input_data_directory dataset_name(laptop/rest/twitter) mode(train/test) wikifier_output_directory
  ```
  
## 2. Two level aspect entity embediing generation:
Go to folder: /src/two_level_aspect_entity_embedding_generation/
  ### 2.1 Clustering of DBPedia Page-link Knowledge Graph:
  Go to folder: ./kg_clustering/
  #### 2.1.1 For preparing numeric edge list (src, dest) and dictionary entity label --> id and vice-versa from KG:
  ```
  python kg_pre_processing_create_edge_tuple_list.py path of the DBPedia Page-link English KG
  ```
  #### 2.1.2 For creating SNAP undirected graph and retrieving maximum connected componenet required for next step clustering:
  ```
  python dbpedia_pagelink_edge_tuple_list_to_snap_graph.py
  ```
  #### 2.1.3 For preparing numeric edge list (src, dest) and dictionary entity label --> id and vice-versa from maximum connected componenet of KG:
  ```
  python connected_kg_pre_processing_create_edge_tuple_list.py
  ```
  

