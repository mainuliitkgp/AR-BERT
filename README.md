# AR-BERT: Aspect-relation enhanced Aspect-level Sentiment Classification with Multi-modal Explanations
Codebase for our WWW'22 paper: "AR-BERT: Aspect-relation enhanced Aspect-level Sentiment Classification with Multi-modal Explanations"

## Requirement:
* Python 3.6.8
* Tensorflow-gpu 1.15.0 (See suitable tensorflow 1.x GPU version based on your system CUDA: https://www.tensorflow.org/install/source#gpu)
* Spacy 2.3.2
* Snappy 1.1.7 (http://snap.stanford.edu/snappy/index.html)

## Infrastructure:
* Quadro P5000 Single core GPU (16278MiB)
* CUDA version: 10.0

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
  #### 2.1.4 Clustering of KG using Louvain algorithm:
  Go to folder: ./kg_clustering/louvain_clustering/
  See readme there to perform clustering step-by-step. 
  
  ### 2.2. Create node id to cluster id and cluster id to list of node ids dictionaries after clustering:
  ```
  python create_lovain_dictionary_cluster_id_to_node_list.py
  ```
  
  ### 2.3 Prepare cluster weighted graph and check connectivity: 
  ```
  python build_community_graph_and_check_connectivity.py
  python clusterd_knowledge_graph_statistics.py (to get the graph statistics)
  ```
  
  ### 2.4 Prepare aspect entity subgraph 
  ```
  create_entity_graph_for_graphsage.py
  ```
  
  ### 2.5 For cluster graph embedding generation:
  Go to folder: ./weighted_graphsage/
  ```
  python -m weighted_graphsage.unsupervised_train
  ```
  
  ### 2.5 For aspect entity sub-graph embedding generation:
  Go to folder: ./graphsage/
  ```
  python -m graphsage.unsupervised_train --train_prefix path of the entity subgraph --epochs 100 --max_degree maximum node degree of the entity sungraph (varies wrt different dataset)
  ```
  
  
## 3. ALSC:
Go to folder /src/alsc/

  ### 3.1. AR-TNET:
  ```
  cd ./ar-tnet/
  python main_no_split_lstm_after_cpt_first_and_last_hidden_reshape_concat_GraphSAGE.py \
  -ds_name 14semeval_rest \
  -connection_type AS \
  ```
  Change the parameter values based on the dataset name [14semeval_laptop/14semeval_rest/twitter] and connection type [AS/LF]
  
  ### 3.2. AR-BERT:
  ```
  cd ./ar-bert/bert_tensorflow/
  python run_classifier_alsc.py \
  --data_dir path of the raw data file (.xml) \
  --task_name SemEval2014AtscProcessor \
  --vocab_file path of the bert-base-uncased vocab file\
  --bert_config_file path of the bert-base-uncased config file\
  --output_dir path of the output directory
  ```
  
  ### 3.3 AR-BERT-S:
  ```
  cd ./ar-bert-s/
  python run_bert.py
  ```
  
  ### 3.4 Incorrect disambiguation detection:
  ```
  cd ./incorrect_disambiguation_detection/
  
  python sampling_entity_embedding_from_graph.py \
  unique_entity_list_file_path \
  train_raw_data_file_path \
  test_raw_data_file_path \
  dataset_name \
  10
  
  python prepare_bert_embedding_for_nodes_i_j_k.py \
  raw_sentence_file_path raw_aspect_file_path \
  raw_character_start_and_end_index_of-aspect_in_sentence_file_path \
  bert-base-uncased_vocab_file_path \
  bert_token_embedding_file_path \
  entity_sampling_file_path \
  entity_index_to_list_of_sent_index_file_path \
  output_dir_path \
  dataset_name 
  
  python semantic_probe.py \
  --train_file_path path of the training data \
  --output_dir path of the output directory \
  --ds_name laptop/rest/twitter \
  
  
  
  

