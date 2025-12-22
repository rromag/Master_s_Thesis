Seeing Through Different Lenses? A Large-scale Analysis of Critic and Public Film Reviews
Master's Thesis

This repository contains the code used in the data analysis process for the master's thesis "Seeing Through Different Lenses? A Large-scale Analysis of Critic and Public Film Reviews."

The functions for this thesis are written to work with the structure below and handle in- and output paths accordingly.
Jupyter Notebooks will have to be moved from the Jupyter_Notebooks directory to the Workspace level to find the necessary file paths. 

Scripts that do not contain functions that can be imported are marked with (!). The outputs of these Scripts are saved within this structure (Review Data and Actor_List) and do not need to be run.

The function NLPAnalysis calls the functions for Argument Detection, Aspect Extraction, Emotion Detection and Sentiment Analysis, while controlling in- and output paths.

Similarly, the function PreprocessMovieReviews can be called to either mask actor names, or movie titles and controls in- and outputs.



Code/Workspace Structure:  

Master_s_Thesis/  
├── ColabNotebooks                                              # Contains Notebooks used to execute code on Google Colab  
│   ├── BERTopicModelTuning.ipynb                               # Used to Inspect and Merge the Topics of the Trained BERTopic Model (using BERTopicLoadModel.py to load the model, embeddings, training documents and topics) 
│   ├── BERTopicRunInference.ipynb                              # Used to Run Inference on the Extracted Movie Review Aspects (using BERTopicLoadModel.py to load the model)  
│   ├── RunAggregateEmbeddings.ipynb                            # Used to Aggregate Critic and Audience Review Embeddings Separately (using AggregateEmbeddings.py)  
│   ├── RunAspectExtraction.ipynb                               # Used to Run Aspect Extraction (using NLPAnalysis.py)  
│   ├── RunCalculateEmbeddings.ipynb                            # Used to Calculate Embeddings for all Movie Reviews (using CalculateEmbeddings.py)  
│   ├── RunSentimentAnalysis.ipynb                              # Used to Run Sentiment Analysis (using NLPAnalysis.py)  
│   └── TrainBERTopic.ipynb                                     # Used to train the Topic Model (using BERTopicTraining.py)  
  
├── Jupyter_Notebooks                                           # Contains Jupyter Notebooks for Data Cleaning, Summary Statistics and Research Question-specific analyses  
│   ├── A_RT_Data_Cleaning.ipynb                                # Initial Data Cleaning pre-translation  
│   ├── B_RT_audience_reviews_SummaryStats.ipynb                # Post-translation cleaning and summary statistics of clean audience review data  
│   ├── B_RT_critic_reviews_SummaryStats.ipynb                  # Post-translation cleaning and summary statistics of clean critic review data  
│   ├── B_RT_movies_SummaryStats.ipynb                          # Movie-level summary statistics and final comparison of audience and critic review dataset congruency  
│   ├── RQ1_Topic_Analysis.ipynb                                # Analysis for RQ1  
│   ├── RQ2_Valence_Disagreement_&_Content_Divergence.ipynb     # Analysis for RQ2  
│   ├── RQ3_A_Ratings.ipynb                                     # Analysis for RQ3 and RQ3a, Groups defined by Rating Difference and Content Similarity 
│   ├── RQ3_A_Sentiment.ipynb                                   # Analysis for RQ3 and RQ3a, Groups defined by Difference in textual sentiment and Content Similarity  
│   ├── RQ3_A.1_DivergingValenceSubGroups_Rating.ipynb          # Deepdive Public-Facing Valence Measure Distributions for Groups with diverging valence (using rating as valence)  
│   ├── RQ3_A.1_DivergingValenceSubGroups_Sentiment.ipynb       # Deepdive Public-Facing Valence Measure Distributions for Groups with diverging valence (using sentiment as valence)  
│   └── RQ3_B_SimilarValenceGroups.ipynb                        # Analysis for RQ3b and RQ3c, Groups defined by Valence-Level (Sentiment) and Content Similarity  
  
├── NLP_Analysis                                                # Contains Functions used for natural language processing  
│   ├── AggregateEmbeddings.py                                  # Aggregate Critic and Audience Embeddings  
│   ├── AggregateValence.py                                     # Aggregate Valence by Critics and Audiences  
│   ├── CalculateEmbeddings.py                                  # Embeddings Calculation of Film Reviews  
│   └── NLPAnalysis.py                                          # Calls Functions to Run Argument Detection, Aspect Extraction, Emotion Detection or Sentiment Analysis and handels input/output data  
│       ├── MovieReviewArgumentDetection.py                             # Subfunction  
│       ├── MovieReviewAspectExtraction.py                              # Subfunction    
│       ├── MovieReviewEmotionDetection.py                              # Subfunction   
│       └── MovieReviewSentimentAnalyser.py                             # Subfunction  
  
├── NLP_Preprocessing                                           # Film Review Preprocessing Functions and actor lists for name-masking  
│   ├── Actor_List.csv                                          # Actor List used for Masking of Actor Names  
│   ├── Celebritiy.csv                                          # Dataset of 10000 Celebrities used to compile Actor List for masking  
│   ├── IMDb_top_1000_actors.csv                                # IMDb's top 1000 Actor List used to compile Actor List for masking  
│   ├── normalize_text.py                                       # Function used to normalilze text  
│   ├── prepare_actor_list.py                       (!)         # Script (!) used to prepare the actor list based on the two Datasets  
│   └── PreprocessMovieReviews.py                               # Calls Function to mask actor names or movie titles and handles input/output data  
│       ├── ReplaceActorNames.py                                        # Subfunction  
│       └── ReplaceMovieTitles.py                                       # Subfunction  
  
├── Rotten Tomatoes Reviews                                     # Contains translated and cleaned audience and critic review dataset, as well as movie-level data  
│   ├── Audience Reviews Clean                                  # 25 Json files containing audience review data  
│   ├── Critic Reviews Clean                                    # 20 Json files containing critic review data  
│   └── rt_movies_clean.Json                                    # Json file containing cleaned movie-level data  
   
├── TopicModelling                                              # Contains Functions for Topic Modelling and Topic Aggregation    
│   ├── AggregateTopics.py                                      # Aggregate Topic Data for critics and audiences separately  
│   ├── BERTopicInference.py                                    # Run Inference (inputs and outputs handled automatically)  
│   ├── BERTopicLoadModel.py                                    # Load a previously trained model (inputs and outputs handled automatically).  
│   └── BERTopicTraining.py                                     # Train model (inputs and outputs handled automatically)  
   
├── Translation                                                 # Contains Functions for Language Detection and Translation (The provided Dataset is already translated and cleaned, therefore should not be required).  
│   ├── TranslateMovieReview.py                                 # Loads reviews, calls Language Detection to identify non-English reviews, and translates those into English.  
│       ├── DetectLanguage.py                                           # Subfunction  
│       ├── MovieReviewTranslatorDeepl.py                               # Subfunction    (Do not use, unless someone else is footing the bill :) )  
│       └── MovieReviewTranslatorGoogle.py                              # Subfunction  
   
├── Webscraping_RT_BoxOffice                                    # Contains Function to Scrape Box Office Revenues from Rotten Tomatoes and a Runner Script  
│   ├── Scraping Box Office off of RT.py            (!)         # Runner Script (!) to scrape Box office figures from Rotten Tomatoes  
│       ├── prepare_movie_list_for_scraping.py                  # Function to prepare a list of movies to scrape for (input file will not be available)  
│       └── ScrapeBoxOfficeRT.py                                # Function doing the scraping, called by the runner script  
  
├── Webscraping_RT_XHR                                          # Contains Code needed to Scrape Film Reviews from Rotten Tomatoes  
│   ├── scrape_emsId.py                             (!)         # Script (!) to scrape emsIds (Rotten Tomatoes internal movie identifier)  
│   └── XHR_BatchScrapingRT.py                                  # Function to handle batching of review scraping and saving of outputs, uses the following functions for parallelized web scraping  
│       ├── XHR_ParalleliseScraping.py                                  # Subfunction    (Called by previous function to parallelize the scraping process)  
│       └── XHR_RTScraper.py                                            # Subfunction    (The function doing the web scraping)  
