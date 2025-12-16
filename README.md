Seeing Through Different Lenses? A Large-scale Analysis of Critic and Public Film Reviews
Master's Thesis

This repository contains the code used in the data analysis process for the master's thesis "Seeing Through Different Lenses? A Large-scale Analysis of Critic and Public Film Reviews."

It is structured as follows:  

Master_s_Thesis/  
├── ColabNotebooks                        # Contains Notebooks used to execute code on Google Colab  

├── NLP_Analysis                          # Contains Functions used for natural language processing  
│   ├── AggregateValence.py               # Aggregate Valence by Critics and Audiences  
│   ├── AggregateEmbeddings.py            # Aggregate Critic and Audience Embeddings  
│   ├── CalculateEmbeddings.py            # Embeddings Calculation of Film Reviews  
│   ├── NLPAnalysis.py                    # Calls Argument Detection, Aspect Extraction, Emotion Detection or Sentiment Analysis function and handels input/output data  
│   ├── MovieReviewArgumentDetection.py  
│   ├── MovieReviewAspectExtraction.py  
│   ├── MovieReviewEmotionDetection.py  
│   └── MovieReviewSentimentAnalyser.py  

├── NLP_Preprocessing                     # Film Review Preprocessing Functions and actor lists for name-masking  
│   ├── Actor_List.csv                    # Actor List used for Masking of Actor Names  
│   ├── Celebritiy.csv                    # Dataset of 10000 Celebrities used to compile Actor List for masking  
│   ├── IMDb_top_1000_actors.csv          # IMDb's top 1000 Actor List used to compile Actor List for masking  
│   ├── normalize_text.py                 # Function used to normalilze text  
│   ├── prepare_actor_list.py             # Script used to prepare the actor list based on the two Datasets  
│   ├── PreprocessMovieReviews.py         # Calls the function to mask actor names or movie titles and handles input/output data  
│   ├── ReplaceActorNames.py  
│   ├── ReplaceMovieTitles.py  
