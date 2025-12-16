import pandas as pd
import numpy as np
from Webscraping_RT_BoxOffice.prepare_movie_list_for_scraping import prepare_movie_list_for_scraping
from Webscraping_RT_BoxOffice.ScrapeBoxOfficeRT import ScrapeBoxOffice
import multiprocessing
import time



movie_list = prepare_movie_list_for_scraping(movie_review_data="Rotten Tomatoes reviews/critic_reviews_clean_all_rating_scales.csv", min_critic_reviews=10)

movie_list = movie_list[8350:]
print(len(movie_list))


start_time = time.time()

split_movie_list = [*np.array_split(movie_list, 9)]

for i in range(0, 9):
    print(len(split_movie_list[i]))

pool = multiprocessing.Pool(9)

result = pool.map(ScrapeBoxOffice, split_movie_list)

box_office_data = pd.concat(result).reset_index(drop=True)

end_time = time.time()
print("Runtime Parallelization:", end_time-start_time)

print(box_office_data)

#start_time = time.time()

#box_office_data = ScrapeBoxOffice(movie_list=movie_list)

#end_time = time.time()
#print("Runtime normal:", end_time-start_time)