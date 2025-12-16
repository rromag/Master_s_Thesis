import asyncio
import time
import pandas as pd
from playwright.async_api import async_playwright
import random
import re
from Webscraping_RT_BoxOffice.prepare_movie_list_for_scraping import prepare_movie_list_for_scraping
import os
import numpy as np


MAX_RETRIES = 3
RETRY_DELAY = 2

headers_mod = [{
                                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
                                    'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8', 'Referer': 'https://google.com'},
                                {
                                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36',
                                    'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8', 'Referer': 'https://google.com'},
                                {
                                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15',
                                    'Accept-Language': ' en-GB,en;q=0.9', 'Referer': 'https://google.com'},
                                {
                                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:139.0) Gecko/20100101 Firefox/139.0',
                                    'Accept-Language': 'en-US,en;q=0.5', 'Referer': 'https://google.com'},
                                {
                                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:128.0) Gecko/20100101 Firefox/128.0',
                                    'Accept-Language': 'en-US,en;q=0.5', 'Referer': 'https://google.com'},
                                {
                                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
                                    'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8', 'Referer': 'https://google.com'},
                                {
                                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36',
                                    'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8', 'Referer': 'https://google.com'}]

sem = asyncio.Semaphore(9)  # Limit concurrency


async def get_movie_id_async(slug, context, sem):
    async with sem:
        page = None
        try:
            page = await context.new_page()
            await page.set_extra_http_headers(random.choice(headers_mod))
            await page.goto(f"https://www.rottentomatoes.com/m/{slug}", timeout=10000)
            await page.wait_for_timeout(3000)

            html = await page.content()
            match = re.search(r'"emsId"\s*:\s*"([a-f0-9\-]{36})"', html)
            return {"slug": slug, "emsId": match.group(1) if match else None}
        except Exception as e:
            return {"slug": slug, "emsId": None, "error": str(e)}
        finally:
            if page:
                await page.close()

async def main(slugs):
    sem = asyncio.Semaphore(9)
    user_data_dir = "/tmp/playwright"
    os.makedirs(user_data_dir, exist_ok=True)

    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir,
            headless=True,
            args=["--no-sandbox"]
        )

        try:
            tasks = [get_movie_id_async(slug, context, sem) for slug in slugs]
            results = await asyncio.gather(*tasks)
            return results
        finally:
            await context.close()



if __name__ == "__main__":
    import json
    movies = prepare_movie_list_for_scraping(movie_review_data="Rotten Tomatoes reviews/critic_reviews_clean_for_emsId.csv", min_critic_reviews=20)
    print(len(movies))
    no_batches = int(round(len(movies)/60, ndigits=0))
    print(no_batches)

    split_list = [*np.array_split(np.array(movies), no_batches)]

    for idx, batch in enumerate(split_list):
        print("Start Batch", idx)
        start_batch = time.time()

        results = asyncio.run(main(batch))

        end_batch = time.time()
        print("End Batch", idx)
        print("Batch Runtime:", end_batch-start_batch)

    # Save to file
        with open(f"Rotten Tomatoes reviews/ems_id_new/movie_ems_ids_{idx}.json", "w") as f:
            json.dump(results, f, indent=2)

    print("Done")