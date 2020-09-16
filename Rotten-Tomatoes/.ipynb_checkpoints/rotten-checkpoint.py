import bs4 as bs
from urllib.request import Request, urlopen
import pandas as pd
from string import ascii_letters
import numpy as np
import os


newfolder = 'rottentomatoes'

if not os.path.isdir(newfolder):
    os.mkdir(newfolder)

os.chdir(newfolder)

website = "https://www.rottentomatoes.com"
alphabet = ascii_letters[:26].replace('x', '')  # remove x


def fetch(page, addition=''):
    """Fetches HTML data"""
    headers = {'User-Agent': 'Opera/9.80 (X11; Linux i686; Ub'
               'untu/14.10) Presto/2.12.388 Version/12.16'}
    req = Request(page + addition, headers=headers)
    open_request = urlopen(req).read()
    soup = bs.BeautifulSoup(open_request, 'lxml')
    return soup


def critics_letters(letters):
    """Creates URL for 26 pages of critics, based on the first letter of their name"""
    letters_url = list()
    for elem in letters:
        letters_url.append("/critics/authors?letter=" + elem)
    return letters_url


def critics_list(catalog):
    """Fetches the url of all listed critics"""
    critics_url = list()
    for ix, letter_pages in enumerate(catalog, 1):
        for a in fetch(website, letter_pages).find_all("a", {"class": "a critic-authors__name"}):
            href_critic = a['href']
            if str(href_critic)[:7] != "/source":
                critics_url.append(href_critic + "/movies")
    print('1/4 — of movie critic URLs scraped.'.format(ix/len(catalog)), end='   ')
    print('\r{} pages of movie critic URLs successfully scraped.'.format(ix), end='  '); print()
    return critics_url


def movies(catalog):
    """Fetches the url of the movies reviewed by the critic"""
    movies_url = list()
    errors = 0
    for ix, critic_profile in enumerate(catalog, 1):
        try:
            checker = fetch(website, critic_profile).find_all("h2", {"class": "panel-heading js-review-type"})
            if len(checker) > 0:
                if checker[0].text == "Movie Reviews Only":
                    for td in fetch(website, critic_profile).find_all("td",
                                    {"class": "col-xs-12 col-sm-6 critic-review-table__title-column"}):
                        for a in td.find_all("a"):
                            if a['href'] not in movies_url:
                                movies_url.append(a['href'])
        except:
            errors += 1
    # print('\r2/4 — {:.2%} of movie URLs scraped. Error rate: {:.2%}'.format(ix/len(catalog),
    #                                     errors/ix), end='   ')
    # print('\r{} movie URLs successfully scraped. Error rate: {:.2%}'.format(len(movies_url)-errors, errors/ix), end='\n')
    return movies_url


def review_pages(catalog):
    """List the pages of reviews from all movies in chunks of 1000 and exports 17 csv files"""
    review_pages_list = list()
    errors = 0
    for ix, movie in enumerate(catalog.iloc[:, 0], 1):
        try:
            soup_2 = fetch(movie, "/reviews/?page=1").find_all("span", {"class", "pageInfo"})
            if len(soup_2) >= 1:
                for n in range(1, int(soup_2[0].text[-2:]) + 1):
                    review_pages_list.append(movie + "/reviews/?page=" + str(n))
        except:
            errors += 1
    print('\r3/4 — {:.2%} of review page URLs scraped. Error rate: {:.2%}'.format(
            ix/len(catalog), errors/ix), end='    ')
    print('\r{} review page URLs successfully scraped. Error rate: {:.2%}'.format(
        len(review_pages_list)-errors, errors/ix), end='\n')
    return review_pages_list


def rating_review(catalog):
    """Scrapes all the reviews and rating from the pages"""
    reviews = list()
    errors = 0
    for ix, page in enumerate(catalog.iloc[:, 0], 1):
        try:
            soup_2 = fetch(page, "").find_all("div", {"class": "col-xs-16 review_container"})
            for comment in soup_2:
                comment_text = comment.find_all("div", {"class": "the_review"})[0].text.strip()
                icon = str(comment.find_all("div")[0])
                if "fresh" in icon:
                    reviews.append('1 - ' + comment_text)
                elif "rotten" in icon:
                    reviews.append('0 - ' + comment_text)
        except:
            errors += 1
    print('\r4/4 — {:.2%} of reviews scraped. Error rate: {:.2%}'.format(ix/len(catalog),
                                            errors/ix), end='    ')
    print('\r{} reviews successfully scraped. Error rate: {:.2%}'.format(
        len(reviews)-errors, errors/ix), end='\n')
    return reviews


def process(document):
    """Prepares the document and exports it to the working directory"""
    df = document
    df['freshness'] = df.iloc[:, 0].str.split(' - ').str.get(0)
    df['review'] = df.iloc[:, 0].str.split(' - ').str.get(1)
    df = df.loc[df['review'].str.len() >= 18]
    df = df.loc[:, ['freshness', 'review']]

    num_to_keep = (df.shape[0] - df.freshness.astype(np.int32).sum()) // 10_000 * 10_000
    rotten = df.loc[df.freshness == '0'].sample(num_to_keep)
    fresh = df.loc[df.freshness == '1'].sample(num_to_keep)

    df = pd.concat([rotten, fresh], axis=0, sort=False)
    df = df.sample(frac=1).reset_index(drop=True)

    df.to_csv('all_rotten_tomatoes_reviews.csv', index=False)

    print('\nThe web scraper has finished.',
          '\nCheck your directory: {}'.format(os.getcwd()),
          '\nThe file with all reviews is named: all_rotten_tomatoes_reviews.csv')
    return df


if __name__ == '__main__':
    critic_main = critics_letters(alphabet)
    list_critics = critics_list(critic_main)
    all_movies = movies(list_critics)

    df = pd.DataFrame(all_movies)
    df.to_csv(r'movies.csv', header=False, index=None)
    all_movies = pd.read_csv('movies.csv', header=None)

    review_pages = review_pages(all_movies)
    pd.DataFrame(review_pages).to_csv('all_pages.csv', index=False, header=None)
    review_pages = pd.read_csv('all_pages.csv', header=None)

    rating_reviews = rating_review(review_pages)
    pd.DataFrame(rating_reviews).to_csv('reviews.csv', index=False, header=None)
    df = pd.read_csv('reviews.csv', header=None)

    final_doc = process(df)