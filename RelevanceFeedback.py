import math
import string
import sys
import time

import nltk
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from googleapiclient.discovery import build
from heapq import heappop, heappush, heapify
import re


nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


'''
Steps:

1. Build the vocabulary with the initial query. The vocabulary maps a word to its "dimension" or "row" in a vector representation.

2. Issue the query and call pare_result_data() to get the title, url, and description for each result

3. Grow the voabulary with the newly found words in the collection of search results in the current iteration via build_vocabulary_df()

4. Main Loop: 

a. iterate through each document retrieved by the current query. Vectorize each document, show the document to the user, 
mark it as relevant or nonrelevant and update the count of relevant and nonrelevant documents, and add the document's vector to the 
appropriate sum (sum of relevant doc vectors or sum of nonrelevant doc vectors).
    
b. Run Rocchio's algorithm via build_next_query_vector() to get the updated query vector based on the relevance feedback:
next_query_vector = previous_query_vector + 0.75(sum_of_relevant_vectors/num_relevant_docs) - 0.15(sum_of_nonrelevant_vectors/num_nonrelevant_docs)

c. Extract up to 2 new words from the updated query vector via get_new_keywords(). Only add two new keywords if the second-highest weighted
keyword is no less than 0.8 times the weight of the first highest weighted keyword to avoid adding unnecessary words to the query.

d. calculate P@K. If above threshold or 0, stop. Else, repeat the main loop.
'''

# search_engine_id = 'b305c2cc7c4272302'
# search_api_key = 'AIzaSyAgJ1HuBv8EeTQ6jRvZVryrwwGFYXKbFfE'


class RelevanceFeedback:

    vocabulary = {}            # maps a word to its dimension in vector representation
    inverted_vocabulary = {}   # maps a dimension to its associated word
    document_frequencies = {}  # document frequency of each word
    used_query_words = set()   # set of all words used in the current query

    def __init__(self, google_api_key, search_engine_id, precision, query):

        self.service = build("customsearch", "v1", developerKey=google_api_key)
        self.search_engine_id = search_engine_id
        self.target_precision = float(precision)
        self.query = query

    def relevanceFeedback(self):

        precision = -1

        # seed the vocabulary with the initial query's words
        self.build_vocabulary_idf(self.query, True)

        # build an initial query vector
        self.query_vector = self.vectorize(self.query)

        iteration = 1

        while self.target_precision > precision != 0:

            print('\nIteration: ', iteration)
            print('Query: ', self.query, '\n')

            # retrieve up to the top 10 search results for the current query and parse the data
            search_results = self.get_result_data(self.query)

            if len(search_results) == 0:
                print('No search results found for given query. ')
                exit(1)

            if iteration == 1 and len(search_results) < 10:
                print('Less than 10 search results retrieved for the first iteration. Exiting program. ')
                exit(1)

            # grow the vocabulary with the newly found words in the retrieved results
            self.build_vocabulary_idf(search_results)

            # initialize empty vectors to track the sum of all relevant and
            # nonrelevant vectors for Rocchio's algorithm
            r_vector_sum = np.zeros((len(self.vocabulary), 1))
            nr_vector_sum = np.zeros((len(self.vocabulary), 1))
            r_count = 0
            nr_count = 0

            i = 1
            for result in search_results:

                # gather the description and title of the document to use for vectorizing the document
                info = result['description'] + ' ' + result['title']
                doc_vector = self.vectorize(info)

                print('Result ', i, ':')
                print('URL: ', result['link'])
                print('Title: ', result['title'])
                print('Description: ', result['description'])

                # keep asking the user for relevance judgement if they enter invalid input (other than y/n)
                while True:

                    user_response = input('Is this document relevant? (Y/N)  ')

                    if user_response.lower() == 'y':
                        r_vector_sum += doc_vector
                        r_count += 1
                        break

                    elif user_response.lower() == 'n':
                        nr_vector_sum += doc_vector
                        nr_count += 1
                        break

                    else:
                        print('Invalid input. Please try again.')

                i += 1
                print('\n\n')

            precision = (r_count / (r_count + nr_count))
            print('Precision for iteration ', iteration, ': ', precision)

            if precision >= self.target_precision or precision <= 0:
                break

            # remake the previous query vector with the new vocabulary built by the current search results so that we can run Rocchio's algo
            updated_previous_query_vector = self.vectorize(self.query_vector)

            # generate next query vector via Rocchio's algorithm
            next_query_vector = self.build_next_query_vector(updated_previous_query_vector, r_vector_sum,
                                                             nr_vector_sum, r_count, nr_count)
            self.query_vector = next_query_vector

            # retrieve the highest weighted words in the updated query vector
            new_keywords = self.get_new_keywords(self.query_vector)

            # update query with new keywords
            self.update_query(new_keywords)

            print('\n\n\n')
            iteration += 1

        # check if we are done, else repeat the loop and issue the updated query
        if precision >= self.target_precision:
            self.success_report(precision)
        else:
            self.failure_report(precision)

        return




    def success_report(self, precision):

        print('\n====================')
        print('FEEDBACK SUMMARY')
        print('Final Query: ', self.query)
        print('Precision: ', precision)
        print('Desired precision reached. Done.')





    def failure_report(self, precision):
        print('\n====================')
        print('FEEDBACK SUMMARY')
        print('Final Query: ', self.query)
        print('Precision: ', precision)
        print('Below desired precision, but can no longer augment query.')





    # after finding new keywords from Rocchio's algorithm, update the search query
    def update_query(self, new_keywords):

        print('Augmenting query by:')

        new_keywords.sort(reverse=True)

        if len(new_keywords) == 0:
            print('No new keywords found. ')
            exit(1)

        max_weight = new_keywords[0][0]

        for weight, new_word in new_keywords:

            # only add two words to the query if the second candidate word is within 80% of the weight of the first (highest weighted)
            # candidate word. this is to avoid adding unimportant keywords to the query

            if weight <= 0.8*max_weight:
                continue

            print(new_word)

            self.used_query_words.add(new_word)

            self.query = self.query + ' ' + new_word






    # issues a given search query and gets up to 10 html results
    # converts results into a list of dictionaries containing data for each document (url, title, summary)
    def get_result_data(self, query):

        results = self.service.cse().list(
            q=query,
            cx=self.search_engine_id,
        ).execute()

        if not 'items' in results:
            print('No results retrieved for given query. ')
            exit(1)

        results = results['items']
        parsed_data = []

        i = 0
        for res in results:

            # if fileFormat is a field in this result, skip it, since this field appears in non html documents
            if 'fileFormat' in res:
                continue

            if i == 10:
                break
            document_data = {
                'link': res.get('link', ''),
                'title': res.get('title', ''),
                'description': res.get('snippet', '')}

            parsed_data.append(document_data)

        return parsed_data





    # converts a string to a bag of words without stop words
    def convert_to_bow(self, text):

        # replace all non-alphanumeric characters with space
        text = re.sub('[^0-9a-zA-Z]+', ' ', text)

        text = word_tokenize(text)
        bag_of_words = []

        for word in text:

            # if this is not a word or number, skip it
            if word in string.punctuation or word.lower() in stop_words or all(p in string.punctuation for p in word):
                continue

            bag_of_words.append(word.lower())

        return bag_of_words





    # accepts list of search results or initial query and updates the vocabulary
    # (the vocabulary is a dict with key=word and value=dimension in the vector representation)
    def build_vocabulary_idf(self, results, query=False):

        self.document_frequencies = {}  # maps a word to its df for the current result collection
        index = len(self.vocabulary)    # keeps track of the dimension to add new words to

        # if this is not a query, we are building vocabulary from search results
        if query == False:
            for res in results:

                info = res['description'] + ' ' + res['title']

                bag_of_words = self.convert_to_bow(info)
                unique_words = set(bag_of_words)

                for word in unique_words:

                    # increment the df of this word
                    self.document_frequencies[word] = self.document_frequencies.get(word, 0) + 1

                    # only create a dimension for this word if it does not already exist in the vocabulary
                    if word not in self.vocabulary:
                        self.vocabulary[word] = index
                        self.inverted_vocabulary[index] = word
                        index += 1

        # building vocabulary from initial query vector
        else:

            bag_of_words = results.lower().split()

            for word in bag_of_words:
                if word not in self.vocabulary:
                    self.vocabulary[word] = index
                    self.inverted_vocabulary[index] = word
                    self.used_query_words.add(word)
                    index += 1
        return






    # accepts a document/query and converts it into a vector
    def vectorize(self, item):

        # initialize empty vector for this document or query
        vector = np.zeros((len(self.vocabulary), 1))

        # converting a query or document to a vector
        if isinstance(item, list) or isinstance(item, str):

            bag_of_words = self.convert_to_bow(item)
            for word in bag_of_words:

                word_index = self.vocabulary[word]
                vector[word_index, 0] += 1

        # rebuilding query vector with new vocabulary built from current search results for current iteration
        else:
            for row in range(len(item)):
                vector[row] = item[row]

        # apply 1 + log(tf) to each row in the vector to "dampen" the effect of term frequency. for example,
        # a doc that has the word "food" in it 10 times more than another doc is not necessarily 10 times more relevant
        vector = self.apply_log_tf(vector)

        return vector





    # given num of Relevant and Nonrelevant results and vector sums,  runs rocchio's algo, and returns next query vector
    # q_i' = q_i + 0.75(r_vector_sum/r_count) - 0.15(nr_vector_sum/nr_count)
    def build_next_query_vector(self, original_query_vector, r_vector_sum, nr_vector_sum, r_count, nr_count):

        next_query_vector = original_query_vector + 0.75*(r_vector_sum/r_count) - 0.15*(nr_vector_sum/nr_count)


        '''
        # used for debugging
        # prints all the extracted words from the search results for the current query and their weights, in sorted order
        words = []

        for row in range(len(next_query_vector)):
            tuple = (next_query_vector[row][0], self.inverted_vocabulary[row])
            words.append(tuple)

        words.sort(reverse=True)
        for weight, word in words:
            print('Word: ', word, "Weight: ", weight)
        '''

        return next_query_vector





    # dampens the weight of each word to prevent high tfs from dominating the document's score
    def apply_log_tf(self, vector):

        for row in range(len(vector)):
            if vector[row] <= 0:
                continue
            vector[row] = 1 + math.log(vector[row], 10)

        return vector





    # given an updated query vector, extracts up to 2 new keywords with the highest weights
    def get_new_keywords(self, next_query_vector):

        # use a min heap to track the top 2 weighted new words from the resulting query vector from Rocchio's algorithm
        heap = []
        heapify(heap)

        for row_num in range(len(next_query_vector)):

            word = self.inverted_vocabulary[row_num]

            # if this keyword is not already in the query
            if word not in self.used_query_words:

                weight = next_query_vector[row_num][0]

                heappush(heap, (weight, word))

                if len(heap) > 2:
                    heappop(heap)

        return heap


def main():

    # REDIRECT TEST OUTPUT to test.txt
    # sys.stdout = open("test.txt", "w")

    if len(sys.argv) != 5:
        print('Invalid number of input arguments.\nFormat: RelevanceFeedback.py '
              '<google api key> <google engine id> <precision> <query>')

    # API Key: AIzaSyAbUyFNJp6VrdunILLcN-OecO0K7_ZH1OU
    # Search engine ID: b305c2cc7c4272302
    # python RelevanceFeedback.py AIzaSyAbUyFNJp6VrdunILLcN-OecO0K7_ZH1OU b305c2cc7c4272302 0.9 "cases"
    # /home/giannis/run AIzaSyAbUyFNJp6VrdunILLcN-OecO0K7_ZH1OU b305c2cc7c4272302 0.9 "cases"
    google_api_key = sys.argv[1]
    search_engine_id = sys.argv[2]
    precision = sys.argv[3]
    query = sys.argv[4]

    time.sleep(1)
    print('Parameters: ')
    print('Client Key: ', google_api_key)
    print('Engine Key: ', search_engine_id)
    print('Query: ', query)
    print('Precision: ', precision)

    client = RelevanceFeedback(google_api_key, search_engine_id, precision, query)
    client.relevanceFeedback()

    #sys.stdout.close()


if __name__ == '__main__':
    main()
