# Relevance-Feedback

Name: Kaan Avci
UNI: koa2107

Name: Brian Yang
UNI: by2289




List of all the files submitted:
1. RelevanceFeedback.py     (contains the code)
2. README.txt               (description of steps, program structure, etc.)
3. transcript.txt           (contains transcript of the 3 test cases)





How to Run Program:

Steps:
1. install all relevant libraries using "pip3 install" command
    pip3 install nltk
    pip3 install numpy
    pip3 install google-api-python-client

2. Run using the following command line format (use python3):

    python3 RelevanceFeedback.py <google api key> <google engine id> <precision> <query>

Examples:
    
    python3 RelevanceFeedback.py AIzaSyAbUyFNJp6VrdunILLcN-OecO0K7_ZH1OU b305c2cc7c4272302 0.9 "per se"
    python3 RelevanceFeedback.py AIzaSyAbUyFNJp6VrdunILLcN-OecO0K7_ZH1OU b305c2cc7c4272302 0.9 "brin"
    python3 RelevanceFeedback.py AIzaSyAbUyFNJp6VrdunILLcN-OecO0K7_ZH1OU b305c2cc7c4272302 0.9 "cases"






Description of Internal Design:

Design Steps:

The program has the following structure:

1. Build the vocabulary with the initial query. The vocabulary maps a word to its "dimension" or "row" in a vector representation.

2. Issue the query and call pare_result_data() to get the title, url, and description for each result

3. Grow the voabulary with the newly found words in the collection of search results in the current iteration via build_vocabulary_df()

4. Main Loop: 

    a. Iterate through each document retrieved by the current query. Vectorize each document, show the document to the  
    user, mark it as relevant or nonrelevant and update the count of relevant and nonrelevant documents, and add the    
    document's vector to the appropriate sum (sum of relevant doc vectors or sum of nonrelevant doc vectors).
    
    b. Run Rocchio's algorithm via build_next_query_vector() to get the updated query vector based on the relevance
    feedback: 

    next_query_vector = previous_query_vector + 0.75(sum_of_relevant_vectors/num_relevant_docs) - 0.15 *
    (sum_of_nonrelevant_vectors/num_nonrelevant_docs)

    c. Extract up to 2 new words from the updated query vector via get_new_keywords(). Only add two new keywords if the
    second-highest weighted keyword is no less than 0.8 times the weight of the first highest weighted keyword to avoid 
    adding unnecessary words to the query.

    d. calculate P@K. If above threshold or 0, stop. Else, repeat the main loop.

In our retrieved results, we tested if the result had a "File Format" field. Those that did have this field were not html, so we simply skipped these results.






External Libraries:
nltk                                - used to download punkt and stopwords
nltk word_tokenize                  - used to convert text into a list of words
nltk.corpus stopwords               - used to eliminate stop words when building vocab
numpy                               - used to work with arrays easily
googleapiclient.discovery's build   - used to connect to Google API







Description of Query-Modification Method:

We implemented Rocchio's algorithm to iteratively update the query vector that most closely 
aligns with the user's search goals. We transform each query and document into a Euclidean space 
by maintaining and building a vocabulary. The vocabulary is a dictionary that maps a word to 
its "dimension" or row in a vector. This allows us to represent documents and queries as a 
vector via numpy arrays to use for Rocchio's algorithm. Initially, the vocabulary only 
contains the initial query words, but is iteratively built with the new unique words 
found in each search result collection.

Each iteration of the algorithm, we initialize two zero vectors with the dimension of the length of the 
vocabulary. One of these zero vectors represents the running sum of all relevant documents 
for the current iteration, and the other vector represents the running sum of all nonrelevant 
documents for the current iteration. As the user provides relevance feedback and marks each result 
as relevant or nonrelevant, we keep track of the number of relevant and nonrelevant documents and
add the vectorized form of the marked document to the appropriate vector sum 
(relevant or nonrelevant document vector sum) that we initalized to 0 at the beginning of the iteration.
After the user provides all relevance feedback for the current iteration, we pass the vector
sum of all relevant documents, the vector sum of all nonrelevant documents, 
the number of relevant documents, and the number of nonrelevant documents to a helper 
function called build_next_query_vector(). This function essentially implenets Rocchio's algorithm 
to determine the next query vector based on the user's relevance feedback:

next_query_vector = 1 * previous_query_vector + 0.75(sum_of_relevant_vectors/num_relevant_docs) - 
0.15(sum_of_nonrelevant_vectors/num_nonrelevant_docs)

We found that providing a coefficient of 1 for previous_query_vector, 0.75 for the average of 
relevant vectors, and 0.15 for the average of nonrelevant vectors yielded optimal results. 
It is important for the average of relevant vectors to be weighted less than the original query 
vector, since we do not want to overly trust the results of this very small sample collection of 10 documents.

Using this next_query_vector, we extract up to 2 new keywords with the highest weight, 
and add those words to the query. To do this, we maintain a set that remembers all the 
keywords we have already used in the current query. Then, we iterate over every row in the 
next_query_vector and tuples of (weight, word) to a min heap of size 2 (if the keyword is not 
already used in the used_query_words set), popping the top element if the size of the heap 
exceeds 2, in order to maintain the 2 highest weighted words. We know the weight of the word 
from the next_query_vector, and we know the associated word for that row from our 
inverted_vocabulary that maps a row to its associated word in Euclidean representation. 
Then, we examine the final 2 words in the heap and their weights. If the weight of the 
second best candidate keyword is less than 80% the weight of the first best candidate keyword, 
we only add the first best candidate keyword. This is to avoid adding unimportant words to the query. 

We then append the new keywords to the query and repeat the process, issuing the query, 
extracting search results, getting relevance feedback, running Rocchio's algorithm, 
obtaining an updated query vector, and extracting new keywords until our precision@k 
reaches the threshold or 0.

Notes:

-the query order is determined each round by appending each newly discovered keyword 
to the end of the query. We experimented with several other strategies, such as putting
the highest weighted keywords earlier in the query, but found that simply appending to 
the end of the query provided the best results.

-to calculate the tf-weight for each term in a document, we used 1 + log(tf) 
instead of raw tf. This provided much better results and dampened the negative 
effect of words that were excessively repeated in some documents.







Your Google Custom Search Engine JSON API Key and Engine ID (so we can test your project)

Google Custom Search Engine JSON API Key: AIzaSyAbUyFNJp6VrdunILLcN-OecO0K7_ZH1OU

Engine ID: b305c2cc7c4272302

