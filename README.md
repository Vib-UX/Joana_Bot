# Joana_Bot
A basic chatbot from scratch using ML Algorithms and DL

Code for retrieval-based chatbot.
For preprocessing:
1.  It removes all the stopwords in each sentence (i.e., the words unnecessary in predicting text).
2.  Along with stopword removal, we also create training sentences out of lemmatized words.
3.  All the training labels corresponding to each training sentences are label encoded. 
4.  The words are then tokenized and further assigned a unique index which gives us our word index
5.  Our training sentences are converted into sequence of numbers with the help of the reference of words i.e., word index
6.  The training sentences now, sequences are padded with zeroes.
