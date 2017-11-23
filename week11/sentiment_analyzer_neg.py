from nltk.corpus import movie_reviews 
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
 
# Extract features from the input list of words
def extract_features(words):
    return dict([(word, True) for word in words])
 
if __name__=='__main__':
    # Load the reviews from the corpus 
    fileids_pos = movie_reviews.fileids('pos')
    fileids_neg = movie_reviews.fileids('neg')
     
    # Extract the features from the reviews
    features_pos = [(extract_features(movie_reviews.words(
            fileids=[f])), 'Positive') for f in fileids_pos]
    features_neg = [(extract_features(movie_reviews.words(
            fileids=[f])), 'Negative') for f in fileids_neg]
     
    # Define the train and test split (80% and 20%)
    threshold = 0.8
    num_pos = int(threshold * len(features_pos))
    num_neg = int(threshold * len(features_neg))
     
     # Create training and training datasets
    features_train = features_pos[:num_pos] + features_neg[:num_neg]
    features_test = features_pos[num_pos:] + features_neg[num_neg:]  

    # Print the number of datapoints used
    print('Number of training datapoints:', len(features_train))
    print('Number of test datapoints:', len(features_test))
     
    # Train a Naive Bayes classifier 
    classifier = NaiveBayesClassifier.train(features_train)
    print('Accuracy of the classifier:', nltk_accuracy(
            classifier, features_test))

    N = 15
    print('Top ' + str(N) + ' most informative words:')
    for i, item in enumerate(classifier.most_informative_features()):
        print(str(i+1) + '. ' + item[0])
        if i == N - 1:
            break

    # Test input movie reviews
    input_reviews = [

        "There is no meaningless dialog; not a single extraneous character.",
        "I don't see any point to the vast majority of this film.",
        "there is no justice in the world and that's acceptable for humankind.",
        "Not 'terrifying', The main problem is that I don't see any message in this movie, nor any pleasant feature. It is not funny, it is not horrifying or realistic enough, there is no real character development, no interesting statements about anything. It is an empty, overrated mess. It is basically a shock-parade with old instrumental background music.",
        "nothing really interesting or compelling ever happens in this.",
        "It is interesting",
        "It is not interesting",
        "It is terrifying",
        "It is not terrifying",
        "It is meaningful",
        "It is no meaningful",
        "It is awful",
        "It is not awful",
        "It has meaning",
        "It has no meaning",
        "A Clockwork Orange is the finest film",
        "A Clockwork Orange is not the finest film",
    
    
    ]

    print("Movie review predictions:")
    for review in input_reviews:
        print("Review:", review)

        # Compute the probabilities
        probabilities = classifier.prob_classify(extract_features(review.split()))

        # Pick the maximum value
        predicted_sentiment = probabilities.max()

        # Print outputs
        print("Predicted sentiment:", predicted_sentiment)
        print("Probability:", round(probabilities.prob(predicted_sentiment), 2))
