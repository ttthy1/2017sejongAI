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

        "This is a great, awesome, funny, beautiful,perfect movie.",
        "This is a great, awesome, funny movie. This is a beautiful, perfect movie.",
        "This is a great, awesome movie. This is a funny movie. This is a beautiful, perfect movie.",
        "This is a great, awesome movie. This is a funny movie. This is a beautiful movie. This is a perfect movie.",
        "This is a great movie. This is a awesome movie. This is a funny movie. This is a beautiful movie. This is a perfect movie.",
        "People say that this film is masterpiece.", 
        "A Clockwork Orange is a one of a kind masterpiece.",
        "People say that this film is masterpiece and A Clockwork Orange is a one of a kind masterpiece.",
        "People say that this film is masterpiece. A Clockwork Orange is a one of a kind masterpiece."
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
