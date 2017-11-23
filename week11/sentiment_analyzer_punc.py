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

        "This is just possibly the most perfect movie ever made There is no meaningless dialog not a single extraneous character",
        "People say that this film is somehow amazing To me it is far more about image and very light on any content I am all for the freedom to depict unpleasant acts where appropriate and where it has some purpose to it but I do not see any point to the vast majority of this film The whole thing seems to be designed to appeal to base instincts through violence and sex without ever making any valid or thought provoking points about either these or any other issue",
        "Bad Story Bad Behavior Bad Scenes Usually before watching any movie I look up IMDb to see its rating and viewers comments on it I did the same before seeing the Clockwork Orange It said mostly yes violence yes the main heros a monster but what a masterpiece It seemed to deserve its place in the first 50 in the IMDb rating Then I watched the film And I believe it is one of the most disgusting films I ever saw It is no doubt intended to be full of hidden significance the grotesque manner in which characters speaks move dress live. This is supposed to be new and frighteningly surrealistic a sharp futuristic social satire",
        "This is the most god awful disgusting movie I have ever seen The fact that so many people think its brilliant or amazing is very insightful and informative for me It tells me that there are a lot more sick and twisted people out there and it makes me doubly glad that I am not afraid to look out for me and my own The only message this story has is that it is great to be a criminal there is no justice in the world and that's acceptable for humankind",
        "This movie is terrible Not terrifying there are a lot of excellent horror and war movies for that Rather terribly empty The main problem is that I do not see any message in this movie nor any pleasant feature It is not funny it is not horrifying or realistic enough there is no real character development no interesting statements about anything It is an empty overrated mess It is basically a shock-parade with old instrumental background music",
        "This movie sucks From the clothes to the retarded mix of English and Russian to the theatrical way the violent scenes are shot The theatrical shooting of the violence makes it impossible to take it seriously I have seen more graphic violence in modern TV series than in this over rated crap If you feel like you HAVE to watch this movie because it is considered a classic do not put yourself through the torture It is classic BS and it should be avoided completely",
        "This was a very unsatisfying movie It starts out with promise and spectacularly fails to deliver . Where to start? The basic premise of a chap going off to live alone in the wild with adventures all along the way is a good one. But nothing really interesting or compelling ever happens in this",
        "Uninteresting and very boring story about a selfish arrogant man blaming all his problems on his parents who runs away from his family to live in the wilderness for 4 months before he finds out that happiness is found in the company of other people and family",
        "I really hated it The movie had a really moralizing tone and it was incredibly self conscious All the time it felt like the guy was posing for the camera"
        
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
