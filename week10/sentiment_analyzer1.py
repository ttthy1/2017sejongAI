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
        "It's hard to judge a film such as this. Its cold and hard, yet can be exhilarating and sarcastic. It can be average. yet it can be visionary. Exploitive? Satirical? Too many questions to consider when one watches this film.", 
        "A Clockwork Orange is the finest film that has ever been made, in my view. Stanley Kubrick has made so many masterpieces, and is by far the best director that ever graced our world. A Clockwork Orange is simply his finest hour!",
        "A Clockwork Orange is by no means an easy film to get through, as many will be turned off by the scenes of violence and rape. But this masterpiece is far more complex than a simple romp through a world of youthful violence. It's a rare example of film-making that demands that the viewer actually think.", 
        "This is just possibly the most 'perfect' movie ever made. There is no meaningless dialog; not a single extraneous character.",
        "Without a doubt, my absolute favorite film of all time. I first saw this movie three years ago and I have been in love with it ever since. I never get tired of seeing this movie.",
        "What a masterpiece!! He loves to hold a mirror up to our monstrous faces beneath our masks and laugh at our vanity. Stanley delighted in having fun with our hubris about ourselves. Yes, little Alex has all violence removed from him and he is set free in that idyllic paradise we kid ourselves is our society.",
        "Kubrick's's best film ever.absolutely mind-blowing.quite disturbing though but that's what moves us.he has amazingly blended the disturbing scenes with the lovely music of Beethoven",
        "Clockwork Orange is definitely the most bizarre film I've ever seen. The whole idea of brainwashing a criminal to never do harm again in itself is genius. Stanley Kubrick amazes me with just the music he picked for the movie alone. I was hooked from the opening scene!",
        "A Clockwork Orange is a one of a kind masterpiece. It's odd, intelligent, funny and sickening. Kubrick at his best! The opening sequence in the milk bar might very well be the best and most well known opening in cinema history. The movie manages to stay strong throughout the whole movie after the opening and only gets better and never weakens for a bit. Every minute is a pleasure to watch because the movie never gets predictable in anyway and the oddness level of the movie makes it so that it will be always a surprise what will happen next and how it will happen.",
        "A Clockwork Orange is my favourite film of all time, and deservedly so; I've watched it 10 times, and it never fails to disappoint. Whether you love it or you hate it, you will never forget it. It's a disturbing and dark film, but if you can stomach it, you'll almost certainly like it.",
        "People say that this film is somehow amazing. To me it is far more about image and very light on any content. I am all for the freedom to depict unpleasant acts where appropriate and where it has some purpose to it, but I don't see any point to the vast majority of this film. The whole thing seems to be designed to appeal to base instincts through violence and sex without ever making any valid or thought provoking points about either these or any other issue.",
        "Bad Story, Bad Behavior, Bad Scenes ... Usually, before watching any movie I look up IMDb to see its rating and viewers' comments on it. I did the same before seeing the Clockwork Orange. It said, mostly, yes, violence, yes, the main hero's a monster, but what a masterpiece! It seemed to deserve its place in the first 50 in the IMDb rating... Then I watched the film. And I believe it is one of the most disgusting films I ever saw. It is no doubt intended to be full of hidden significance, the grotesque manner in which characters speaks, move, dress, live. This is supposed to be 'new and frighteningly surrealistic', a 'sharp, futuristic social satire'.",
        "Yes a give it a flat out 1. 1 has (awful) next to it and that is what this movie is in every way. Very simply put, this movie is sick, twisted, hateful, disgusting, and people love it. What a shame.",
        "If I could give this movie zero stars, I would, no let me take that back, if I could give this movie negative stars I would. ",
        "What I really hated was how the violence towards women was presented. The rape scenes are quite graphic with out conveying the gravity of situation.",
        "Many people like this movie, maybe they do have their reasons .. well, I like Kubrick and he is one of the greatest directors but this movie is the black sheep in his career. It is too violent and too disgusting; no wonder it was banned in the USA and other countries. so, save your time and money .. do yourself a favor and skip it. Mark my words: you wouldn't miss much.",
        "A Clockwork Orange was done in really poor taste. I was looking forward to this, everyone was raving about it. The worst thing about it is that you remember certain bits of awfulness for ages and ages afterwards. The movie has nothing (I really mean it). Every thing about the movie is slow. It simply suck. One of the worst pointless movies I have ever seen. I fail to see why this movie seems to be regarded as fresh and groundbreaking.",
        "This is the most god awful, disgusting movie I have ever seen. The fact that so many people think its brilliant or amazing is very insightful and informative for me. It tells me that there are a lot more sick and twisted people out there and it makes me doubly glad that I'm not afraid to look out for me and my own. The only message this story has is that it's great to be a criminal, there is no justice in the world and that's acceptable for humankind.",
        "This movie is terrible. Not 'terrifying', there are a lot of excellent horror and war movies for that. Rather terribly empty... The main problem is that I do not see any message in this movie, nor any pleasant feature. It is not funny, it is not horrifying or realistic enough, there is no real character development, no interesting statements about anything. It is an empty, overrated mess. It is basically a shock-parade with old instrumental background music.",
        "This movie sucks!! From the clothes, to the retarded mix of English and Russian to the theatrical way the violent scenes are shot. The theatrical shooting of the violence, makes it impossible to take it seriously. I've seen more graphic violence in modern TV-series than in this over-rated crap. If you feel like you HAVE to watch this movie because it's considered a classic, don't put yourself through the torture. It's classic BS and it should be avoided completely."
       
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
