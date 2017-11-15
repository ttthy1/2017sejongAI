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
       "This is a movie of real beauty. It made me cry. I felt moved in a way that happens very rarely. It was an inspiration.",
       "Certainly the best movie of this year and one of the best ever made. The story, the story itself is great.",
       "Honestly let me just say this 1st Sean Pean made this story come to life, true life in every moment in the film. This movie has nothing wrong with it, it is perfect in every way shape and form. He did something that brought me to something I never could understand of what either i have to do or what i need to do in my life.",
       "How can you describe a film like Into the Wild? One of the greats & by far my favourite film of all time. The only film I have ever seen which manages to capture the true spirit of adventure. The beauty of Into the Wild for me is that what Christopher McCandless is running from is never as important as what he is running to. Sean Penn has directed this film with magnificent precision and imaginative grace. Every scene, every image used in this film tells it's own story and the countless people Chris meets along the way are as magical and integral to his journey as his survival in the wilderness.",
       "Sean Penn's Into the Wild is beautiful, staggering, thoughtful, a labor of love made by a filmmaker with real passion for the story he set out to tell. Like a Terrence Malick picture, Into the Wild transcends any conception such as 'if you only see one film this year' and goes beyond the very notion of 'Oscar-worthy'",
       "This is a great movie, which makes you think of your life. I don't know why did I cry so hard when I saw this movie. Maybe the character Christopher (Alex) resembled my own life. And I think, it resembles the life of every guy to a certain extent, as everyone of us go for escapism at some points of our life due to various pressures. This movie shows the life of a man who leaves every thing, absolutely everything and go for an ultimate and pure escapism, and what does he learn and realize at the end.",
       "I've seen so many good movies, but very less perfect movies. Into The Wild IS a perfect movie : perfect images, perfect main actor, perfect acting, perfect music, perfect message. Everybody should watch this film, it pictures a model of a philosophic path to happiness.",
       "Into the Wild was spectacular, breathtaking and compelling to say the least. I sat and watched it with my home girls and we were all three enthralled by it. The messages behind the movie, the acting done by Johnny Truelove, it was outstanding. The only thing I'd ever heard of the movie is that older gentleman was nominated for an Oscar, and it's a shame he didn't win it, I was pushed to tears at the end of his performance. I highly recommend this movie to anybody, its a movie to make you think about your life. I was shocked.",
       "this movie is one of my favorite movies I've seen . i like the way Christopher choose for his life. i recommended this movie to anyone who like travel and live in nature. i can imagine al the days and nights he live alone in nature . Emie Hirsch was great for this character, i think if you like to live in nature you should like this movie.",
       "After watching this movie I felt a strange empty feeling inside. I was sad but first I guess I was more shocked. The most shocking part of this movie is off course the end. Although the sadness the movie brings, I am very glad this movie was made and shows how hard reality can be sometimes. It reminded me that I should cherish the feeling of freedom I had during my traveling and remembers me to live life to the fullest.",
       "Please spare yourself and avoid this film, it is utterly boring and undeserving of a high rating. This should not be in the top 250.",
       "This film is a 1...at best. I will never, not ever, get the time back that I wasted watching this dreadful film. I am completely perplexed as to how this movie scored an 8?! I can only assume that no one else saw the film that I saw.",
       "Judging from some of the comments I've read, this has to be one of the most overrated and misguided films of all time. Had there been an option for 0 out of 10, I would have voted thus for sure.",
       "This was a very unsatisfying movie. It starts out with promise and spectacularly fails to deliver . Where to start? The basic premise of a chap going off to live alone in the wild with adventures all along the way is a good one. But nothing really interesting or compelling ever happens in this.",
       "Uninteresting and very boring story about a selfish, arrogant man, blaming all his problems on his parents, who runs away from his family to live in the wilderness for 4 months before he finds out that happiness is found in the company of other people and family.",
       "I would rate it a zero if possible. The movie was so boring and contrived that I felt nauseous after the first hour. I wish I had asked for a refund. Sean Penn owes me $16.00!",
       "Oh dear. This film is in the IMDb top 250? How on earth did that happen? All I will say is that if you are ever inclined to watch this pile of trash run away very, very quickly.",
       "I really hated it. The movie had a really moralizing tone, and it was incredibly self-conscious. All the time it felt like the guy was posing for the camera.",
       "This movie disgust me. I feel sorry for all people being brainwashed by the movieindustry and thinking this is something real. I can honestly say that this is the worst movie I've ever seen..And I have seen a lot, so it's probably hard to beat.",
       "Although I enjoyed some of the scenery, this movie was mostly a wast of time.Sean Penn does a great job directing, the actors are excellent, and the scenery is breathtaking. However, none of this is able to hide the fact that this is a movie about someone who wasted his life trying to prove something, and in the end proved the exact opposite. Yes, I know he was an actual person who really died, but I don't really care. I personally know lots of people who died (some of which also went to Emory) in much more tragic and meaningful ways."
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
