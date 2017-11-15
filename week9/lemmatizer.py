from nltk.stem import WordNetLemmatizer

input_words = ['cooking', 'calves', 'be', 'is', 'are', 'done', 'does', 'branded', 'horse', 'randomize', 
        'possibly', 'provision', 'hospital', 'kept', 'scratchy', 'code', 'considers', 'working','eating',
        'application', 'concentration', 'authority', 'variety', 'commitment', 'maintenance', 'residence', 'buyer', 'consumer', 'artist'     ]

# Create lemmatizer object 
lemmatizer = WordNetLemmatizer()

# Create a list of lemmatizer names for display
lemmatizer_names = ['NOUN LEMMATIZER', 'VERB LEMMATIZER']
formatted_text = '{:>24}' * (len(lemmatizer_names) + 1)
print('\n')
print(formatted_text.format('INPUT WORD', *lemmatizer_names))
print('='*75)

# Lemmatize each word and display the output
for word in input_words:
    output = [word, lemmatizer.lemmatize(word, pos='n'),
           lemmatizer.lemmatize(word, pos='v')]
    print(formatted_text.format(*output))
