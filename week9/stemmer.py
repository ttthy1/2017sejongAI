from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

input_words=['am','are','is','car','cars','','affecting','affection','affects','kept','keep','caresses','flies','dies','mules','denied','died','agreed','owned','humbled','sized','meeting','stating','siezing','temization','sensational', 'traditional', 'reference', 'colonizer','plotted']

porter=PorterStemmer()
lancaster=LancasterStemmer()
snowball=SnowballStemmer('english')

stemmer_names=['PORTER','LANCASTER','SNOWBALL']
formatted_text='{:>16}'*(len(stemmer_names)+1)
print('\n')
print(formatted_text.format('INPUT WORD', *stemmer_names))
print('='*68)

for word in input_words:
    output=[word,porter.stem(word),lancaster.stem(word),snowball.stem(word)]
    print(formatted_text.format(*output))
