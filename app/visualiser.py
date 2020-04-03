from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import extract_words
import count_words
import csv

extract_words.start()
focus_corpus = count_words.start()
# KEYNESS SCORE
reader = csv.reader(open(
    'keyness_score_1_cs/keyness_score_{}.csv'.format(focus_corpus), 'r', newline='\n'))
d = {}
for k, v in reader:
    d[k] = float(v)

# FREQUENCY
# reader = csv.reader(open(
#     'frequency/frequency_{}.csv'.format(focus_corpus), 'r', newline='\n'))
# d = {}
# for k, v in reader:
#     d[k] = int(v)

mask = np.array(Image.open("../mask/lu-coat-white.png"))

# Create the visualisations
def create_wordcloud(tokens, mask):
    word_cloud = WordCloud(width=412, height=412, max_words=800, background_color='black',
                        mask=mask).generate_from_frequencies(tokens)
    plt.figure(figsize=(10, 8), facecolor='black', edgecolor='red')
    image_colors = ImageColorGenerator(mask)
    plt.imshow(word_cloud.recolor(color_func=image_colors),
               interpolation="bilinear")
    plt.axis('off')
    plt.tight_layout(pad=0)
    # Save visualisation to folder
    word_cloud.to_file("wordclouds_1_lu/wc_{}.png".format(focus_corpus))

# Start the program
create_wordcloud(d, mask)
