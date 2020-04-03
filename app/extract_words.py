from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder
from nltk import pos_tag
import nltk
import matplotlib
import os
import re
import csv
import enchant


# Converts PDF files to HTML format with PDFMiner
def pdf_2_html(pdf_doc):
    html_output = "../../papers_html/{}.html".format(pdf_doc)
    convert = "pdf2txt.py -o ../../papers_html/{}.html -t html ../../papers_pdf/{}.pdf".format(
        pdf_doc, pdf_doc)
    os.system(convert)

# Get the abstract of the PhD thesis
def get_abstract(soup):
    # Convert BeautifulSoup object to String
    str_soup = str(soup)
    # Find beginning of Abstract
    if re.search('(font-size:([1][2-9]|[2-3][0-9])px">(Abstract|ABSTRACT|Summary|[A-Za-z]{2,}(\t| )ABSTRACT|[A-Za-z]{2,}(\t| )Abstract))', str_soup):
        first_index = re.search(
            '(font-size:([1][2-9]|[2-3][0-9])px">(Abstract|ABSTRACT|Summary|[A-Za-z]{2,}(\t| )ABSTRACT|[A-Za-z]{2,}(\t| )Abstract))', str_soup).start()
        new_soup = str_soup[first_index+24:]
    # 1st attempt at finding the end of Abstract
    second_index = re.search(
        '(font-size:([1-3][1-9]|[2][0])px">([^\Wv]{4,}\n|[^\Wv]{4,}\t\n|[^\Wv]{4,} \n|[^\Wv]{4,}<|[A-Za-z]{2,8} [A-Za-z]{2,8} [A-Za-z]{2,8}\n|[A-Za-z]{2,8} [A-Za-z]{2,8} [A-Za-z]{2,8} \n|([A-Za-z]{2,8} [A-Za-z]{2,8} [A-Za-z]{2,8}<){1}))', new_soup).start()
    # Check if Abstract is longer than 500 characters
    if second_index > 500:
        # 2nd attempt at finding the end of Abstract
        second_index = re.search(
            '(font-size:([1-3][1-9]|[2][0])px">([^\Wv]{4,}\n|[^\Wv]{4,}\t\n|[^\Wv]{4,} \n|[^\Wv]{4,}<|[A-Za-z]{2,8} [A-Za-z]{2,8} [A-Za-z]{2,8}\n|[A-Za-z]{2,8} [A-Za-z]{2,8} [A-Za-z]{2,8} \n|([A-Za-z]{2,8} [A-Za-z]{2,8} [A-Za-z]{2,8}<){1}))', new_soup).start()
        abstract_string = new_soup[:second_index] + '">'
    else:
        newer_soup = new_soup[second_index+16:]
        # 2nd attempt at finding the end of Abstract
        new_second_index = re.search(
            '(font-size:([1-3][1-9]|[2][0])px">([^\Wv]{4,}\n|[^\Wv]{4,}\t\n|[^\Wv]{4,} \n|[^\Wv]{4,}<|[A-Za-z]{2,8} [A-Za-z]{2,8} [A-Za-z]{2,8}\n|[A-Za-z]{2,8} [A-Za-z]{2,8} [A-Za-z]{2,8} \n|([A-Za-z]{2,8} [A-Za-z]{2,8} [A-Za-z]{2,8}<){1}))', newer_soup).start()
        abstract_string = newer_soup[:new_second_index] + '">'
    return abstract_string

# Delete the references from the thesis
def cut_ref(soup):
    offset = 100000
    # Convert BeautifulSoup object to String
    str_soup = str(soup)
    second_half = str_soup[offset:]
    # Find beginning of References / Bibliohraphy
    if re.search('(font-size:([1][2-9]|[2-4][0-9])px\">(References(\n| \n|\t\n|<)|Bibliography(\n| \n|\t\n)|References: (\n| \n|\t\n|<)|[1-9][0-9]\.  References \n))', second_half, flags=re.I):
        ref_index = re.search(
            '(font-size:([1][2-9]|[2-4][0-9])px\">(References(\n| \n|\t\n|<)|Bibliography(\n| \n|\t\n)|References: (\n| \n|\t\n|<)|[1-9][0-9]\.  References \n))', second_half, flags=re.I).start()
    return ref_index+offset

# Delete the acknowledgements from the thesis
def cut_ack(soup):
    # Convert BeautifulSoup object to String
    str_soup = str(soup)
    # Find beginning of Acknowledgements
    if re.search('(font-size:([1][2-9]|[2-4][0-9])px\">(Acknowledgement(\n| \n|\t\n|<)|Acknowledgements(\n| \n|\t\n)))', str_soup, flags=re.I):
        ack_index = re.search(
            '(font-size:([1][2-9]|[2-4][0-9])px\">(Acknowledgement(\n| \n|\t\n|<)|Acknowledgements(\n| \n|\t\n)))', str_soup, flags=re.I).start()
    new_soup = str_soup[ack_index+32:]
    # Find ending of Acknowledgements
    second_index = re.search(
        '(font-size:([1-3][1-9]|[2][0])px">([^\Wv]{4,}\n|[^\Wv]{4,}\t\n|[^\Wv]{4,} \n|[^\Wv]{4,}<|[A-Za-z]{2,8} [A-Za-z]{2,8} [A-Za-z]{2,8}\n|[A-Za-z]{2,8} [A-Za-z]{2,8} [A-Za-z]{2,8} \n|([A-Za-z]{2,8} [A-Za-z]{2,8} [A-Za-z]{2,8}<){1}))', new_soup).start()
    no_ack_string = str_soup[ack_index:second_index+ack_index-32]
    return ack_index, ack_index+second_index

# Find the introduction of the thesis
def get_intro(soup):
    # Convert BeautifulSoup object to String
    str_soup = str(soup)
    # Find beginning of Introduction
    if re.search('(font-size:([1][2-9]|[2-4][0-9])px\">((An |Chapter 1:  |1 |1. |1. 0 |1.\t|1.  |)Introduction(\n| \n|\t\n)|(1. |)Research Overview (\n| \n|\t\n|<)))', str_soup, flags=re.I):
        intro_index = re.search(
            '(font-size:([1][2-9]|[2-4][0-9])px\">((An |Chapter 1:  |1 |1. |1. 0 |1.\t|1.  |)Introduction(\n| \n|\t\n)|(1. |)Research Overview (\n| \n|\t\n|<)))', str_soup, flags=re.I).start()
    return intro_index

# Delete the declaration of the thesis
def cut_dec(soup):
    # Convert BeautifulSoup object to String
    str_soup = str(soup)
    # Find beginning of Declaration
    if re.search('(font-size:([1][2-9]|[2-4][0-9])px\">(Declaration(\n| \n|\t\n|<)|Declaration of Authorship(\n| \n|\t\n)))', str_soup, flags=re.I):
        dec_index = re.search(
            '(font-size:([1][2-9]|[2-4][0-9])px\">(Declaration(\n| \n|\t\n|<)|Declaration of Authorship(\n| \n|\t\n)))', str_soup, flags=re.I).start()
    new_soup = str_soup[dec_index+32:]
    # Find ending of Declaration
    second_index = re.search(
        '(font-size:([1-3][1-9]|[2][0])px">([^\Wv]{4,}\n|[^\Wv]{4,}\t\n|[^\Wv]{4,} \n|[^\Wv]{4,}<|[A-Za-z]{2,8} [A-Za-z]{2,8} [A-Za-z]{2,8}\n|[A-Za-z]{2,8} [A-Za-z]{2,8} [A-Za-z]{2,8} \n|([A-Za-z]{2,8} [A-Za-z]{2,8} [A-Za-z]{2,8}<){1}))', new_soup).start()
    no_dec_string = str_soup[dec_index:second_index+dec_index-32]
    return no_dec_string

# Get the main context of the thesis
def get_top_words(html_file):
    # Convert PDF file to HTML
    pdf_2_html(html_file)
    html = open("../../papers_html/{}.html".format(html_file))
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html, "html.parser")
    all_text = soup.find_all(text=True)
    # Convert BeautifulSoup object to String
    str_soup = str(soup)
    # Extract the Abstract and "clean" the PhD paper
    abstract_string = get_abstract(soup)
    index_intro = get_intro(soup)
    index_ref = cut_ref(soup)
    clean_string = str_soup[index_intro:index_ref]

    # Create an HTML file for the Abstract
    with open("../../abstracts_html/{}-abstract.html".format(html_file), "w") as file:
        file.write(abstract_string)

    # Create an HTML file for the Papers without Acknowledgements
    with open("../../clean_html/{}-clean.html".format(html_file), "w") as file:
        file.write(clean_string)

    abstract_html = open(
        "../../abstracts_html/{}-abstract.html".format(html_file))

    clean_html = open("../../clean_html/{}-clean.html".format(html_file))

    abstract_soup = BeautifulSoup(abstract_html, "html.parser")
    clean_soup = BeautifulSoup(clean_html, "html.parser")
    abstract_text = abstract_soup.find_all(text=True)
    clean_paper = clean_soup.find_all(text=True)
    #abstract_tokens = clean_text(abstract_text)
    corpus_tokens = clean_text(clean_paper)
    #bigrams = get_bigrams(clean_paper)
    #trigrams = get_trigrams(clean_paper)

    # Save words to text file
    # Need to change path name, depending on what you want to do
    with open('unigram_lists/unigram_list_{}.txt'.format(html_file), 'w') as f:
        for i in corpus_tokens:
            f.write("%s\n" % i)

    text = " ".join(corpus_tokens)
    return text

# Get the bigrams from the thesis
def get_bigrams(text):
    output = ''
    html_tags = [
        '[document]',
        'noscript',
        'body',
        'header',
        'html',
        'meta',
        'head',
        'input',
        'script',
        'style',
    ]

    # Remove blacklisted words
    for word in text:
        if word.parent.name not in html_tags:
            output += '{} '.format(word)

    clean_tokens = []
    root_tokens = []
    wnl = WordNetLemmatizer()
    # Tokenize the words
    word_tokens = word_tokenize(output)
    tagged = pos_tag(word_tokens)
    word_tokens = [x for (x, y) in tagged if y not in (
        'NNP') and y not in ('NNPS')]
    stop_words = get_stop_words()
    punctuations = [",", "'", ".",
                    "(", ")", ":", ";", "-", "[", "]", "/", "?", "!", "%"]
    english = enchant.Dict("en_GB")

    root_tokens = [wnl.lemmatize(t) for t in word_tokens]
    # Remove numbers, punctuations and 'minify' the tokens
    for token in root_tokens:
        if token.lower() in stop_words:
            pass
        else:
            if token not in punctuations and token.isalpha() and len(token) > 2:
                clean_tokens.append(token.lower())

    # Bigrams
    bigrams = []
    bigrm = nltk.bigrams(clean_tokens)
    for tup in list(bigrm):
        bigrams.append(" ".join(tup))
    return bigrams

# Get the trigrams from the thesis
def get_trigrams(text):
    output = ''
    html_tags = [
        '[document]',
        'noscript',
        'body',
        'header',
        'html',
        'meta',
        'head',
        'input',
        'script',
        'style',
    ]

    # Remove blacklisted words
    for word in text:
        if word.parent.name not in html_tags:
            output += '{} '.format(word)

    clean_tokens = []
    root_tokens = []
    wnl = WordNetLemmatizer()
    # Tokenize the words
    word_tokens = word_tokenize(output)
    tagged = pos_tag(word_tokens)
    word_tokens = [x for (x, y) in tagged if y not in (
        'NNP') and y not in ('NNPS')]
    stop_words = get_stop_words()
    punctuations = [",", "'", ".",
                    "(", ")", ":", ";", "-", "[", "]", "/", "?", "!", "%"]
    english = enchant.Dict("en_GB")

    root_tokens = [wnl.lemmatize(t) for t in word_tokens]
    # Remove numbers, punctuations and 'minify' the tokens
    for token in root_tokens:
        if token.lower() in stop_words:
            pass
        else:
            if token not in punctuations and token.isalpha() and len(token) > 2:
                clean_tokens.append(token.lower())

    # Trigrams
    trigrams = []
    trigrm = nltk.trigrams(clean_tokens)
    for tup in list(trigrm):
        trigrams.append(" ".join(tup))
    # print(trigrams)
    return trigrams

# Get the unigrams from the thesis
def clean_text(text):
    output = ''
    html_tags = [
        '[document]',
        'noscript',
        'body',
        'header',
        'html',
        'meta',
        'head',
        'input',
        'script',
        'style',
    ]

    # Remove blacklisted words
    for word in text:
        if word.parent.name not in html_tags:
            output += '{} '.format(word)

    clean_tokens = []
    root_tokens = []
    wnl = WordNetLemmatizer()
    # Tokenize the words
    word_tokens = word_tokenize(output)
    tagged = pos_tag(word_tokens)
    word_tokens = [x for (x, y) in tagged if y not in (
        'NNP') and y not in ('NNPS')]
    stop_words = get_stop_words()
    punctuations = [",", "'", ".",
                    "(", ")", ":", ";", "-", "[", "]", "/", "?", "!", "%"]
    english = enchant.Dict("en_GB")

    # Remove numbers, punctuations and 'minify' the tokens
    for token in word_tokens:
        if token.lower() not in stop_words and token not in punctuations and token.isalpha() and len(token) > 2 and english.check(token):
            clean_tokens.append(token.lower())

    # Perform lemmatization on the tokens (nltk)
    clean_tagged = pos_tag(clean_tokens)
    for word, tag in clean_tagged:
        if tag in (('NN') or ('NNS') or ('NNP') or ('NNPS')):
            if len(wnl.lemmatize(word, pos="n")) > 2 and wnl.lemmatize(word, pos="n") not in stop_words and word not in stop_words:
                root_tokens.append(wnl.lemmatize(word, pos="n"))
        elif tag in (('VB') or ('VBD') or ('VBG') or ('VBN') or ('VBP') or ('VBZ')):
            if len(wnl.lemmatize(word, pos="v")) > 2 and wnl.lemmatize(word, pos="v") not in stop_words and word not in stop_words:
                root_tokens.append(wnl.lemmatize(word, pos="v"))
        else:
            if len(wnl.lemmatize(word, pos="v")) > 2 and wnl.lemmatize(word, pos="n") not in stop_words and word not in stop_words:
                root_tokens.append(wnl.lemmatize(word, pos="v"))
    return root_tokens

# Expand the list of stop words
def get_stop_words():
    stop_words = stopwords.words("english")
    more_stop_words = ["research", "idea", "use", "find", "present", "boolean", "expr", "tag", "params", "therefore", "however", "include", "reduce", "discuss", "one", "two", "give", "could", "would", "errors", "correct", "show", "also", "result", "error", "spell", "sample", "evaluate", "paper", "research", "kinda", "use", "find", "present", "describe"]
    stop_words.extend(more_stop_words)
    return stop_words

# 'Main' function
def start():
    os.system("clear")
    doc = input("Enter the name of the PDF doc: ")
    text = get_top_words(doc)
    return text


# Call start function
#start()
