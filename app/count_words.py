import operator
import nltk
import csv
from math import log
from nltk.collocations import *

# Creates a Python dictionary based on the list of ngrams
# Make sure to have a valid path and filename
def create_dic(filename):
    # Open and read unigram_list_{filename} text file
    f = open("unigram_lists/unigram_list_{}.txt".format(filename), "r")
    file_contents = f.read().splitlines()

    # Create a dictionary to count all the frequency
    count = {}
    for i in file_contents:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1

    # Sort the dictionary in a descending order
    dic = dict(sorted(count.items(), key=operator.itemgetter(1), reverse=True))
    f.close()
    return dic


# Create a csv file with word frequency
def frequency(focus_name):
    focus_dic = create_dic(focus_name)
    with open("frequency/frequency_{}.csv".format(focus_name), "w") as o:
        for key in focus_dic.keys():
            o.write("%s,%s\n" % (key, focus_dic[key]))
    o.close()


# Create a csv file with keyness score calculates with log ratio
def keyness_score(focus_name, ref_name):
    # Create dictionary for focus corpus
    focus_dic = create_dic(focus_name)
    # Create dictionary for reference corpus
    ref_dic = create_dic(ref_name)

    key_dic = calc_keyness(focus_dic, len(focus_dic), ref_dic, len(ref_dic))
    key_sorted = sorted(
        key_dic.items(), key=operator.itemgetter(1), reverse=True)

    key_dic_sorted = dict(key_sorted)

    # Create a csv file with keyness score
    with open("keyness_score_1_cs/keyness_score_{}.csv".format(focus_name), "w") as o:
        for key in key_dic_sorted.keys():
            if 0.4 < key_dic_sorted[key]:
                o.write("%s,%s\n" % (key, key_dic_sorted[key]))
    o.close()


# Calculate keyness with log ratio
def calc_keyness(corpus1, corpus1_size, corpus2, corpus2_size):
    # Calculate keyness score
    # Make sure the term appears at least 3 times in the focus corpus, otherwise remove it
    to_remove = []
    for term in corpus1:
        if corpus1[term] < 3:
            to_remove.append(term)
    for rem in to_remove:
        if rem in corpus1:
            corpus1.pop(rem)
    return {term: log((corpus1[term]/corpus1_size)/((corpus2[term] if term in corpus2 else 0.5)/corpus2_size), 2) for term in corpus1}


# 'Main' function
def start():
    ref_corpus = input("Enter the name of the reference corpus: ")
    focus_corpus = input("Enter the name of the focus corpus: ")
    keyness_score(focus_corpus, ref_corpus)
    return focus_corpus

# start()
