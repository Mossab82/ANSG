import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import wikipediaapi
import hashlib
import pyarabic.araby as araby
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pyarabic.arabrepr
import os
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter
from nltk.parse import stanford as SParse
from nltk.tag import stanford as STag
from nltk.tokenize import StanfordSegmenter
from polyglot.text import Text
from rake_nltk import Rake
import nltk
from nltk.tag import StanfordPOSTagger
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.isri import ISRIStemmer
from nltk.tag import StanfordNERTagger
from nltk.stem.porter import PorterStemmer
from tashaphyne.stemming import ArabicLightStemmer
import re
import mysql.connector
import ssl
import argparse
import repustate
from sklearn.feature_extraction.text import CountVectorizer

import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

from polyglot.downloader import downloader

#downloader.download("embeddings2.ar")
#downloader.download("ner2.ar")
os.environ['STANFORD_MODELS'] = 'stanford-segmenter-2018-10-16/data/;stanford-postagger-full-2018-10-16/models/'
os.environ['STANFORD_SEGMENTER'] = 'stanford-segmenter-2018-10-16'
os.environ['STANFORD_PARSER'] = 'stanford-parser-full-2018-10-17'
os.environ['CLASSPATH'] = 'stanford-parser-full-2018-10-17'
os.environ["JAVA_HOME"] = "C:\Program Files\Java\jdk-13.0.1"

def document_Clustering():
    tokenizer = RegexpTokenizer(r'\w+')
    s_word = create_stop_wrods()
    cv = CountVectorizer(max_df=0.85, stop_words=s_word)
    db = DataBase_Connector("Story_Narative")
    cursor = db.cursor()
    cursor.execute("SELECT * FROM Naitave_Story")
    myresult = cursor.fetchall()
    Docs = list()
    for r in myresult:
        story_id = r[0]
        full_text = str(r[2])
        full_text = text_segmenter(full_text)
        full_text = SW_Remover(full_text)

        Docs.append(full_text)
        DF_S(full_text)
    DF_D(Docs)
    # print("Full text"+str(full_text))
    # Segmented_text = text_segmenter(full_text)
    # tokens = tokenizer.tokenize(full_text)
    # word_Count_Vector=cv.fit_transform(Docs)
    # print(cv.vocabulary_.keys())
    # tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    # tfidf_transformer.fit(word_Count_Vector)
    # print(tfidf_transformer)
    # df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf_weights"])
    # df_idf.sort_values(by=['idf_weights'])
    # print(df_idf)
    # tf_idf_vector = tfidf_transformer.transform(word_Count_Vector)
    # feature_names = cv.get_feature_names()
    # get tfidf vector for first document
    # print(feature_names)
    # first_document_vector = tf_idf_vector[3]
    # print the scores


#  dfv = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf-document"])
# dfv.sort_values(by=["tfidf-document"])
# print(dfv)

# settings that you use for count vectorizer will go here
# tfidf_vectorizer = TfidfVectorizer(use_idf=True)

# just send in all your docs here
# tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(Docs)
# first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[0]

# place tf-idf values in a pandas data frame
# df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(),
#                  columns=["tfidf"])
# df.sort_values(by=["tfidf"])
# print(df)

# function to ssplit the documents into substories and populate the Story_Narative Database


def DF_IDF_D(processed_text):
    tf_idf = {}
    for i in range(len(processed_text)):
        tokens = processed_text[i]
        counter = Counter(tokens + processed_title[i])
        for token in np.unique(tokens):
            tf = counter[token] / words_count
            df = doc_freq(token)
            idf = np.log(N / (df + 1))
            tf_idf[doc, token] = tf * id


def DF_S(processed_text):
    DF = {}
    for i in range(len(processed_text)):
        tokens = processed_text[i]
        # print(tokens)
        try:
            DF[tokens].add(i)
        except:
            DF[tokens] = {i}

    for i in DF:
        DF[i] = len(DF[i])
    print("DF_S=", DF)


def DF_D(processed_text):
    DF = {}
    fDF = {}
    for i in range(len(processed_text)):
        tokens = processed_text[i]
        for w in tokens:
            try:
                DF[w].add(i)
            except:
                DF[w] = {i}

    for i in DF:
        DF[i] = len(DF[i])
    print("DF_D=", DF)


def Stories_Split(Paragraphs):
    db = DataBase_Connector("Story_Narative")
    cursor = db.cursor()

    Titles = list()
    Stories = list()
    for par in Paragraphs:
        # print("<--------------"+str(Paragraphs.index(par))+"-------------->")

        t = Title_extractor(par)
        if t:
            Titles.append(t)
        # print("<------------------------------------------------>")
    for x in range(0, len(Titles)):
        if x == len(Titles) - 1:
            Stories.append(story_text_extractor(Paragraphs, Titles[x], "None"))
        else:
            Stories.append(story_text_extractor(Paragraphs, Titles[x], Titles[x + 1]))

    for s in Stories:
        for key in s:
            sql = "INSERT INTO Naitave_Story (Story_Title,Story_Text) VALUES (%s, %s)"
            story_text = TreebankWordDetokenizer().detokenize(s[key])
            story_text = story_text.replace("\\n", '\n')
            val = (str(key), story_text)
            cursor.execute(sql, val)
            db.commit()
    narative_Ngram()

    return Stories


def create_stop_wrods():
    s_word = stopwords.words("arabic")
    s_word.append('ان')
    s_word.append('كان')
    s_word.append('و')
    s_word.append('ه ')
    s_word.append(' ه ')
    s_word.append(' ه')
    s_word.append('/tه')
    s_word.append('ه/t')
    s_word.append(' ')
    s_word.append('اذ')
    s_word.append('في')
    s_word.append('من')
    s_word.append('ب')
    s_word.append('كان')
    s_word.append(' كان')
    s_word.append('بينما')
    s_word.append('هو')
    s_word.append('تحت')
    s_word.append('بعد')
    s_word.append('ف')
    s_word.append('عم')
    return s_word


# fucntion to create the story nartive Ngrams and populate the database
def narative_Ngram():
    db = DataBase_Connector("Story_Narative")
    cursor = db.cursor()
    s_word = create_stop_wrods()

    tokenizer = RegexpTokenizer(r'\w+')
    POS = make_tagger()
    N_Words = list()
    db = DataBase_Connector("Story_Narative")
    cursor = db.cursor()
    cursor.execute("SELECT * FROM Naitave_Story")
    myresult = cursor.fetchall()
    for r in myresult:
        story_id = r[0]
        full_text = str(r[2])
        # print("Full text"+str(full_text))
        Segmented_text = text_segmenter(full_text)
        tokens = tokenizer.tokenize(Segmented_text)
        Tagged_Text = POS.tag(tokens)
        for t in Tagged_Text:
            if "NN" in t[1]:
                v = t[1].split("/")
                if not v[0] in s_word:
                    N_Words.append(v[0])
            elif "JJ" in t[1]:
                v = t[1].split("/")
                if not v[0] in s_word:
                    N_Words.append(v[0])
            elif "RB" in t[1]:
                v = t[1].split("/")
                if not v[0] in s_word:
                    N_Words.append(v[0])

        # print(str(N_Words))

        # print("Tagged Text=",str(Tagged_Text))
        for x in range(1, 4):
            n = N_Gram(N_Words, x)
            res = Counter(n).items()
            print(type(res))
            for key, value in res:
                sql = "INSERT INTO Narative_Ngram  (Ngram,Freq,Stroy_ID,N_Type) VALUES (%s, %s, %s,%s)"
                val = (key, value, story_id, str(x))
                cursor.execute(sql, val)
                db.commit()


# function to split and document into srories and sentances  and populating the database for further Analaysis
def Para_Segmenter(Paragraphs, db):
    Titles = list()
    Stories = list()
    for par in Paragraphs:
        # print("<--------------"+str(Paragraphs.index(par))+"-------------->")

        t = Title_extractor(par)
        if t:
            Titles.append(t)
        # print("<------------------------------------------------>")
    for x in range(0, len(Titles)):
        if x == len(Titles) - 1:
            Stories.append(story_text_extractor(Paragraphs, Titles[x], "None"))
        else:
            Stories.append(story_text_extractor(Paragraphs, Titles[x], Titles[x + 1]))

    for s in Stories:
        for key in s:
            print(key)
            print(s[key])

    for s in Stories:
        for key in s:
            Sentances_extractor(Stories.index(s), key, s[key])


# function to extract the story text by parssing the text between 2 titles
def story_text_extractor(Par, start, end):
    story_text = list()
    Story = dict()
    for P in Par:
        if start in P:
            x = Par.index(P)

            while x < len(Par):
                if not end in Par[x]:
                    story_text.append(Par[x])
                    x = x + 1
                else:
                    break
    Story[start] = story_text
    return Story


# function to extract the story title
def Title_extractor(par):
    Par = par.split("\n")
    Title = None
    if len(Par) > 1:
        if len(Par[1]) > 3:
            Title = Par[1]
    return Title


# Function to extract the verbs in a storty with coordinates and sentament
def Verbs_Extractor():
    db = DataBase_Connector("Story_Narative")
    cursor = db.cursor()
    client = repustate.Client(api_key=os.environ["ANSG_REPUSTATE_API_KEY"])
    sql_select_Query = "select Sentance,Sent_Loc_Story,Sent_Loc_Par,Sent_Loc_In_Par from Nartive_Sentances"
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    print(str(cursor.rowcount))
    for r in records:
        sentance = r[0]
        tokenized_sentance = nltk.word_tokenize(sentance)
        segmented_sentance = nltk.word_tokenize(sentance_segmenter(tokenized_sentance))
        # sw_sentance=SW_Remover(segmented_sentance)
        POS = make_tagger()
        Tagged_Sentance = POS.tag(segmented_sentance)
        counter = 1
        for ts in Tagged_Sentance:
            if "V" in ts[1]:
                v = ts[1].split("/")
                sentement = client.sentiment(text=v[0], lang='ar')
                print(sentement)
                sr = float(sentement['score'])
                print(sr)

                sql = "INSERT INTO Nartive_Verbs (Verb,Verb_Loc_Story,Verb_Loc_Par,Verb_Loc_Sent,Verb_Loc_In_Sent,Verb_Type,Verb_Sentiment) VALUES (%s, %s, %s, %s, %s, %s ,%s)"
                val = (v[0], r[1], r[2], r[3], str(counter), v[1], sr)
                cursor.execute(sql, val)
                db.commit()
                counter = counter + 1


# Function to  Extract Sentances from Story find their setnaments and populate the database
def Sentances_extractor(s_id, s_title, s_body):
    # print("Story ID = "+str(s_id+1))
    # print("Story Title = " + s_title)
    # print(type(s_body))
    db = DataBase_Connector("Story_Narative")
    cursor = db.cursor()
    Story_Sentances = list()
    client = repustate.Client(api_key=os.environ["ANSG_REPUSTATE_API_KEY"])
    for p in s_body:
        par_sentances = p.split('،')
        for sent in par_sentances:
            Story_Sentances.append(
                ((s_id + 1, s_body.index(p) + 1, par_sentances.index(sent) + 1, sent.replace('\n', " "))))

    for s in Story_Sentances:

        try:
            # print("Paragraph_ID="+ str(s[1]))
            # sentement=client.sentiment(text=s[3])
            # sent_hash=hashlib.sha1(s[3].encode())
            # print(sentement)
            r = float(sentement['score'])

            sql = "INSERT INTO Nartive_Sentances (Sent_Hash,Sentance,Sent_Loc_Story,Sent_Loc_Par,Sent_Loc_In_Par) VALUES ( %s, %s, %s, %s, %s)"
            val = (sent_hash.hexdigest(), str(s[3]), str(s[0]), str(s[1]), str(s[2]))
            cursor.execute(sql, val)
            db.commit()
            print("--------------------> Added to Database")
        except:
            print("Duplicated Sentance")


# Function to check a Token for WIKIPEIDA for finding name enttites
def Wiki_NER(s):
    print("checking Wiki-NER for", s)
    s = s.replace(" ", "_")
    wiki_wiki = wikipediaapi.Wikipedia('ar')
    page_py = wiki_wiki.page(s)
    if (page_py.exists()):
        return True
    else:
        return False


# Function to create an Ngram of a give token
def N_Gram(s, n):
    # Replace all none alphanumeric characters with spaces
    # Break sentence in the token, remove empty tokens
    s = TreebankWordDetokenizer().detokenize(s)
    token = [token for token in s.split(" ") if token != ""]

    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[token[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]


#

# Stanford NER Function
def ANER(t):
    print(len(t), "Checking ANER for", t)
    try:
        text = Text(t)
        entity = text.entities
        return entity[0]
    except:
        return None


# Database Connector Function
def DataBase_Connector(database_Name):
    mydb = mysql.connector.connect(user=os.getenv("ANSG_MYSQL_USER", "root"), password=os.environ["ANSG_MYSQL_PASSWORD"],
                                   host=os.getenv("ANSG_MYSQL_HOST", "127.0.0.1"), database=database_Name,
                                   auth_plugin='mysql_native_password')

    return mydb


# Stop Word Remover Function
def SW_Remover(t):
    s_word = stopwords.words("arabic")
    s_word.append('ان')
    s_word.append('كان')
    s_word.append('و')
    s_word.append('ه ')
    s_word.append(' ه ')
    s_word.append(' ه')
    s_word.append('/tه')
    s_word.append('ه/t')
    s_word.append(' ')
    s_word.append('اذ')
    s_word.append('في')
    s_word.append('من')
    s_word.append('ب')
    s_word.append('كان')
    s_word.append(' كان')
    s_word.append('بينما')
    s_word.append('هو')
    s_word.append('تحت')
    s_word.append('بعد')
    text = nltk.word_tokenize(t)
    for word in text:
        word = re.sub(r"[\n\t\s]*", "", word)
        if word in s_word:
            text.remove(word)
            #   print("removed"+word)
    for word in text:
        word = re.sub(r"[\n\t\s]*", "", word)
        if len(word) < 3:
            text.remove(word)
            # print("removed"+word)
    for word in text:
        word = re.sub(r"[\n\t\s]*", "", word)
        if word == 'ه' or word == "كان" or word == "ان" or word == "تحت" or word == "هو" or word == "بعد":
            text.remove(word)
            # print("removed"+word)

    return text


# NGram NER Function
def N_ANER(t, l, db, of):
    # print("text before removal ="+ t)
    text = SW_Remover(t)
    N = N_Gram(text, l)

    for n in N:
        if (len(n) > 1 and n != None):
            if Wiki_NER(n):
                of.write(str(n) + "----->Found in WIKI-NER\n")
                print((str(n) + "----->Found in WIKI-NER"))
            else:
                # print(n,"----->Not Found")
                x = 0
        else:
            if ANER(n):
                of.write(str(n) + "----->Found in ANER\n")
                print((str(n) + "----->Found in ANER"))
            elif Wiki_NER(n):
                of.write(str(n) + "----->Found in WIKI-NER\n")
                print((str(n) + "----->Found in WIKI-NER"))
            else:
                # print(n,"----->Not Found")
                x = 0


# Function to Catagorize Tokens based on POS Taggaging
def word_catagrizer(tokens, db, ntf):
    cursor = db.cursor()
    counter = 0

    for token in tokens:
        # if token not in s_word:
        words.append(token)
        word = tagged_data[counter][1]
        word = word.split('/')
        if (len(word) > 1):
            if "NN" in word[1]:
                try:
                    ntf.write(word[0].rstrip('\n') + " ")
                    sql = "INSERT INTO Nouns (Token,Type) VALUES (%s, %s)"
                    val = (word[0], word[1])
                    cursor.execute(sql, val)
                    db.commit()
                    print("--------------------> Added to Database")
                except:
                    print("Duplicated")

            elif "V" in word[1]:
                try:

                    sql = "INSERT INTO Verbs (Token,Type) VALUES (%s, %s)"
                    val = (word[0], word[1])
                    cursor.execute(sql, val)
                    db.commit()
                    print("--------------------> Added to Database")
                except:
                    print("Duplicated")
            elif "JJ" in word[1]:
                try:
                    ntf.write(word[0].rstrip('\n') + " ")
                    sql = "INSERT INTO Adjectives (Token,Type) VALUES (%s, %s)"
                    val = (word[0], word[1])
                    cursor.execute(sql, val)
                    db.commit()
                    print("--------------------> Added to Database")
                except:
                    print("Duplicated")
            elif "PRP" in word[1]:
                try:
                    sql = "INSERT INTO Pronouns (Token) VALUES (%s)"
                    val = (word[0],)
                    cursor.execute(sql, val)
                    db.commit()
                    print("--------------------> Added to Database")
                except:
                    print("Duplicated")
            elif "IN" in word[1]:
                try:
                    sql = "INSERT INTO Prepos (Token) VALUES (%s)"
                    val = (word[0],)
                    cursor.execute(sql, val)
                    db.commit()
                    print("--------------------> Added to Database")
                except:
                    print("Duplicated")
            elif "CC" in word[1]:
                try:

                    sql = "INSERT INTO Con (Token) VALUES (%s)"
                    val = (word[0],)
                    cursor.execute(sql, val)
                    db.commit()
                    print("--------------------> Added to Database")
                except:
                    print("Duplicated")
            elif "RB" in word[1]:
                try:
                    ntf.write(word[0].rstrip('\n') + " ")
                    sql = "INSERT INTO Adverbs (Token) VALUES (%s)"
                    val = (word[0],)

                    cursor.execute(sql, val)
                    db.commit()
                    print("--------------------> Added to Database")
                except:
                    print("Duplicated")
            else:
                try:
                    sql = "INSERT INTO Others (Token) VALUES (%s)"
                    val = (word[0],)
                    cursor.execute(sql, val)
                    db.commit()
                    print("--------------------> Added to Database")
                except:
                    print("Duplicated")

            counter = counter + 1
        else:
            counter = counter + 1
    # else:
    # Outfile.write("\t" + token + "----->" + str(tagged_data[counter]) + "\n")
    #    counter = counter + 1


# Text Normaliaztion Function
def Normalize_Text(text):
    N_H_T = araby.strip_harakat(text)
    N_T_T = araby.strip_tashkeel(N_H_T)

    return N_T_T


# POS Tagger Function
def make_tagger():
    stanford_dir = "stanford-postagger-full-2018-10-16"  # change it into your own path
    modelfile = stanford_dir + "/models/arabic.tagger"  # model file
    jarfile = stanford_dir + "/stanford-postagger.jar"  # jar file
    st = StanfordPOSTagger(model_filename=modelfile, path_to_jar=jarfile)  # create a tagger
    return st


# Setancess Segmenter
def sentance_segmenter(t):
    segmenter = StanfordSegmenter('stanford-segmenter-2018-10-16/stanford-segmenter-3.9.2.jar')
    segmenter.default_config('ar')
    text = segmenter.segment(t)
    return text


# Text Segmenter
def text_segmenter(text):
    temp_file = open('tmp_f_out.txt', 'w', encoding='utf-8')
    # print("Text="+text)
    temp_file.write(str(text))
    temp_file.close()
    segmenter = StanfordSegmenter('stanford-segmenter-2018-10-16/stanford-segmenter-3.9.2.jar')
    segmenter.default_config('ar')
    s_text = segmenter.segment_file('tmp_f_out.txt')
    # print("segmented text=",s_text)
    return s_text


# Text Parsser
def text_parser(t):
    parser = SParse.StanfordParser(model_path='edu/stanford/nlp/models/lexparser/arabicFactored.ser.gz')
    sentences = parser.raw_parse_sents(t.split('.'))
    # for line in sentences:
    # for sentence in line:
    # print("<------sentence----->")
    # print(sentence)


# sentence.draw()


if __name__ == '__main__':
    Sentences = list()
    words = list()
    document_Clustering()
# wiki_wiki = wikipediaapi.Wikipedia('ar')
# page_py = wiki_wiki.page('محمد_بن_سلمان_آل_سعود')
# links=page_py.links
# print (type(links))
# for l in links:
#   if "بوابة" not in l:
#   o=l.replace(" ", "_")
#   page_py2=wiki_wiki.page(o)
#   ANER(page_py2.text)
# POS_DB=DataBase_Connector("POS_Tag")
# print("Page - Title: %s" % page_py.title)
# Infile = page_py.text

# Load the document
# Infile = open('Input_Stroy.txt', encoding='utf-8')
# input_text =Infile.read()

# Normalize the document text
# Normilzed_text=Normalize_Text(input_text)

# Split the text into Stories, Populate the database and return the stories with titles in a dictionary object
# Stories_Split(Normilzed_text.split('.'))

# Para_Segmenter(Normilzed_text.split('.'))

# extract verbs from text in database
# Verbs_Extractor()
# Out_Tmp_file = open('tmp_out.txt', 'w', encoding='utf-8')
# ANER(Normilzed_text)
# Out_Tmp_file.write(Normilzed_text)

# of=open('tmp_result.txt','w',encoding='utf-8')
# of.write(result)
# os.system('java -jar FarasaSegmenterJar.jar -i tmp_result.txt -o tmp_result2.txt')

# Segmented_text=text_segmenter(Normilzed_text)
# N_ANER(Segmented_text)
# print("-------Segmented Text------->"+Segmented_text+"\n<----------------->")
# Paragraphs=Segmented_text.split('.')


# for sentences in Paragraphs:
#   new_results=re.split(',',sentences)
#   for s in new_results:
#      Sentences.append(s)

# Outfile = open('Final_results.txt', 'w', encoding='utf-8')
# tokenizer = RegexpTokenizer(r'\w+')
# s_word=stopwords.words("arabic")
# Out_file = open('N_NER_Out.txt', 'w', encoding='utf-8')
# nt_file = open('NTF.txt', 'w', encoding='utf-8')
# for sentence in Sentences:

# N_ANER(sentence, 1, DB,Out_file)
#     N_ANER(sentence,2,DB,Out_file)
#     N_ANER(sentence, 3, DB,Out_file)
# Outfile.write(sentence + "\n<----->\n")
# POS = make_tagger()
# Parsed_text=text_parser(sentence)
# tokens =tokenizer.tokenize(sentence)
# tagged_data=POS.tag(tokens)
# word_catagrizer(tagged_data,DB,nt_file )
# print("*")
#   #Outfile.write("\n<----->\n")

# nt_file.close()

# nt_file2=open('NTF.txt', 'r', encoding='utf-8')
# N_ANER(nt_file2.read(), 1, DB,Out_file)
# N_ANER(nt_file2.read(), 2, DB,Out_file)
# N_ANER(nt_file2.read(), 3, DB,Out_file)

# Out_file.close()


# sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good."""
# tokens = nltk.word_tokenize(sentence)
# print(tokens)
# tagged = nltk.pos_tag(tokens)
# print(tagged)