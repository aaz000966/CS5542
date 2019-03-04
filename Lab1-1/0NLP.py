#This code was inspired direclty by the ICPs given in the class
#i'll be using
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#I tried to use spacy, but I faced some issues
import spacy
from nltk.tokenize import word_tokenize

#to count how many words and lines in the file, helped during the development
Words=0
Lines=0

#linking the dataset from the project folder and read the file
file= open("SBU_captioned_photo_dataset_captions.txt","r")

data=file.read()

#simple iteration to count the content
with file as f1:
    for line in f1:
        Lines += 1
        for word in line.split():
            Words += 1

print(Words)
print(Lines)

#to creat a newfile with the output required
TokenF=open("Tokenized.txt", "w")

#tokenizing the edetatset using word_tokenize(str), a predefined nltk method
TokenF.write(str(word_tokenize(data)))


#creating a new output file for lemma operation
LemmaF=open("Lemmatized.txt", "w")
lemma = WordNetLemmatizer()


# file= open("SBU_captioned_photo_dataset_captions.txt","r")
# #for the sake of demonstration, I'm only writng the words with different architecture
# with file as f2:
#     for line in f2:
#         for word in line.split():
#             if word != lemma.lemmatize(word):
#                 LemmaF.write(str(word + " : " + lemma.lemmatize(word) + "\n"))


print("done")


