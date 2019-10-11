import sys
sys.path.insert(0,'/home/davide/workspace/tesi/cider')
from pydataformat.loadData import LoadData
from pyciderevalcap.eval import CIDErEvalCap as ciderEval
from nltk.translate.bleu_score import sentence_bleu
import ast

def blue_4(references_dict ,to_test_dict):
    references = []

    references_dict = references_dict.values()
    for l in references_dict:
        tmp = []
        for caption in l:
            tmp.append( caption.values()[0].split())
        references.append(tmp)
    to_test_dict = [el.values()[1] for el in to_test_dict]
    to_test = [el.split(" ") for el in to_test_dict]
    return sentence_bleu( references, to_test)

def CIDEr (references_dict, toTest_dict):

    df_mode = 'corpus'
    scorer = ciderEval(references_dict, toTest_dict, df_mode)
    return scorer.evaluate()

def extract_reference(file):
    dict = {}
    current_img="53043785_c468d6f931"
    current_captions=[]

    f=open(file, "r")
    line = " "
    read = False

    while (not read):
        #separate image name (imga variable) and caption (line[1])
        line = f.readline()
        line = line.split("\t")
        img = line[0].split(".")
        img = img[0]

        #if this is a new image update current variable and put current dict in wall dict
        if current_img!=img:
            dict [current_img] = current_captions
            current_img=img
            current_captions = []

        #otherwise append to current captions list
        if line==['']:
            read = True
        else:
            current_captions.append({'caption':line[1]})
    return dict


a = ' A black dog in a blue shirt is a in the in the .'
a = a.split(" ")
print(a)
b =  [['A black boy in orange and white trucks on playing in the sand '], ['A black boy is sitting in the sand '], ['A boy plays in the sand '], ['A young child wearing a striped bathing suit sits on the sand '], ['The boy sits in the sand with no shirt ']]
bb = []
for i in b:
    bb.append(i[0].split(" "))
print(bb)
print(sentence_bleu(bb,a))