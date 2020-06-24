"""
Based on https://towardsdatascience.com/custom-named-entity-recognition-using-spacy-7140ebbb3718

Train data format: https://spacy.io/api/annotation#training
"""

# Training additional entity types using spaCy
from __future__ import unicode_literals, print_function
import pickle
import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding

from to_spaCy_format import tsv_to_json_format, to_spaCy_format

tsv_to_json_format("/_data_files/Data/ner_corpus_260.tsv",'/_data_files/Data/ner_corpus_260.json','abc')
to_spaCy_format(input_file='/_data_files/Data/ner_corpus_260.json', output_file='/_data_files/Data/ner_corpus_260.pkl')

# New entity labels
# Specify the new entity labels which you want to add here
LABEL = ['I-geo', 'B-geo', 'I-art', 'B-art', 'B-tim', 'B-nat', 'B-eve', 'O', 'I-per', 'I-tim', 'I-nat', 'I-eve', 'B-per', 'I-org', 'B-gpe', 'B-org', 'I-gpe']

# -------------------------
model = None
new_model_name = 'test_model'
output_dir = '_model_files//Trained Models'
n_iter = 30
# -------------------------

"""
geo = Geographical Entity
org = Organization
per = Person
gpe = Geopolitical Entity
tim = Time indicator
art = Artifact
eve = Event
nat = Natural Phenomenon
"""
# Loading training data 
with open ('_data_files/Data/ner_corpus_260.pkl', 'rb') as fp:
    TRAIN_DATA = pickle.load(fp)
    TRAIN_DATA = [data for data in TRAIN_DATA if data[0] != '' ]


# Setting up the pipeline and entity recognizer.
if model is not None:
    nlp = spacy.load(model)  # load existing spacy model
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank('sk')  # create blank Language class
    print("Created blank 'sk' model")

if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner)
else:
    ner = nlp.get_pipe('ner')


# Add new entity labels to entity recognizer
for i in LABEL:
    ner.add_label(i)

# Inititalizing optimizer
if model is None:
    optimizer = nlp.begin_training()
else:
    optimizer = nlp.entity.create_optimizer()


# Get names of other pipes to disable them during training to train # only NER and update the weights
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    for itn in range(n_iter):
        print('Iter:   {}'.format(itn))
        random.shuffle(TRAIN_DATA)
        losses = {}
        batches = minibatch(TRAIN_DATA, 
                            size=compounding(4., 32., 1.001))
        for batch in batches:
            texts, annotations = zip(*batch) 
            # Updating the weights
            try:
                nlp.update(texts, annotations, sgd=optimizer, 
                           drop=0.35, losses=losses)
            except Exception as e:
                print('OLD texts:')
                print(texts_old)
                print('OLD annotations:')
                print(annotations_old)
                print('ERROR texts:')
                print(texts)
                print('ERROR annotations:')
                print(annotations)
                raise e
            texts_old = texts
            annotations_old = annotations
        print('Losses', losses)

# Test the trained model
test_text = 'Gianni Infantino is the president of FIFA.'
doc = nlp(test_text)
print("Entities in '%s'" % test_text)
for ent in doc.ents:
    print(ent.label_, ent.text)

# Save model 
if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.meta['name'] = new_model_name  # rename model
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)

    # Test the saved model
    print("Loading from", output_dir)
    nlp2 = spacy.load(output_dir)
    doc2 = nlp2(test_text)
    for ent in doc2.ents:
        print(ent.label_, ent.text)

