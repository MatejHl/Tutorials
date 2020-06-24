import os
import pandas

# sep='\t'

# ----- ner_dataset.csv :
ner_dataset = pandas.read_csv(os.path.join('_data_files', 'Data', 'ner_corpus_260.csv'), sep=',', header=[0], encoding = "ISO-8859-1")
ner_dataset = ner_dataset[['Word', 'Tag']]
ner_dataset.to_csv(os.path.join('_data_files', 'Data', 'ner_corpus_260.tsv'), sep='\t', index=False, header = False)
# -----