from allennlp_models.classification.dataset_readers import StanfordSentimentTreeBankDatasetReader
split = "test"
data_path = f"https://allennlp.s3.amazonaws.com/datasets/sst/{split}.txt"

clean_data_reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class")
instances = list(clean_data_reader.read(data_path))

text_lst = []
label_lst = []
for i in instances:
    text = ' '.join([str(token) for token in i.fields['tokens'].tokens])
    label = i.fields['label'].label
    text_lst.append(text)
    label_lst.append(label)
import pandas as pd
df = pd.DataFrame({'text': text_lst, "label": label_lst}, )
df.to_csv(f'{split}.csv', index=False, header=False)

