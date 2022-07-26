import pickle
from tabnanny import check
import torch
import pandas as pd
import argparse
import os
from resource import check_valid_name

def read_df(output_dir, dataset_name, model_name):
    # texts
    with open(f'{output_dir}/{dataset_name}_noisy.pickle', 'rb') as file:
        noisy_data = pickle.load(file)
    noisy_data = dict(noisy_data)

    # corruption results
    with open(f'{output_dir}/{dataset_name}_{model_name}.pickle', 'rb') as file:
        examples = pickle.load(file)

    df = pd.DataFrame.from_dict( examples )
    df['countN'] = df['N_sets'].apply(lambda x: len(x[0]))
    df['text'] = [noisy_data[index][f'x'] for index in df['index']]
    df['noisy_txt'] = [noisy_data[index][f'x_{noise_type}'] for index, noise_type in zip(df['index'], df['noise_type'])]
    return df, noisy_data

def evaluate(
    output_dir,
    dataset_name, 
    model_name='bert-base-uncased-SST-2', 
    acc=True, 
    mode = None,
    countM_by_word_len_max_val = 0.8,
    countM_by_word_len_min_val = 0.3,
    print_acc_sim_by_noise=False,
    seq_max_len=None,
    no_consistent_pred=False
    ):
    """ 
        ## return 
        df:  for classifiers, samples whose clean texts are correctly predicted and satisfy the seq_max_len
         for LM for sentiment lexicon, we return all the samples
    """ 
    model_name = check_valid_name(model_name)
    df, noisy_data = read_df(output_dir, dataset_name, model_name)
    
    # for seq_max_len
    valid_index = []
    if seq_max_len is not None:
        for index, data in noisy_data:
            if len(data['x']) < seq_max_len:
                valid_index.append(index)
        print('Valid Length: ', len(valid_index))
        df = df[[True if i in valid_index else False for i in df['index'] ]]
    print('All noisy words: ', len(df))
    # for intact word corruption
    if mode == 'intact':
        df = df[df.countO== 1][df.wcr1 == 1]
        print('Intact Word Corruption: ', len(df))
    elif mode == "partial":
        df = df[df.countO>=1][df.countN>0]
        print('Partial Word Corruption: ', len(df))
        df['countM_by_O'] = [cm/co for cm, co in zip(df['countM'], df['countO'])]
        df['countM_by_O_round'] = [round(cm/co) for cm, co in zip(df['countM'], df['countO'])]
    elif mode == "all":
        pass
    
    # no compound words
    if dataset_name == "sentiment-lexicon":
        df = df.iloc[[True if len(lst)==1 else False for lst in df['text'] ]]

    _ = df.plot.scatter( x='countM', y='cos_sim', c='blue', alpha=0.3 )
    _ = df.plot.scatter( x='countM_by_word_len', y='cos_sim', c='blue', alpha=0.3 )
    _ = df.plot.scatter( x='dispersion_score', y='cos_sim', c='blue', alpha=0.3 )
    
    max_val = countM_by_word_len_max_val
    min_val = countM_by_word_len_min_val
    if acc:
        if dataset_name != "sentiment-lexicon":
            df['conf'] = df.apply(lambda example:  torch.nn.functional.softmax(example.logit, dim=0).tolist()[example.label], axis=1)
            df['conf_clean'] = df.apply(lambda example: torch.nn.functional.softmax(example.logit_clean, dim=0).tolist()[example.label], axis=1)
            df['pred'] = df.logit.map(lambda logit: torch.argmax(logit, dim=0).item())
            df['pred_clean'] =df.logit_clean.map(lambda logit: torch.argmax(logit, dim=0).item())
            df['correct'] = df.pred == df.label
            df['correct_clean'] = df.pred_clean == df.label
            
            print('Clean Accuracy:', df['correct_clean'].mean(), f" for {len(df)} examples.")
            print('only keep rows of which clean text is correctly predicted in df' )
            df = df[df['correct_clean']==True]
            
            if no_consistent_pred:
                #  not include examples consistenct predictoins
                consistent_pred_index = []
                for i, group in df[['index', 'correct']].groupby(['index']):
                    if len(group.correct.unique())==1:
                        consistent_pred_index.append(list(group['index'])[0])
                df = df[~df['index'].isin(consistent_pred_index)]
            df = df.reset_index()
            
            print('Misclassification Rate:')
            
            # <0.3, 0.3 - 0.5, >0.5
            select = df[df["countM_by_word_len"]<min_val]
            print(
                f'<{min_val} ({len(select)}):', 
                1- select['correct'].mean(),
                ' (cos_sim:', {select['cos_sim'].mean()}, ')'
                )
            
            select = df[df["countM_by_word_len"]>=min_val][df["countM_by_word_len"]<=max_val]
            print(
                f'{min_val}-{max_val} ({len(select)}):', 
                1- select['correct'].mean(),
                ' (cos_sim:', {select['cos_sim'].mean()}, ')'
                )
            select = df[df["countM_by_word_len"]>max_val]
            print(
                f'>{max_val} ({len(select)}):', 
                1- select['correct'].mean(),
                ' (cos_sim:', {select['cos_sim'].mean()}, ')'
                )
            # print(
            #     'for samples with min countM:', 
            #     1- df.iloc[df.groupby("index")["countM"].idxmin()]['correct'].mean(),
            #     ' (cos_sim:', {df.iloc[df.groupby("index")["countM"].idxmin()]['cos_sim'].mean()}, ')'
            #     )
            # print(
            #     'for all samples:', 
            #     1-df['correct'].mean(),
            #     ' (cos_sim:', df['cos_sim'].mean(), ')')
            # print(
            #     'for samples with max countM:', 
            #     1-df.iloc[df.groupby("index")["countM"].idxmax()]['correct'].mean(),
            #     ' (cos_sim:',{df.iloc[df.groupby("index")["countM"].transform(lambda x:x.max()).idx()]['cos_sim'].mean()},')')
        else:
            print('Cos Sim:')
            # <0.3, 0.3 - 0.5, >0.5
            select = df[df["countM_by_word_len"]<min_val]
            print(
                f'<{min_val} ({len(select)}):', 
                select['cos_sim'].mean()
                )
            select = df[df["countM_by_word_len"]>=min_val][df["countM_by_word_len"]<=max_val]
            print(
                f'{min_val}-{max_val} ({len(select)}):', 
                 {select['cos_sim'].mean()}
                )
            select = df[df["countM_by_word_len"]>max_val]
            print(
                f'>{max_val}: ({len(select)})', 
                {select['cos_sim'].mean()}
                )
        wcs1_round_dist = {}
        for score in sorted(df['countM_round'].unique()):
            examples = df[(df['countM_round'] == score)]

            print(f'Metric for {score}')
            
            if dataset_name != "sentiment-lexicon":
                print(  1-examples.correct.mean())
                print( 'conf:', examples.conf.mean())

            wcs1_round_dist[score] =  len(examples)

        print('Score Distribution: ', wcs1_round_dist)
        
        if print_acc_sim_by_noise:          
            noiser_names = [k[2:] for k in noisy_data[0].keys() if k.startswith('x_')]
            for noise_type in noiser_names:
                if dataset_name != "sentiment-lexicon":
                    print(f'Misclassification Rate for {noise_type}: ', 1-df[df.noise_type==noise_type].correct.mean())
                print(f'avg sim for {noise_type}: ', df[df.noise_type==noise_type].cos_sim.mean())

    else: # evaluating word corruption
        
        noiser_names = [k[2:] for k in noisy_data[0].keys() if k.startswith('x_')]
        print(noiser_names)
        for noise_type in noiser_names:
            print(f'avg countM for {noise_type}: ', df[df.noise_type==noise_type]['countM'].mean())
            # print(f'avg countO for {noise_type}: ', df[df.noise_type==noise_type]['countO'].mean())
            print(f'avg countM_by_word_len for {noise_type}: ', df[df.noise_type==noise_type]['countM_by_word_len'].mean())
            # print(f'avg wcr1 for {noise_type}: ', df[df.noise_type==noise_type]['wcr1'].mean())
            # print(f'avg wcr2 for {noise_type}: ', df[df.noise_type==noise_type]['wcr2'].mean())

    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generating...")
    parser.add_argument(
        "--dataset_name", 
        default="yelp", 
        type=str,
        help=''' Valid name in RegExp: (sentiment-lexicon|yelp|sst2|ag-news) ''')
    parser.add_argument(
        "--model_name", 
        default="roberta-base-yelp", 
        type=str,
        help=''' Valid template for `model_name` in RegExp: (bert|albert|roberta)-*(yelp|sst2|ag-news)* ''')
    args = parser.parse_args()
    dataset_name = args.dataset_name
    model_name=args.model_name
    output_dir = "outputs_local_mixed_noise"
    seq_max_len=None

    # Evaluating PLMs
    df = evaluate(
        output_dir = output_dir,
        dataset_name = dataset_name, 
        model_name=model_name, 
        acc=True, 
        mode="intact", 
        seq_max_len=seq_max_len, 
        countM_by_word_len_min_val = 0.3,
        countM_by_word_len_max_val=0.8,
        no_consistent_pred=False,
        print_acc_sim_by_noise=True)
