from transformers import RobertaTokenizerFast, TrainingArguments, Trainer, RobertaForSequenceClassification, DataCollatorWithPadding, EarlyStoppingCallback, logging
from datasets import load_dataset, Dataset
from datasets.dataset_dict import DatasetDict
from argparse import ArgumentParser
from scipy.stats import entropy
from collections import Counter
import random, torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import csv


tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

def get_dist(x, add=0):
    # Get distibution from votes
    counts = Counter(x)
    for x in range(0,5):
        if x not in counts:
            counts[x] = add
        else:
            counts[x] += add
    counts = [counts[key] for key in sorted(counts.keys(), reverse=False)]
    return (np.asarray(counts)/sum(counts)).tolist()

def setup_argparse() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("-l", "--label", type=str, required=True, help="The label to train on in {'sentiment', 'respect', 'insult', 'humiliate', 'status', 'dehumanize', 'violence', 'genocide', 'attack_defend'}")
    parser.add_argument("-r", "--num-runs", type=int, default=3, help="Number of training rounds")
    parser.add_argument("-n", "--num-egs", type=int, default=400, help="Top n egs (by entropy difference) will get a `new' annotation")
    parser.add_argument("-t", "--training-dir", type=str, default='../data/measuring-hate-speech/', help="Path to Rasch train and test data")
    parser.add_argument("-a", "--num-annots", type=str, help="Numbre of annots (or comments) to use (if not using the whole set)")
    parser.add_argument("-s", "--new-subset", type=bool, default=True, help="make a new train subset or just use the existing saved one")
    parser.add_argument("-c", "--counter", type=str, help="counter to add to filename")
    parser.add_argument("-b", "--base", type=str, help="The experiment type: _random_pt_1_ or _random_pt_2_")
    parser.add_argument("-z", "--zero-human", type=bool, default=False, help="enforce 0 human entropy?")
    parser.add_argument("-v", "--vote", type=bool, default=False, help="should train on majority vote")
    parser.add_argument("-bp", "--budget-annots-per", type=int,  help="Number of annotations per commentfor the ent predictor budget")
    return parser




def predict(model, df, attr):
    preds = {'comment_id':[], 'pred':[], 'vote':[], 'probs':[], 'human_probs':[], 'human_entropy':[], 'model_entropy':[], 'kl_div':[]}

    probs = []

    text = list(df['text'])


    # batch predict
    with torch.no_grad():
        for i in tqdm(range(0, len(text), 100)):
            t= text[i:i + 100]
            model_inputs = tokenizer(
                t, return_tensors='pt', truncation=True, padding=True,
                add_special_tokens=True).to(device)
            outputs = model(**model_inputs)
            probs.extend(torch.nn.functional.softmax(outputs.logits, dim=1).tolist())


    # calculate entropies and whatnot
    for comment_id, text, vote, human_probs, prob in zip(df['comment_id'], df['text'], 
                                                        df[attr], df[attr+"_vote_dist"], probs):
        preds['comment_id'].append(comment_id)
        preds['probs'].append(prob)
        preds['pred'].append(prob.index(max(prob)))
        preds['vote'].append(vote)
        preds['human_probs'].append(human_probs.tolist())
        preds['human_entropy'].append(entropy(human_probs))
        preds['model_entropy'].append(entropy(prob))
        preds['kl_div'].append(entropy(human_probs, qk=prob))

    return pd.DataFrame.from_dict(preds)

def agg_dataset(df, attrs):
    grp = df.groupby(['comment_id','text'], as_index=False)
    agg = grp.agg(lambda x: random.choice(pd.Series.mode(x)))
    agg = agg.drop([col for col in agg.columns if col not in ['comment_id', 'text']+attrs], axis=1)
    for attr in attrs:
        agg[attr+"_vote_dist"] = agg['comment_id'].apply(lambda x: get_dist(df[df['comment_id']==x][attr].tolist()))
    return Dataset.from_pandas(agg, preserve_index=False)

def sample_df(df, holdout_df, n, single_annots):
    if single_annots:
        holdout = list(holdout_df.comment_id)
        df = df[~df.comment_id.isin(holdout)]

        return Dataset.from_pandas(df.sample(n), preserve_index=False)

        
    else:

        cids = list(set(list(df.comment_id)))
        hold_out = list(set(list(holdout_df.comment_id)))
        remainder = [x for x in cids if x not in hold_out]

        hold_out = random.sample(remainder, n)

        df = df[df['comment_id'].isin(hold_out)]


        return Dataset.from_pandas(df, preserve_index=False)


if __name__ == '__main__':
    args = setup_argparse().parse_args()


    attr = args.label
    attrs = ['sentiment', 'respect', 'insult', 'humiliate', 'status', 'dehumanize', 'violence', 'genocide', 'attack_defend']


    num_annots = int(args.num_annots)

    new_subset = args.new_subset
    zero_human = args.zero_human
    budget_annots_per = args.budget_annots_per


    base = args.base 

    if args.vote:
        base = "_maj_vote_" + base

    if zero_human:
        base = "_0_human_" + base

    if budget_annots_per is not None:
        base += str(budget_annots_per)+"_"
    
    if num_annots:
        base+= args.counter+"_"


    path = "pred_"+attr+"/"+attr+"_annotation_nums"+base+".csv"

    print(path)

    base = args.base 

    if args.vote:
        base = "_maj_vote_" + base

    if zero_human:
        base = "_0_human_" + base

    if budget_annots_per is not None:
        base += str(budget_annots_per)+"_"
    
    if num_annots:
        base+=str(num_annots)+"_"+ args.counter+"_"

    if num_annots and num_annots == 10:
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["comments","annots"])

    test = load_dataset('csv', 'data/', data_files=[args.training_dir+'test_subset.csv'] )

    if (not num_annots) or new_subset:
        train = load_dataset('csv', 'data/', data_files=[args.training_dir+'train_subset.csv'] )
    else:
        train = load_dataset('csv', 'data/', data_files=[args.training_dir+'train_subset_'+str(num_annots)+'.csv'] )

    if budget_annots_per is not None:
        train_df = train['train'].to_pandas()
        train['train'] = Dataset.from_pandas(train_df.groupby('comment_id').apply(lambda x: x.sample(budget_annots_per)).reset_index(drop=True))
    
    train['test'] = test['train'].remove_columns([col for col in test['train'].column_names if col not in ['comment_id', 'text', attr]])
    train['train'] = train['train'].remove_columns([col for col in train['train'].column_names if col not in ['comment_id', 'text', attr]])
    
    if zero_human:
        train_agg = agg_dataset(train['train'].to_pandas(), [attr])
        #import pdb; pdb.set_trace()
        zero_human_ids = [comment_id for comment_id, dist in zip(train_agg.to_pandas().comment_id, train_agg.to_pandas()[attr+"_vote_dist"]) if entropy(dist)==0]
        #zero_human_ids = list(train_agg[train_agg.human_entropy == 0].comment_id)

        df_train = train['train'].to_pandas()
        train['train'] = Dataset.from_pandas(df_train[df_train.comment_id.isin(zero_human_ids)], preserve_index=False)


    train_heldout = DatasetDict(train.copy())
    train_heldout['train'] = agg_dataset(pd.read_csv(args.training_dir+'train_holdout.csv'), [attr])

    if new_subset and "_random_pt_1_" in base:
        train['train'] = sample_df(train['train'].to_pandas(), train_heldout['train'].to_pandas(), num_annots, False)
    elif new_subset:
        train['train'] = sample_df(train['train'].to_pandas(), train_heldout['train'].to_pandas(), num_annots, True)

    # import pdb; pdb.set_trace()

    tokenized = train.map(preprocess_function, batched=True)

    train_agg = DatasetDict(train.copy())
    train_agg['train'] = agg_dataset(train_agg['train'].to_pandas(), [attr])
    train_agg['test'] = agg_dataset(train_agg['test'].to_pandas(), [attr])

    if args.vote:
        train = train_agg
        tokenized = train_agg.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    logging.set_verbosity_info()
    

    model = RobertaForSequenceClassification.from_pretrained('roberta-base',num_labels=5).to(device)



    training_args = TrainingArguments(
        output_dir="./runs/roberta_"+attr,
        evaluation_strategy="steps",
        eval_steps = 100,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        save_total_limit = 1,
        load_best_model_at_end = True
    )


    to_remove = ["comment_id"]
    train_dataset = tokenized["train"].rename_column(attr, 'label').remove_columns(to_remove)
    eval_dataset = tokenized["test"].rename_column(attr, 'label').remove_columns(to_remove)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )



    prev_kl_div = None
    tolerance = 0
    max_tolerance = 1
    should_cont = True
    tot = 0

    while should_cont:


        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        trainer.train()
        df = predict(model, train_heldout['train'].to_pandas(), attr)
        kl_div = np.mean(df['kl_div'])
        

        if prev_kl_div:
            if prev_kl_div < kl_div:
                if tolerance == max_tolerance: 
                    should_cont = False
                tolerance += 1
            elif prev_kl_div >= kl_div:
                tolerance = 0
                prev_kl_div = kl_div
        else:
            prev_kl_div = kl_div
        tot += 1
        

        print(kl_div, tolerance)

        if tot > 50:
            should_cont = False

        



    


    df = predict(model, train_agg['test'].to_pandas(), attr)
    df.to_csv("pred__"+attr+"/"+attr+"_test_predictions"+base+".csv")


    num_annots = len(train['train'].to_pandas())
    num_comments = len(set(train['train'].to_pandas().comment_id))

    
    
    with open(path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([num_comments, num_annots])

