from transformers import RobertaTokenizerFast, TrainingArguments, Trainer, RobertaForSequenceClassification, DataCollatorWithPadding, EarlyStoppingCallback, logging
from datasets import load_dataset, Dataset
from datasets.dataset_dict import DatasetDict
from argparse import ArgumentParser
from scipy.stats import entropy
from collections import Counter
import random, torch, evaluate, math
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds

print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.device_count())

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

NUM_EGS=10
MAX = 50


## For debug purposes
DL = False
COMP = False
JUST_ENT = False
SKIP_TRAINING = False
##


mse_metric = evaluate.load("mse")

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', force_download=DL)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

def agg_dataset(df, attrs):
    cols = [col for col in ['comment_id', 'text', 'annot_id']+attrs if col in df.columns]

    grp = df[cols].groupby(['comment_id','text'], as_index=False)
    agg = grp.agg(lambda x: random.choice(pd.Series.mode(x)))
    agg = agg.drop([col for col in agg.columns if col not in cols], axis=1)
    for attr in attrs:
        agg[attr+"_vote_dist"] = agg['comment_id'].apply(lambda x: get_dist(df[df['comment_id']==x][attr].tolist()))
        agg[attr+"_entropy"] = agg[attr+"_vote_dist"].apply(lambda x: entropy(x))
    return Dataset.from_pandas(agg, preserve_index=False)

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
    parser.add_argument("-t", "--training-dir", type=str, default='../data/measuring-hate-speech/', help="Path to Rasch train and test data")
    parser.add_argument("-m", "--sampling-method", type=str, default='abs_ent_diff', help="ent_diff, abs_ent_diff, just_human, or just_model")
    parser.add_argument("-c", "--counter", type=str, help="counter to add to filename to differentiate runs")
    parser.add_argument("-e", "--ent-pred", type=str, help="entropy prediction method {exact, offline, online}")
    parser.add_argument("-a", "--add-method", type=str, default='all', help="method for adding new annotations {all, single_new, single_any} get all annotations per new comment or get single annotations on only unseen comments or on any comment")
    parser.add_argument("-z", "--zero-human", type=bool, default=False, help="enforce 0 human entropy?")
    parser.add_argument("-v", "--vote", type=bool, default=False, help="should train on majority vote")
    parser.add_argument("-bc", "--budget-comments", type=int,  help="Total number of comments for ent predictor budget (-1 for all)")
    parser.add_argument("-ba", "--budget-annots", type=int,  help="Total number of annotations for the ent predictor budget")
    parser.add_argument("-bp", "--budget-annots-per", type=int,  help="Number of annotations per commentfor the ent predictor budget")
    parser.add_argument("-ce", "--control-ent", type=bool, default=False, help="control H(f_theta(x)) to match f_ent during inference")
    parser.add_argument("-se", "--sample-entropy", type=bool, default=False, help="sample from human dist w/ entropy controlted to match f_ent during training")
    parser.add_argument("-set", "--sample-entropy-true", type=bool, default=False, help="sample from human dist w/ entropy controlted to match human entropy during training")
    return parser


def normalize(p, beta):
    p_new = [np.log(p_i) for p_i in p]
    p_beta = [math.e**(p_i*beta) for p_i in p]
    z = sum(p_beta)
    for i in range(len(p_new)):
        if p_beta[i] == 0:
            p_new[i] = 0
        else:
            p_new[i] = math.e**(np.log(p_beta[i])-np.log(z))

        if type(p_new[i]) == np.ndarray:
            p_new[i] = p_new[i][0]

    if sum(p_new) >.99:
        return p_new
    else:
        print("normalization failed", p_new)
        return p

def smooth(x, l=.0001):
    x = [x_i+l for x_i in x]
    z = sum(x)
    return [math.e**(np.log(x_i)-np.log(z)) for x_i in x]

def adjust_entropy(p,t):
    p_sm = smooth(p) 
    def objective(beta):
        return abs(t - entropy(normalize(p_sm, beta)))
    beta = minimize(objective, 1)['x'][0]
    return normalize(p_sm, beta)

def sample_cids(df, n, sampling_method=None, zero_human=False, used=None):


    pos = 0
    num_diff = 0
    pos_rean = 0

    if zero_human:
        df = df[df.human_entropy == 0]

    if sampling_method is  None:
        if n == -1:
            hold_out = df.comment_id.to_list()
        else:
            hold_out = random.sample(df.comment_id.to_list(), n)
    else:

        if sampling_method == "ent_diff":
            df['sort'] = df.apply(lambda x: x['human_entropy'] - x['model_entropy'], axis = 1)
        elif sampling_method == "just_human":
            df['sort'] = df['human_entropy']
        elif sampling_method == "just_model":
            df['sort'] = df['model_entropy']
        else:
            df['sort'] = df.apply(lambda x: abs(x['human_entropy'] - x['model_entropy']), axis = 1)
            df['diff'] = df.apply(lambda x: x['human_entropy'] - x['model_entropy'], axis = 1)

            


        
        hold_out = []
        for cid in df.sort_values(by='sort', ascending=False).comment_id:
            if len(hold_out) == n:
                break
            if cid not in hold_out:
                hold_out.append(cid)
                if 'diff' in df.columns and list(df[df.comment_id == cid]['diff'])[0] > 0:
                    pos += 1
                    if used is not None and cid in used:
                        pos_rean += 1



        if COMP:
            df['sort'] = df['model_entropy']
            hold_out_2 = []
            for cid in df.sort_values(by='sort', ascending=False).comment_id:
                if len(hold_out_2) == n:
                    break
                if cid not in hold_out_2:
                    hold_out_2.append(cid)
                    if cid not in hold_out:
                        num_diff += 1


    

    
    return {cid: df[df['comment_id']==cid].annot_id.to_list() for cid in hold_out}, pos, num_diff, pos_rean


def predict(model, df, attr, batch, just_pred_ent = False, ent_predictor = False, ents=False):
    # prediciting either the label or human entropy based on the text
    # can either return just the predicted (human or model) entropy or a full prediction (w/ other stats)

    if ent_predictor:
        preds = {'comment_id':[], 'pred':[], 'human_entropy':[]}
    else:
        preds = {'comment_id':[], 'pred':[], 'vote':[], 'probs':[], 'human_probs':[], 'human_entropy':[], 'model_entropy':[], 'kl_div':[]}


    probs = []

    text = list(df['text'])


    #batch predict
    with torch.no_grad():
        for i in tqdm(range(0, len(text), batch)):
            t= text[i:i + batch]
            model_inputs = tokenizer(
                t, return_tensors='pt', truncation=True, padding=True,
                add_special_tokens=True).to(device)
            outputs = model(**model_inputs)
            if ent_predictor:
                probs.extend([float(x[0]) for x in outputs.logits])
            else:
                probs.extend(torch.nn.functional.softmax(outputs.logits, dim=1).tolist())

    if ents:
        
        entropies = [entropy(x) for x in list(df[attr+"_vote_dist"])]
        for i in range(len(probs)):
            p = probs[i]
            target = entropies[i]

            probs[i] = adjust_entropy(p,target)




    if just_pred_ent and ent_predictor:
        return {comment_id:prob for comment_id, prob in zip(df['comment_id'], probs) }
    elif just_pred_ent:
        return {comment_id:entropy(prob) for comment_id, prob in zip(df['comment_id'], probs) }
    elif ent_predictor:
        for comment_id, text, vote, human_probs, prob in zip(df['comment_id'], df['text'], 
                                                        df[attr], df[attr+"_vote_dist"], probs):
            preds['comment_id'].append(comment_id)
            preds['pred'].append(prob)
            preds['human_entropy'].append(entropy(human_probs))
    else:
        # not the entropy predictor and returning everything
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

def compute_metrics(eval_preds):
    # used by the entropy predictor
    predictions, labels = eval_preds
    return mse_metric.compute(predictions=predictions, references=labels)
    
def df_minus(orig_df, sub_df):
    return Dataset.from_pandas(orig_df[~orig_df['comment_id'].isin(sub_df['comment_id'])], preserve_index=False)

def sample_and_split_df(df):
    # pick one annotation out from each comment
    # return Dataset with only the sampled annotations and another Dataset w/ the remainder
    # I'm confident there's a better way to do this lol
    df = df.rename_axis('old_index').reset_index()
    
    

    
    sampled = df.groupby('comment_id').agg(lambda x: x.sample(1))
    remainder = df[~df.old_index.isin(sampled.old_index.to_list())]


    return Dataset.from_pandas(sampled.drop('old_index', axis=1), preserve_index=False), Dataset.from_pandas(remainder.drop('old_index', axis=1), preserve_index=False)


def sample_and_split_cids(ids):
    # pick one annotation out from each comment
    # return id dict with only the sampled annotations and another id dict w/ the remainder
    singles = {}
    remainders = {}


    for cid in ids.keys():
        annot_ids = ids[cid]
        random.shuffle(annot_ids)
        singles[cid] = [annot_ids[0]]
        remainders[cid] = annot_ids[1:]

    return singles, remainders

def combine_ids(d1, d2):
    d3 = d2.copy()
    for k,v in d1.items():
        if k in d3:
            for x in v:
                if x not in d3[k]:
                    d3[k].append(x)
        else:
            d3[k] = v
    return d3

def cid_filter(cid_dict, cid, annot_id):
    return cid in cid_dict and annot_id in cid_dict[cid]

def update_synth(synthetic_egs, train_df, cids, attr, ents, n=10):
    synthetic_egs = synthetic_egs[~synthetic_egs.comment_id.isin(cids)] #remove old synthetic egs if we've gotten a new real label

    train_agg = agg_dataset(train_df, [attr]).to_pandas()

    for cid in cids:
        count = len(train_df[train_df.comment_id == cid])
        

        human_dist = train_agg[train_agg.comment_id == cid][attr+"_vote_dist"].tolist()[0]
        human_dist = adjust_entropy(human_dist,ents[cid])

        labels = np.arange(len(human_dist))

        

        eg = train_df[train_df.comment_id == cid].to_dict(orient='records')[0]
        for _ in range(n-count):

            eg[attr] = np.random.choice(labels, p=human_dist) # might need to change these "attr"'s to "label"

            synthetic_egs = synthetic_egs.append(eg, ignore_index = True)


    return synthetic_egs

def main():
    args = setup_argparse().parse_args()

    comment_nums = [10,20,40,80,160,320,640,1280]
    annot_nums = {}

    tot_pos = {}
    tot_diff = {}
    tot_pos_rean = {}
    num_pos=0
    num_diff = 0
    num_pos_rean = 0



    attr = args.label
    if attr == "toxicity":
        attrs = ['toxicity']
        BATCH = 8
        predict_batch=50
    elif attr == "toxicity_3":
        attrs = ['toxicity_3']
        BATCH=16
        predict_batch=50
    else:
        predict_batch=100
        BATCH=16
        attrs = ['sentiment', 'respect', 'insult', 'humiliate', 'status', 'dehumanize', 'violence', 'genocide', 'attack_defend']


    sampling_method = args.sampling_method
    ent_method = args.ent_pred
    add_method = args.add_method
    zero_human = args.zero_human

    sample_corrected_entropy = args.sample_entropy
    sample_corrected_true_entropy = args.sample_entropy_true

    budget_annots = args.budget_annots
    budget_annots_per = args.budget_annots_per
    budget_comments = args.budget_comments

    if ent_method == "offline":
        ent_model = RobertaForSequenceClassification.from_pretrained('runs/roberta_entropy_'+attr, local_files_only=True).to(device)
    else:
        ent_model = None


    if ent_method is not None:
        base = sampling_method+"_"+ent_method+"_"+ args.counter+"_"
    else:
        base = sampling_method+"_"+ args.counter+"_"

    if args.vote:
        base = "maj_vote_" + base
    if zero_human:
        base = "zero_human_" + base
    if add_method != "all":
        base = add_method+"_"+base
    if args.control_ent:
        
        base = "ent_human_control_"+base

    if sample_corrected_entropy:
        base = "sample_corrected_entropy_" + base
    elif sample_corrected_true_entropy:
        base = "sample_corrected_true_entropy" + base

    if ent_method == "initial_budget":
        
        budget_name = "_".join([str(x) for x in [budget_comments, budget_annots_per, budget_annots] if x is not None])
        base =  budget_name+ "_"+base
    
    if budget_annots_per is not None:
        base = str(budget_annots_per)+"_"+base

    print(attr,base)

    addition = ""

    if "3" in attr:
        addition = "_3"

    test = load_dataset('csv', 'data/', data_files=[args.training_dir+'test_subset'+addition+'.csv'] )
    train = load_dataset('csv', 'data/', data_files=[args.training_dir+'train_subset'+addition+'.csv'] )
    train['test'] = test['train'].remove_columns([col for col in test['train'].column_names if col not in ['comment_id', 'text', attr]])
    train['train'] = train['train'].remove_columns([col for col in train['train'].column_names if col not in ['comment_id', 'text', attr]])
    
    train_df = train['train'].to_pandas()
    if budget_annots_per is not None:
        train_df = train_df.groupby('comment_id').apply(lambda x: x.sample(budget_annots_per)).reset_index(drop=True)

    train_df['annot_id'] = train_df.groupby(['comment_id']).cumcount()
    train['train'] = Dataset.from_pandas(train_df, preserve_index=False)

    
    tokenized = train.map(preprocess_function, batched=True)

    train_heldout = DatasetDict(train.copy())
    train_heldout['train'] = agg_dataset(pd.read_csv(args.training_dir+'train_holdout'+addition+'.csv'), [attr])

    train_heldout_df = train_heldout['train'].to_pandas()
    train_heldout_df['annot_id'] = train_heldout_df.groupby(['comment_id']).cumcount()
    train_heldout['train'] = Dataset.from_pandas(train_heldout_df, preserve_index=False)

    train_heldout_or_used_cids = {cid:train_heldout_df[train_heldout_df['comment_id']==cid].annot_id.to_list() for cid in train_heldout['train']['comment_id']}


    train_heldout_tokenized = train_heldout.map(preprocess_function, batched=True)





    train_agg = DatasetDict(train.copy())
    train_agg['train'] = agg_dataset(train_agg['train'].to_pandas(), [attr])
    train_agg['test'] = agg_dataset(train_agg['test'].to_pandas(), [attr])
    

    train_agg_tokenized = train_agg.map(preprocess_function, batched=True)

    if args.vote:
        train = train_agg
        tokenized = train_agg_tokenized


    if ent_method == "offline":
        human_dict = predict(ent_model, tokenized["train"].to_pandas(), attr, predict_batch, just_pred_ent=True, ent_predictor=True)
    elif ent_method == "exact":
        human_dict = {comment_id: entropy(dist) for comment_id, dist in zip(train_agg["train"].to_pandas()['comment_id'], train_agg["train"].to_pandas()[attr+"_vote_dist"])}
    else:
        human_dict = {comment_id:None for comment_id in train_agg["train"].to_pandas()['comment_id']}

    if sample_corrected_true_entropy:
        true_human_dict = {comment_id: entropy(dist) for comment_id, dist in zip(train_agg["train"].to_pandas()['comment_id'], train_agg["train"].to_pandas()[attr+"_vote_dist"])}
    


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = RobertaForSequenceClassification.from_pretrained('roberta-base',num_labels=5, force_download=DL).to(device)



    training_args = TrainingArguments(
        output_dir="./runs/roberta_"+attr,
        evaluation_strategy="steps",
        eval_steps = 100,
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        num_train_epochs=1,
        weight_decay=0.01,
        save_total_limit = 1,
        load_best_model_at_end = True
    )

    if ent_method == "initial_budget":
        ent_path =  "./runs/roberta_entropy_"+attr+"_budget"
    else:
        ent_path = "./runs/roberta_entropy_"+attr+"_online"

    ent_training_args = TrainingArguments(
        output_dir=ent_path,
        evaluation_strategy="steps",
        eval_steps = 100,
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        num_train_epochs=20,
        weight_decay=0.01,
        save_total_limit = 1,
        load_best_model_at_end = True
    )

    to_remove = ["comment_id", 'annot_id']

    if ent_method == "initial_budget":

        initial_budget = None

        if budget_comments is not None:
            initial_budget = budget_comments
        elif budget_annots is not None and budget_annots_per is not None:
            initial_budget = budget_annots/budget_annots_per
            assert(int(initial_budget)==initial_budget)
            initial_budget = int(initial_budget)
        else:
            raise Exception("Not sure how to handle budget spec")





        # pick the dataset for this

        train_unused = tokenized["train"].to_pandas()

        train_unused =  train_unused[train_unused.apply(lambda row: not cid_filter(train_heldout_or_used_cids, row['comment_id'], row['annot_id']), axis=1)]

        if initial_budget == -1:
            base = base.replace("-1", str(len(train_unused)))
            budget_name = budget_name.replace("-1", str(len(train_unused)))
            print(base)
        
        ent_dataset_ids, _, _, _ = sample_cids(train_unused, initial_budget)


        # use ent dataset to train
        ent_model = RobertaForSequenceClassification.from_pretrained('roberta-base',num_labels=1, force_download=DL).to(device)


        ent_train_dataset_agg = DatasetDict(train_agg_tokenized.copy())
        df = train_agg_tokenized["train"].to_pandas()

        ent_train_dataset_agg['train'] = Dataset.from_pandas(df[df['comment_id'].isin(ent_dataset_ids.keys())], preserve_index=False).rename_column(attr+"_entropy", 'label')


        cols_not_needed = [x for x in ent_train_dataset_agg['train'].column_names if x not in ['label', 'attention_mask', 'input_ids']]
        ent_train_dataset_agg = ent_train_dataset_agg['train'].remove_columns(cols_not_needed)


        cols_not_needed = [x for x in train_heldout_tokenized['train'].column_names if x not in ['label', 'attention_mask', attr+"_entropy", 'input_ids']]
        ent_eval_dataset_agg = train_heldout_tokenized["train"].rename_column(attr+"_entropy", 'label').remove_columns(cols_not_needed)

        ent_trainer = Trainer(
            model=ent_model,
            args=ent_training_args,
            train_dataset=ent_train_dataset_agg,
            eval_dataset=ent_eval_dataset_agg,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics = compute_metrics,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
        )

        if not SKIP_TRAINING:
            ent_trainer.train()

        # eval the ent predictor

        df = predict(ent_model, train_agg['test'].to_pandas(), attr, predict_batch, ent_predictor=True)
        df.to_csv("ent_predictions/"+attr+"_test_predictions_"+base+budget_name+".csv")

        # make the human entropy dictonary
        human_dict = predict(ent_model, tokenized["train"].to_pandas(), attr, predict_batch, just_pred_ent=True, ent_predictor=True)



        train_dataset_ids = ent_dataset_ids
        train_heldout_or_used_cids = combine_ids(train_heldout_or_used_cids, ent_dataset_ids)

        eval_single_ann, _ = sample_and_split_df(tokenized["test"].to_pandas())
        eval_dataset = eval_single_ann.rename_column(attr, 'label')
        l = initial_budget
        
        
        annot_nums[len(train_dataset_ids)] = np.sum([len(x) for x in train_dataset_ids.values()])
        tot_pos[len(train_dataset_ids)]=0
        tot_diff[len(train_dataset_ids)]=0
        tot_pos_rean[len(train_dataset_ids)]=0




    else:
        
        train_unused = tokenized["train"].to_pandas()
        train_unused =  train_unused[train_unused.apply(lambda row: not cid_filter(train_heldout_or_used_cids, row['comment_id'], row['annot_id']), axis=1)]

        train_dataset_ids, _, _, _ = sample_cids(train_unused, NUM_EGS)
    
        if add_method != "all":
            # remove all but single annot for each comment
            single, _ = sample_and_split_cids(train_dataset_ids)
            train_dataset_ids = single

            train_heldout_or_used_cids = combine_ids(train_heldout_or_used_cids, single)

            eval_single_ann, _ = sample_and_split_df(tokenized["test"].to_pandas())
            eval_dataset = eval_single_ann.rename_column(attr, 'label')

        else:
            eval_dataset = tokenized["test"].rename_column(attr, 'label')

        l = NUM_EGS

    if JUST_ENT:
        # for experiments where we're only training an entropy predictor
        return

    test_df = train_agg['test'].to_pandas()
    ents = [None for _ in range(len(test_df))]

    test_df['ents'] = ents


    if sample_corrected_entropy or sample_corrected_true_entropy:
        synthetic_egs = tokenized["train"].to_pandas().iloc[:0,:].copy()


    while l <= comment_nums[-1]: #no need to make train set larger than the last point we're collecting
        print(sum([len(x) for x in train_dataset_ids.values()]), l)
        prev_kl_div = None
        tolerance = 0
        max_tolerance = 1
        should_cont = True
        tot = 0

        train_dataset_w_ids = tokenized["train"].to_pandas()

        filtered_train_dataset_w_ids = train_dataset_w_ids[train_dataset_w_ids.apply(lambda row: cid_filter(train_dataset_ids, row['comment_id'], row['annot_id']), axis=1)]
        if sample_corrected_entropy or sample_corrected_true_entropy:
            
            filtered_train_dataset_w_ids = pd.concat([filtered_train_dataset_w_ids, synthetic_egs]).sort_values(by='comment_id', ascending=False)


        train_dataset = Dataset.from_pandas(filtered_train_dataset_w_ids, preserve_index=False)
        
        train_dataset = train_dataset.rename_column(attr, 'label').remove_columns(to_remove)


        while should_cont and not SKIP_TRAINING:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
            )

            trainer.train()

            
            df = predict(model, train_heldout['train'].to_pandas(), attr, predict_batch)
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

            print(kl_div, tolerance)

            tot +=1
            if tot > MAX:
                should_cont = False



        if ent_method == "online":
            ent_model = RobertaForSequenceClassification.from_pretrained('roberta-base',num_labels=1, force_download=DL).to(device)

            train_dataset_agg = DatasetDict(train_agg_tokenized.copy())
            df = train_agg_tokenized["train"].to_pandas()
            cids = list(train_dataset_w_ids.to_pandas().comment_id)

            train_dataset_agg['train'] = Dataset.from_pandas(df[df['comment_id'].isin(cids)], preserve_index=False).rename_column(attr+"_entropy", 'label')

            cols_not_needed = [x for x in train_dataset_agg['train'].column_names if x not in train_dataset.column_names]
            train_dataset_agg = train_dataset_agg['train'].remove_columns(cols_not_needed)


            eval_dataset_agg = train_heldout_tokenized["train"].rename_column(attr+"_entropy", 'label').remove_columns(cols_not_needed)

            ent_trainer = Trainer(
                model=ent_model,
                args=ent_training_args,
                train_dataset=train_dataset_agg,
                eval_dataset=eval_dataset_agg,
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics = compute_metrics,
                callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
            )

            if not SKIP_TRAINING:
                ent_trainer.train()

            if l in comment_nums:
                df = predict(ent_model, train_agg['test'].to_pandas(), attr, predict_batch, ent_predictor=True)
                df.to_csv("ent_predictions/"+attr+"_test_predictions_"+base+str(l)+".csv")





        ###############################################

        # get the train data that isn't used yet in training or in kl validation

        
        train_unused = tokenized["train"].to_pandas()

        train_unused =  train_unused[train_unused.apply(lambda row: not cid_filter(train_heldout_or_used_cids, row['comment_id'], row['annot_id']), axis=1)]

        # get the "human" and model entropies
        if ent_method == "online":
            human_dict = predict(ent_model, train_unused, attr, predict_batch, just_pred_ent=True, ent_predictor=True)

        model_dict = predict(model, train_unused, attr, predict_batch, just_pred_ent=True)


        train_unused['human_entropy'] = train_unused.apply(lambda x: human_dict[x['comment_id']], axis=1)
        train_unused['model_entropy'] = train_unused.apply(lambda x: model_dict[x['comment_id']], axis=1)

        


       


        if l % NUM_EGS != 0:
            # if the initial budget wasn't a multiple of 10, do a smaller update to get back to mutiples of 10
            update_size = NUM_EGS-(l % NUM_EGS)
        else:
            update_size = NUM_EGS


        # select either the top 10 by ent diff or abs ent diff
        to_add, pos, diff, pos_rean = sample_cids(train_unused, update_size, sampling_method=sampling_method, zero_human=zero_human, used=train_heldout_or_used_cids)
        num_pos += pos
        num_diff += diff
        num_pos_rean += pos_rean


        if l in comment_nums:
            print("OUT", l)

            df = predict(model, test_df, attr, predict_batch, ents=args.control_ent)
            df.to_csv("pred__"+attr+"/"+attr+"_test_predictions_"+base+str(l)+".csv")
            annot_nums[len(train_dataset_ids)] = np.sum([len(x) for x in train_dataset_ids.values()])

            tot_pos[len(train_dataset_ids)] = num_pos
            tot_diff[len(train_dataset_ids)] = num_diff
            tot_pos_rean[len(train_dataset_ids)] = num_pos_rean

            annot_nums_df = pd.DataFrame.from_dict({"comments":list(annot_nums.keys()),
                                                "annots":list(annot_nums.values()), 
                                                "tot_pos":list(tot_pos.values()), 
                                                "tot_diff":list(tot_diff.values()), 
                                                "tot_pos_rean":list(tot_pos_rean.values())})
            print("pred__"+attr+"/"+attr+"_annotation_nums_"+base+".csv")
            annot_nums_df.to_csv("pred__"+attr+"/"+attr+"_annotation_nums_"+base+".csv")
        else:
            print("CONT", l)
            
        l += update_size


        if add_method != "all":
            # remove all but single annot for each comment
            single_add, _ = sample_and_split_cids(to_add)

            to_add = single_add

            
            
        train_dataset_ids = combine_ids(train_dataset_ids, to_add)
        train_heldout_or_used_cids = combine_ids(train_heldout_or_used_cids, to_add)


        if sample_corrected_entropy:
            filtered_train_dataset_w_ids = train_dataset_w_ids[train_dataset_w_ids.apply(lambda row: cid_filter(train_dataset_ids, row['comment_id'], row['annot_id']), axis=1)]
            synthetic_egs = update_synth(synthetic_egs, filtered_train_dataset_w_ids, list(to_add.keys()), attr, human_dict)

        elif sample_corrected_true_entropy:
            filtered_train_dataset_w_ids = train_dataset_w_ids[train_dataset_w_ids.apply(lambda row: cid_filter(train_dataset_ids, row['comment_id'], row['annot_id']), axis=1)]
            synthetic_egs = update_synth(synthetic_egs, filtered_train_dataset_w_ids, list(to_add.keys()), attr, true_human_dict)

    


        torch.cuda.empty_cache() 
        model = RobertaForSequenceClassification.from_pretrained('roberta-base',num_labels=5, force_download=DL).to(device)

    





        
if __name__ == '__main__':
    main()