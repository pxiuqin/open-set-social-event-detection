import pandas as pd
import numpy as np
from datetime import datetime
import networkx as nx
from scipy import sparse
import dgl
from dgl.data.utils import save_graphs
from dgl.data.utils import load_graphs
# import en_core_web_lg
# import fr_core_news_lg
from sentence_transformers import SentenceTransformer
from time import time
import os
import argparse
import re
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score

def replaceAtUser(text):
    """ Replaces "@user" with "" """
    text = re.sub('@[^\s]+|RT @[^\s]+','',text)
    return text

def removeUnicode(text):
    """ Removes unicode strings like "\u002c" and "x96" """
    text = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', text)       
    text = re.sub(r'[^\x00-\x7f]',r'',text)
    return text

def replaceURL(text):
    """ Replaces url address with "url" """
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','url',text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text

def replaceMultiExclamationMark(text):
    """ Replaces repetitions of exlamation marks """
    text = re.sub(r"(\!)\1+", '!', text)
    return text

def replaceMultiQuestionMark(text):
    """ Replaces repetitions of question marks """
    text = re.sub(r"(\?)\1+", '?', text)
    return text

def removeEmoticons(text):
    """ Removes emoticons from text """
    text = re.sub(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', '', text)
    return text

def removeNewLines(text):
    text = re.sub('\n', '', text)
    return text

def preprocess_sentence(s):
    return removeNewLines(replaceAtUser(removeEmoticons(replaceMultiQuestionMark(replaceMultiExclamationMark(removeUnicode(replaceURL(s)))))))

def preprocess_french_sentence(s):
    return removeNewLines(replaceAtUser(removeEmoticons(replaceMultiQuestionMark(replaceMultiExclamationMark(replaceURL(s))))))

def documents_to_features(s_list, language = 'English'):
    '''
    Use Sentence-BERT to embed sentences.
    s_list: a list of sentences/ tokens to be embedded.
    output: the embeddings of the sentences/ tokens.
    '''
    if language == "English":
        model = SentenceTransformer("C:/Code/LLM/LLM-Model/all-MiniLM-L6-v2")
    elif language == "French":
        model = SentenceTransformer("C:/Code/LLM/LLM-Model/distiluse-base-multilingual-cased-v1")
    embeddings = model.encode(s_list, convert_to_tensor = True, normalize_embeddings = True)
    return embeddings.cpu()

# encode one times-tamp
# t_str: a string of format '2012-10-11 07:19:34'
def time_to_features(t_str):
    t = datetime.fromisoformat(str(t_str))
    OLE_TIME_ZERO = datetime(1899, 12, 30)
    delta = t - OLE_TIME_ZERO
    return [(float(delta.days) / 100000.), (float(delta.seconds) / 86400)]  # 86,400 seconds in day

def extract_node_feature(df, args):
    t_features = np.asarray([time_to_features(t_str) for t_str in df['created_at']])
    print("Time features generated.")
    s_list = df['processed_text'].tolist()
    d_features = np.asarray(documents_to_features(s_list, args.lang))
    print("original dfeatures generated")

    return np.concatenate((d_features, t_features), axis=1)

# construct a graph using tweet ids, user ids, entities and rare (sampled) words (4 modalities)
# if G is not None then insert new nodes to G
def construct_graph_from_df(df, G=None):
    if G is None:
        G = nx.Graph()
    for _, row in df.iterrows():
        tid = 't_' + str(row['tweet_id'])
        G.add_node(tid)
        G.nodes[tid]['tweet_id'] = True  # right-hand side value is irrelevant for the lookup

        user_ids = row['user_mentions']
        user_ids.append(row['user_id'])
        user_ids = ['u_' + str(each) for each in user_ids]
        # print(user_ids)
        G.add_nodes_from(user_ids)
        for each in user_ids:
            G.nodes[each]['user_id'] = True

        entities = row['entities']
        # entities = ['e_' + each for each in entities]
        # print(entities)
        G.add_nodes_from(entities)
        for each in entities:
            G.nodes[each]['entity'] = True

        # words = row['sampled_words']
        # words = ['w_' + each for each in words]
        # #print(words)
        # G.add_nodes_from(words)
        # for each in words:
        #     G.node[each]['word'] = True

        hashtags = row['hashtags']
        # print(words)
        G.add_nodes_from(hashtags)
        for each in hashtags:
            G.nodes[each]['hashtag'] = True

        edges = []
        edges += [(tid, each) for each in user_ids]
        edges += [(tid, each) for each in entities]
        # edges += [(tid, each) for each in words]
        edges += [(tid, each) for each in hashtags]
        G.add_edges_from(edges)

    return G


# convert networkx graph to dgl graph and store its sparse binary adjacency matrix
def to_dgl_graph_v3(G, save_path=None):
    message = ''
    print('Start converting heterogeneous networkx graph to homogeneous dgl graph.')
    message += 'Start converting heterogeneous networkx graph to homogeneous dgl graph.\n'
    all_start = time()

    print('\tGetting a list of all nodes ...')
    message += '\tGetting a list of all nodes ...\n'
    start = time()
    all_nodes = list(G.nodes)
    mins = (time() - start) / 60
    print('\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # print('All nodes: ', all_nodes)
    # print('Total number of nodes: ', len(all_nodes))

    print('\tGetting adjacency matrix ...')
    message += '\tGetting adjacency matrix ...\n'
    start = time()
    A = nx.to_numpy_array(G)  # Returns the graph adjacency matrix as a NumPy matrix.
    mins = (time() - start) / 60
    print('\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # compute commuting matrices
    print('\tGetting lists of nodes of various types ...')
    message += '\tGetting lists of nodes of various types ...\n'
    start = time()
    tid_nodes = list(nx.get_node_attributes(G, 'tweet_id').keys())
    userid_nodes = list(nx.get_node_attributes(G, 'user_id').keys())
    # word_nodes = list(nx.get_node_attributes(G, 'word').keys())
    hash_nodes = list(nx.get_node_attributes(G, 'hashtag').keys())
    entity_nodes = list(nx.get_node_attributes(G, 'entity').keys())
    del G
    mins = (time() - start) / 60
    print('\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\tConverting node lists to index lists ...')
    message += '\tConverting node lists to index lists ...\n'
    start = time()
    indices_tid = [all_nodes.index(x) for x in tid_nodes]
    indices_userid = [all_nodes.index(x) for x in userid_nodes]
    # indices_word = [all_nodes.index(x) for x in word_nodes]
    indices_hashtag = [all_nodes.index(x) for x in hash_nodes]
    indices_entity = [all_nodes.index(x) for x in entity_nodes]
    del tid_nodes
    del userid_nodes
    # del word_nodes
    del hash_nodes
    del entity_nodes
    mins = (time() - start) / 60
    print('\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # tweet-user-tweet
    print('\tStart constructing tweet-user-tweet commuting matrix ...')
    print('\t\t\tStart constructing tweet-user matrix ...')
    message += '\tStart constructing tweet-user-tweet commuting matrix ...\n\t\t\tStart constructing tweet-user matrix ...\n'
    start = time()
    w_tid_userid = A[np.ix_(indices_tid, indices_userid)]
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # convert to scipy sparse matrix
    print('\t\t\tConverting to sparse matrix ...')
    message += '\t\t\tConverting to sparse matrix ...\n'
    start = time()
    s_w_tid_userid = sparse.csr_matrix(w_tid_userid)
    del w_tid_userid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tTransposing ...')
    message += '\t\t\tTransposing ...\n'
    start = time()
    s_w_userid_tid = s_w_tid_userid.transpose()
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tCalculating tweet-user * user-tweet ...')
    message += '\t\t\tCalculating tweet-user * user-tweet ...\n'
    start = time()
    s_m_tid_userid_tid = s_w_tid_userid * s_w_userid_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tSaving ...')
    message += '\t\t\tSaving ...\n'
    start = time()
    if save_path is not None:
        sparse.save_npz(save_path + "s_m_tid_userid_tid.npz", s_m_tid_userid_tid)
        print("Sparse binary userid commuting matrix saved.")
        del s_m_tid_userid_tid
    del s_w_tid_userid
    del s_w_userid_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # tweet-ent-tweet
    print('\tStart constructing tweet-ent-tweet commuting matrix ...')
    print('\t\t\tStart constructing tweet-ent matrix ...')
    message += '\tStart constructing tweet-ent-tweet commuting matrix ...\n\t\t\tStart constructing tweet-ent matrix ...\n'
    start = time()
    w_tid_entity = A[np.ix_(indices_tid, indices_entity)]
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # convert to scipy sparse matrix
    print('\t\t\tConverting to sparse matrix ...')
    message += '\t\t\tConverting to sparse matrix ...\n'
    start = time()
    s_w_tid_entity = sparse.csr_matrix(w_tid_entity)
    del w_tid_entity
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tTransposing ...')
    message += '\t\t\tTransposing ...\n'
    start = time()
    s_w_entity_tid = s_w_tid_entity.transpose()
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tCalculating tweet-ent * ent-tweet ...')
    message += '\t\t\tCalculating tweet-ent * ent-tweet ...\n'
    start = time()
    s_m_tid_entity_tid = s_w_tid_entity * s_w_entity_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tSaving ...')
    message += '\t\t\tSaving ...\n'
    start = time()
    if save_path is not None:
        sparse.save_npz(save_path + "s_m_tid_entity_tid.npz", s_m_tid_entity_tid)
        print("Sparse binary entity commuting matrix saved.")
        del s_m_tid_entity_tid
    del s_w_tid_entity
    del s_w_entity_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # tweet-word-tweet
    print('\tStart constructing tweet-word-tweet commuting matrix ...')
    print('\t\t\tStart constructing tweet-word matrix ...')
    message += '\tStart constructing tweet-word-tweet commuting matrix ...\n\t\t\tStart constructing tweet-word matrix ...\n'
    start = time()
    # w_tid_word = A[np.ix_(indices_tid, indices_word)]
    w_tid_word = A[np.ix_(indices_tid, indices_hashtag)]
    del A
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # convert to scipy sparse matrix
    print('\t\t\tConverting to sparse matrix ...')
    message += '\t\t\tConverting to sparse matrix ...\n'
    start = time()
    s_w_tid_word = sparse.csr_matrix(w_tid_word)
    del w_tid_word
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tTransposing ...')
    message += '\t\t\tTransposing ...\n'
    start = time()
    s_w_word_tid = s_w_tid_word.transpose()
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tCalculating tweet-word * word-tweet ...')
    message += '\t\t\tCalculating tweet-word * word-tweet ...\n'
    start = time()
    s_m_tid_word_tid = s_w_tid_word * s_w_word_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    print('\t\t\tSaving ...')
    message += '\t\t\tSaving ...\n'
    start = time()
    if save_path is not None:
        sparse.save_npz(save_path + "s_m_tid_word_tid.npz", s_m_tid_word_tid)
        print("Sparse binary word commuting matrix saved.")
        del s_m_tid_word_tid
    del s_w_tid_word
    del s_w_word_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'

    # compute tweet-tweet adjacency matrix
    print('\tComputing tweet-tweet adjacency matrix ...')
    message += '\tComputing tweet-tweet adjacency matrix ...\n'
    start = time()
    if save_path is not None:
        s_m_tid_userid_tid = sparse.load_npz(save_path + "s_m_tid_userid_tid.npz")
        print("Sparse binary userid commuting matrix loaded.")
        s_m_tid_entity_tid = sparse.load_npz(save_path + "s_m_tid_entity_tid.npz")
        print("Sparse binary entity commuting matrix loaded.")
        s_m_tid_word_tid = sparse.load_npz(save_path + "s_m_tid_word_tid.npz")
        print("Sparse binary word commuting matrix loaded.")

    s_A_tid_tid = s_m_tid_userid_tid + s_m_tid_entity_tid
    del s_m_tid_userid_tid
    del s_m_tid_entity_tid
    s_bool_A_tid_tid = (s_A_tid_tid + s_m_tid_word_tid).astype('bool')
    del s_m_tid_word_tid
    del s_A_tid_tid
    mins = (time() - start) / 60
    print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
    message += '\t\t\tDone. Time elapsed: '
    message += str(mins)
    message += ' mins\n'
    all_mins = (time() - all_start) / 60
    print('\tOver all time elapsed: ', all_mins, ' mins\n')
    message += '\tOver all time elapsed: '
    message += str(all_mins)
    message += ' mins\n'

    if save_path is not None:
        sparse.save_npz(save_path + "s_bool_A_tid_tid.npz", s_bool_A_tid_tid)
        print("Sparse binary adjacency matrix saved.")
        s_bool_A_tid_tid = sparse.load_npz(save_path + "s_bool_A_tid_tid.npz")
        print("Sparse binary adjacency matrix loaded.")

    # create corresponding dgl graph
    G = dgl.DGLGraph(s_bool_A_tid_tid)
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())
    print()
    message += 'We have '
    message += str(G.number_of_nodes())
    message += ' nodes.'
    message += 'We have '
    message += str(G.number_of_edges())
    message += ' edges.\n'

    return all_mins, message


def construct_incremental_dataset_0922(args, df, save_path):
    data_split = []
    all_graph_mins = []
    message = ""
    # extract distinct dates
    distinct_dates = df.date.unique()
    # print("Distinct dates: ", distinct_dates)
    print("Number of distinct dates: ", len(distinct_dates))
    message += "Number of distinct dates: "
    message += str(len(distinct_dates))
    message += "\n"
    print("Start constructing initial graph ...")
    message += "\nStart constructing initial graph ...\n"

    if args.is_static:
        ini_df = df.loc[df['date'].isin(distinct_dates[:args.days])]
    else:
        ini_df = df.loc[df['date'].isin(distinct_dates[:7])]

    path = save_path + '0/'
    if not os.path.exists(path):
        os.mkdir(path)

    G = construct_graph_from_df(ini_df)
    grap_mins, graph_message = to_dgl_graph_v3(G, save_path=path)
    message += graph_message
    print("Initial graph saved")
    message += "Initial graph saved\n"
    # record the total number of tweets
    data_split.append(ini_df.shape[0])
    # record the time spent for graph conversion
    all_graph_mins.append(grap_mins)
    
    # extract and save the labels of corresponding tweets
    y = ini_df['event_id'].values
    y = [int(each) for each in y]
    np.save(path + 'labels.npy', np.asarray(y))
    print("Labels saved.")
    message += "Labels saved.\n"
    
    # preprocess the text contents of the messages
    ini_df['processed_text'] = [preprocess_sentence(s) for s in ini_df['text']] 
    np.save(path + 'df.npy', ini_df)
    x = extract_node_feature(ini_df, args)
    np.save(path + 'features.npy', x)
    print("Features saved.")
    message += "Features saved.\n\n"

    if not args.is_static:
        if args.lang == "English":
            data_count = len(distinct_dates) - 1   # 排除了最后一天
        else:
            data_count = len(distinct_dates)

        # subsequent days -> insert tweets day by day (skip the last day because it only contains one tweet)
        inidays = 7
        j = 6
        for i in range(inidays, data_count):
            print("Start constructing graph ", str(i - j), " ...")
            message += "\nStart constructing graph "
            message += str(i - j)
            message += " ...\n"
            incr_df = df.loc[df['date'] == distinct_dates[i]]
            path = save_path + str(i - j) + '/'
            if not os.path.exists(path):
                os.mkdir(path)

            G = construct_graph_from_df(
                incr_df)  # remove obsolete, version 2: construct graph using only the data of the day
            grap_mins, graph_message = to_dgl_graph_v3(G, save_path=path)
            message += graph_message
            print("Graph ", str(i - j), " saved")
            message += "Graph "
            message += str(i - j)
            message += " saved\n"
            # record the total number of tweets
            data_split.append(incr_df.shape[0])
            # record the time spent for graph conversion
            all_graph_mins.append(grap_mins)
            # extract and save the labels of corresponding tweets
            # y = np.concatenate([y, incr_df['event_id'].values], axis = 0)
            y = [int(each) for each in incr_df['event_id'].values]
            np.save(path + 'labels.npy', y)
            print("Labels saved.")
            message += "Labels saved.\n"

            incr_df['processed_text'] = [preprocess_sentence(s) for s in incr_df['text']] 
            np.save(path + 'df.npy', incr_df)
            x = extract_node_feature(incr_df, args)
            np.save(path + 'features.npy', x)
            print("Features saved.")
            message += "Features saved.\n"
    return message, data_split, all_graph_mins


def main(args):
    if args.is_static:
        save_path = "data/{}_hash_static-{}-{}/".format(args.date, str(args.days), args.lang)
    else:
        save_path = "data/{}_ALL_{}/".format(args.date, args.lang)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if args.lang == "French" or args.lang == "Arabic":
        df_np = np.load('tweet/{}_Twitter/All_{}.npy'.format(args.lang, args.lang), allow_pickle=True)
        df = pd.DataFrame(data=df_np, columns=["tweet_id", "user_id", "text", "time", "event_id", "user_mentions",
                                               "hashtags", "urls", "words", "created_at", "filtered_words", "entities",
                                               "sampled_words"])
        if args.lang == "Arabic":
            name2id = {}
            for id, name in enumerate(df['event_id'].unique()):
                name2id[name] = id
            print(name2id)
            df['event_id'] = df['event_id'].apply(lambda x: name2id[x])
            df.drop_duplicates(['tweet_id'], inplace=True, keep='first')

    elif args.lang == "English":
        p_part1 = 'tweet/English_Twitter/68841_tweets_multiclasses_filtered_0722_part1.npy'
        p_part2 = 'tweet/English_Twitter/68841_tweets_multiclasses_filtered_0722_part2.npy'
        df_np_part1 = np.load(p_part1, allow_pickle=True)
        df_np_part2 = np.load(p_part2, allow_pickle=True)
        df_np = np.concatenate((df_np_part1, df_np_part2), axis=0)
        df = pd.DataFrame(data=df_np, columns=["event_id", "tweet_id", "text", "user_id", "created_at", "user_loc", \
                                               "place_type", "place_full_name", "place_country_code", "hashtags",
                                               "user_mentions", "image_urls", "entities",
                                               "words", "filtered_words", "sampled_words"])

    df = df.sort_values(by='created_at').reset_index()
    
    # append date
    df['date'] = [d.date() for d in df['created_at']]

    message, data_split, all_graph_mins = construct_incremental_dataset_0922(args, df, save_path)
    with open(save_path + "node_edge_statistics.txt", "w") as text_file:
        text_file.write(message)
    np.save(save_path + 'data_split.npy', np.asarray(data_split))
    print("Data split: ", data_split)
    np.save(save_path + 'all_graph_mins.npy', np.asarray(all_graph_mins))
    print("Time sepnt on heterogeneous -> homogeneous graph conversions: ", all_graph_mins)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_static', type=bool, default=False)
    parser.add_argument('--init_feature', type=bool, default=False)
    parser.add_argument('--date', type=str, default='0620')
    parser.add_argument('--lang', type=str, default='English')   # French   English
    parser.add_argument('--days', type=int, default=7)
    args = parser.parse_args()
    main(args)