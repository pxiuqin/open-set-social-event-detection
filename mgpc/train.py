# from domain_aware_prompt import DAPromptHead
import joblib
from graph_prompt import distance2center2,pre_train_loss, prompt_loss_with_center
from load_data import getdata
from model_dynamic import GAT, GateLayer
# from model import GAT
import torch.optim as optim
import time
import numpy as np
import torch
import os
import dgl
from sklearn import metrics
from sklearn.cluster import KMeans
import torch.nn.functional as F
from gpf_prompt import CenterEmbedding, embeddings

INF = float("inf")

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def process_tensor(tensor, labels):
    result = torch.ones_like(tensor)
    for i, value in enumerate(tensor):
        if value.item() in labels:
            result[i] = 0

    # 统计值为 1 的个数
    count_ones = torch.sum(result == 1)

    # 统计值为 0 的个数
    count_zeros = torch.sum(result == 0)

    print(f"The number of 1's: {count_ones}")
    print(f"The number of 0's: {count_zeros}")

    return result

def run_kmeans(extract_features, extract_labels, indices, args,isoPath=None):
    # Extract the features and labels of the test tweets
    indices = indices.cpu().detach().numpy()

    if isoPath is not None:
        # Remove isolated points
        temp = torch.load(isoPath)
        temp = temp.cpu().detach().numpy()
        non_isolated_index = list(np.where(temp != 1)[0])
        indices = intersection(indices, non_isolated_index)

    # Extract labels
    labels_true = extract_labels[indices]
    # Extract features
    X = extract_features[indices, :]
    assert labels_true.shape[0] == X.shape[0]
    n_test_tweets = X.shape[0]

    # Get the total number of classes
    n_classes = len(set(list(labels_true)))

    # kmeans clustering
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
    labels = kmeans.labels_
    nmi = metrics.normalized_mutual_info_score(labels_true, labels)   # 因为是互信息所以label标识不必一致
    ari = metrics.adjusted_rand_score(labels_true, labels)
    ami = metrics.adjusted_mutual_info_score(labels_true, labels, average_method='arithmetic')
    print("nmi:",nmi,'ami:',ami,'ari:',ari)
    value = nmi
    global NMI
    NMI = nmi
    global AMI
    AMI = ami
    global ARI
    ARI = ari

    if args.metrics =='ari':
        print('use ari')
        value = ari
    if args.metrics=='ami':
        print('use ami')
        value = ami
    # Return number  of test tweets, number of classes covered by the test tweets, and kMeans cluatering NMI
    return (n_test_tweets, n_classes, value)

def evaluate(extract_features, extract_labels, indices, epoch, num_isolated_nodes, save_path, args, is_validation=True):
    epoch +=1   # 让其从1开始
    message = ''
    message += '\nEpoch '
    message += str(epoch)
    message += '\n'

    # with isolated nodes
    n_tweets, n_classes, value = run_kmeans(extract_features, extract_labels, indices, args)
    if is_validation:
        mode = 'validation'
    else:
        mode = 'test'
    message += '\tNumber of ' + mode + ' tweets: '
    message += str(n_tweets)
    message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
    message += str(n_classes)
    message += '\n\t' + mode +' '
    message += args.metrics +': '
    message += str(value)
    if num_isolated_nodes != 0:
        # without isolated nodes
        message += '\n\tWithout isolated nodes:'
        n_tweets, n_classes, value= run_kmeans(extract_features, extract_labels, indices, args,
                                              save_path + '/isolated_nodes.pt')
        message += '\tNumber of ' + mode + ' tweets: '
        message += str(n_tweets)
        message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
        message += str(n_classes)
        message += '\n\t' + mode + ' value: '
        message += str(value)
    message += '\n'
    global NMI
    global AMI
    global ARI
    with open(save_path + '/evaluate.txt', 'a') as f:
        f.write(message)
        f.write('\n')
        f.write("NMI "+str(NMI)+" AMI "+str(AMI) + ' ARI '+str(ARI))
    print(message)

    all_value_save_path = "/".join(save_path.split('/')[0:-1])
    print(all_value_save_path)

    with open(all_value_save_path + '/evaluate.txt', 'a') as f:
        f.write("block "+ save_path.split('/')[-1])
        f.write(message)
        f.write('\n')
        f.write("NMI "+str(NMI)+" AMI "+str(AMI) + ' ARI '+str(ARI) + '\n')

    if is_validation == False:
        # 如果在测试模型，把模型的结果保存为csv数据
        block_path = save_path + '/evaluate.csv'
        local_value = f"\n{epoch},{NMI},{AMI},{ARI}"
        if os.path.exists(block_path):
            with open(block_path, 'a') as f:
                f.write(local_value)
        else:
            with open(block_path, 'w') as f:
                f.write(f"Epoch,NMI,AMI,ARI")
                f.write(local_value)

        all_path = all_value_save_path + '/evaluate.csv'
        global_value = f"\n{save_path.split('/')[-1]},{epoch},{NMI},{AMI},{ARI}"
        if os.path.exists(all_path):
            with open(all_path, 'a') as f:
                f.write(global_value)
        else:
            with open(all_path, 'w') as f:
                f.write(f"Block,Epoch,NMI,AMI,ARI")
                f.write(global_value)

    return value

def extract_embeddings(g, model, num_all_samples, args, prompt = None, prompt_model = None):
    with torch.no_grad():
        model.eval()
        if prompt:
            prompt_model.eval()

        indices = torch.LongTensor(np.arange(0,num_all_samples,1))
        if args.use_cuda:
            indices = indices.cuda()
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.DataLoader(
            g, graph_sampler=sampler,
            batch_size=num_all_samples,
            indices = indices,
            shuffle=False,
            drop_last=False,
            )

        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
            blocks = [b.to(device) for b in blocks]
            extract_labels = blocks[-1].dstdata['labels']
            if prompt == 'prompt':
                blocks[0].srcdata['features'] = prompt_model.add(blocks[0].srcdata['features'])
                extract_features = model(blocks)
            else:
                extract_features = model(blocks)

        assert batch_id == 0
        if torch.any(torch.isnan(extract_features)):
            print('-------------------------------')

        extract_features = extract_features.data.cpu().numpy()
        extract_labels = extract_labels.data.cpu().numpy()

    return (extract_features, extract_labels)

def initial_train(i, args, data_split, metrics, embedding_save_path, model=None):
    save_path_i, in_feats, num_isolated_nodes, g, labels, train_indices, validation_indices, test_indices, _ = getdata(
        embedding_save_path, data_split, i, args)

    if model is None:  # Construct the initial model
        model = GAT(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
    if args.use_cuda:
        model.cuda()

    # print(model.parameters())
    # Optimizer
    optimizer = optim.Adam(params=model.parameters(),lr=args.lr, weight_decay=1e-4)

    # Start training
    message = "\n------------ Start initial training ------------\n"
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)
    # record the highest validation nmi ever got for early stopping
    best_vali_value = 1e-9
    best_epoch = 0
    wait = 0
    # record validation nmi of all epochs before early stop
    all_vali_value = []
    # record the time spent in seconds on each batch of all training/maintaining epochs
    seconds_train_batches = []
    # record the time spent in mins on each epoch
    mins_train_epochs = []
    for epoch in range(args.n_epochs):
        start_epoch = time.time()
        losses = []
        total_loss = 0
        for metric in metrics:
            metric.reset()

        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)  # 两层图结构
        dataloader = dgl.dataloading.DataLoader(
            g, train_indices, sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            )


        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
            blocks = [b.to(device) for b in blocks]
            batch_labels = blocks[-1].dstdata['labels']  # 最后一个块输出的标签

            start_batch = time.time()
            model.train()
            
            # forward
            pred = model(blocks)  # Representations of the sampled nodes (in the last layer of the NodeFlow).

            # 构建 block_adj
            src, dst = blocks[-1].edges()
            block_adj = torch.zeros(blocks[-1].num_src_nodes(), blocks[-1].num_dst_nodes())
            block_adj[src, dst] = 1

            loss,_pred = pre_train_loss(pred, batch_labels, block_adj.T)
            losses.append(loss.item())
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for metric in metrics:
                metric(_pred, batch_labels, loss)

            batch_seconds_spent = time.time() - start_batch
            seconds_train_batches.append(batch_seconds_spent)
            # end one batch

        total_loss /= (batch_id + 1)
        message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.n_epochs, total_loss)
        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
        mins_spent = (time.time() - start_epoch) / 60
        message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        mins_train_epochs.append(mins_spent)

        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
        # np.save(save_path_i + '/features_' + str(epoch) + '.npy', extract_features)
        # np.save(save_path_i + '/labels_' + str(epoch) + '.npy', extract_labels)

        # 这里的值是NMI或者AMI
        validation_value = evaluate(extract_features, extract_labels, validation_indices, epoch, num_isolated_nodes,
                                  save_path_i, args, True)
        all_vali_value.append(validation_value)

        # Early stop
        if validation_value > best_vali_value:
            best_vali_value = validation_value
            best_epoch = epoch
            wait = 0
            # Save model
            model_path = save_path_i + '/models'
            if (epoch == 0) and (not os.path.isdir(model_path)):
                os.mkdir(model_path)
            p = model_path + '/best.pt'
            torch.save(model.state_dict(), p)

        else:
            wait += 1
        if wait == args.patience:
            print('Saved all_mins_spent')
            print('Early stopping at epoch ', str(epoch))
            print('Best model was at epoch ', str(best_epoch))
            break
        # end one epoch

    # Save all validation nmi
    # np.save(save_path_i + '/all_vali_value.npy', np.asarray(all_vali_value))
    # Save time spent on epochs
    # np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
    # print('Saved mins_train_epochs.')
    # Save time spent on batches
    # np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
    # print('Saved seconds_train_batches.')
    # Load the best model of the current block
    best_model_path = save_path_i + '/models/best.pt'
    model.load_state_dict(torch.load(best_model_path))
    print("Best model loaded.")

    extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
    test_value = evaluate(extract_features, extract_labels, test_indices, epoch, num_isolated_nodes,
                                  save_path_i, args, False)
    label_center = {}

    # 变量不同label，并计算该label对应的平均特征表示
    for l in set(extract_labels):
        l_indices = np.where(extract_labels == l)[0]
        l_feas = extract_features[l_indices]
        l_cen = np.mean(l_feas, 0)
        label_center[l] = l_cen
    joblib.dump(label_center,save_path_i + '/models/label_center.dump')

    # label_center_emb = torch.FloatTensor(np.array(list(label_center.values()))).cuda() if args.use_cuda else torch.FloatTensor(np.array(list(label_center.values())))
    # torch.save(label_center_emb,save_path_i + '/models/center.pth')

    return model

def prompt_train(i, data_split, metrics, embedding_save_path, model, label_center, args, old_label_rate):
    save_path_i, in_feats, num_isolated_nodes, g, labels, test_indices, _ = getdata(
        embedding_save_path, data_split, i, args)

    # 训练之前先搞下评估，然后看看一个block完成后，评估的比较
    extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
    test_value = evaluate(extract_features, extract_labels, test_indices, -1, num_isolated_nodes,
                            save_path_i, args, True)
    
    p_num = args.gpf_p_num 
    n_prompt = GateLayer(in_feats)
    c_prompt = CenterEmbedding(args.out_dim, p_num)

    model_param_group = []
    model_param_group.append({"params": n_prompt.parameters()})
    model_param_group.append({"params": c_prompt.parameters()})

    # Optimizer
    optimizer = optim.Adam(params=model_param_group, lr=args.lr, weight_decay=1e-4, amsgrad=False)

    # Start prompt tuning
    message = "\n------------ Start prompt tuning------------\n"
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)
        
    # Create model path
    model_path = save_path_i + '/models'
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    label_center_emb = torch.FloatTensor(np.array(list(label_center.values()))).cuda() if args.use_cuda else torch.FloatTensor(np.array(list(label_center.values())))
    
    # record the time spent in seconds on each batch of all training/maintaining epochs
    seconds_train_batches = []
    # record the time spent in mins on each epoch
    mins_train_epochs = []
    for epoch in range(args.prompt_epochs):
        start_epoch = time.time()
        losses = []
        total_loss = 0
        for metric in metrics:
            metric.reset()

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.DataLoader(
            g, test_indices, sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            )

        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
            blocks = [b.to(device) for b in blocks]
            batch_labels = blocks[-1].dstdata['labels']

            start_batch = time.time()
            n_prompt.train()
            c_prompt.train()

            # forward
            blocks[0].srcdata['features'] = n_prompt.add(blocks[0].srcdata['features'])   # 放到GAT前
            pred = model(blocks)  # Representations of the sampled nodes (in the last layer of the NodeFlow) add prompt vector

            # 构建 block_adj
            src, dst = blocks[-1].edges()
            block_adj = torch.zeros(blocks[-1].num_src_nodes(), blocks[-1].num_dst_nodes())
            block_adj[src, dst] = 1

            # 伪标签的Loss函数
            # pred_norm = F.normalize(pred, 2, 1)
            rela_center_vec = torch.mm(pred,label_center_emb.t())
            rela_center_vec = F.normalize(rela_center_vec,2,1)
            epsilon = 1e-10
            rela_center_vec = torch.clamp(rela_center_vec, min=epsilon)
            entropy = torch.mul(torch.log(rela_center_vec), rela_center_vec)
            entropy = torch.sum(entropy,dim=1)
            value,old_indices = torch.topk(entropy.reshape(-1),int(old_label_rate*entropy.shape[0]),largest=True)   # 最大熵的一半
            # value,novel_indices = torch.topk(entropy.reshape(-1),int(entropy.shape[0]/2),largest=False)   # 最小熵的一半

            # 模型预测结果，通过欧式距离判断属于已知类的那类
            distance = distance2center2(pred, label_center_emb)
            distance = 1/F.normalize(distance, dim=1)
            label_pred = F.log_softmax(distance, dim=1)
            label_pred = torch.argmax(label_pred, dim=1, keepdim=True).squeeze()  # 这里是判断所有预测node，属于label_center_emb的那个类别

            # 开始构建一个样本，创建一个新类的标识数组
            pseudo_new_label = len(label_center.keys())
            pseudo_labels = torch.full((pred.shape[0],), pseudo_new_label)
            pseudo_labels[old_indices] = label_pred[old_indices]

            # 构建loss
            c_embedding_prompt = c_prompt(pred, pseudo_labels)
            loss,_pred = prompt_loss_with_center(pred, pseudo_labels, c_embedding_prompt, block_adj.T)
            losses.append(loss.item())
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # 验证 pseudo old label 是否准确
            # pseudo_old_labels = batch_labels[old_indices]
            # print('pseudo old event:',len(set(pseudo_old_labels.tolist())))
            # print('true old event:',len(label_center.keys()))
            # print(set(label_center.keys()) & set(pseudo_old_labels.tolist()))

            # # 构建一个 pseudo old&new label
            # true_labels = process_tensor(batch_labels, label_center.keys())
            # current_old_label_rate = torch.sum(true_labels == 0).item() / true_labels.shape[0]  # 计算当前旧类占比
            # true_labels[true_labels == 1] = pseudo_new_label
            # true_labels[old_indices] = label_pred[old_indices] 

            for metric in metrics:
                metric(_pred, pseudo_labels, loss)   # true_labels

            batch_seconds_spent = time.time() - start_batch
            seconds_train_batches.append(batch_seconds_spent)
            # end one batch

        total_loss /= (batch_id + 1)
        message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.prompt_epochs, total_loss)
        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
        mins_spent = (time.time() - start_epoch) / 60
        message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        # with open(save_path_i + '/log.txt', 'a') as f:
        #     f.write(message)
        mins_train_epochs.append(mins_spent)

        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args, prompt='prompt', prompt_model=n_prompt)
        
        # 保存测试的 features and labels
        np.save(f'{model_path}/features_{epoch}.npy', extract_features)
        np.save(f'{model_path}/labels_{epoch}.npy', extract_labels)

        # save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, save_path_i, epoch)
        test_value = evaluate(extract_features, extract_labels, test_indices, epoch, num_isolated_nodes,
                                save_path_i, args, False)

    # p = model_path + '/finetune.pt'
    # torch.save(model.state_dict(), p)
    # print('finetune model saved after epoch ', str(epoch))

    with torch.no_grad():
        c_prompt.eval()
        c_embedding_prompt = c_prompt(torch.FloatTensor(extract_features), torch.LongTensor(extract_labels))

    # update & save label_center
    for l in set(extract_labels):
        # 使用CenterEmbedding
        label_center[l] = c_embedding_prompt[l]
    # joblib.dump(label_center,save_path_i + '/models/label_center.dump')

    # 对旧类占比求平均
    # old_label_rate = (old_label_rate + current_old_label_rate) / 2 
    # print(old_label_rate)
    
    # # Save time spent on epochs
    # np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
    # print('Saved mins_train_epochs.')
    # # Save time spent on batches
    # np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
    # print('Saved seconds_train_batches.')

    return model,label_center,old_label_rate
