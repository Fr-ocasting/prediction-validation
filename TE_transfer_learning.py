import torch 

def TE_transfer(model,num_nodes,args,model_dir =  'data/'):
    saved_checkpoint = torch.load(f'{args.abs_path}{model_dir}Trained_Time_Embedding{args.embedding_dim}.pkl')
    embedding_weights = {k: v for k, v in saved_checkpoint['state_dict'].items() if 'Tembedding' in k}

    if args.multi_embedding:
        
        embedding_weights['Tembedding.embedding.0.weight'] = embedding_weights['Tembedding.embedding.0.weight'].repeat(num_nodes,1)
        embedding_weights['Tembedding.embedding.0.bias'] = embedding_weights['Tembedding.embedding.0.bias'].repeat(num_nodes)

        embedding_weights['Tembedding.embedding.1.weight'] = embedding_weights['Tembedding.embedding.1.weight'].repeat(num_nodes,1)
        embedding_weights['Tembedding.embedding.1.bias'] = embedding_weights['Tembedding.embedding.1.bias'].repeat(num_nodes)
        
    model.load_state_dict(embedding_weights, strict=False)
    return(model)