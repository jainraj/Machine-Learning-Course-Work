embedding_types = {
    0: 'pre-trained-untrainable',
    1: 'pre-trained-trainable',
    2: 'untrained'
}

input_path = '../input/stanford-natural-language-inference-corpus/'
word2vec_path = '../input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin'
save_dir = '/kaggle/working/'

label_map = {
    'neutral': 0,
    'entailment': 1,
    'contradiction': 2,
}

label_inverse_map = {
    0: 'neutral',
    1: 'entailment',
    2: 'contradiction',
}

hyper_parameters_search_grid = {
    'optimiser': ['adam', 'sgd'],  # fixed to adam
    'learning_rates': [1e-2, 1e-3],  # fixed to 1e-3
    'regularisation': ['dropout-0.2', 'dropout-0.5', 'none', 'l2-0.005', 'l2-0.0005'],  # fixed to dropout-0.2
    'layers_nodes': ['ip/2-act-drop-pred', 'ip/2-act-drop-ip/4-act-drop-pred'],  # fixed to longer one
    'activations': ['sigmoid', 'relu'],  # fixed to relu
    'recurrent_hidden': [50, 200, 300],  # fixed to 200
    'recurrent_layers': [2, 3],  # fixed to 2
    'embedding': ['w2v', 'w2v-train', 'train(300)'],  # fixed w2v
    'model': ['lstm', 'gru', 'rnn'],  # fixed to gru
    'stop_words': ['remove', 'dont'],  # fixed to dont
    'recurrent_dropout_probability': [0, 0.2],  # fixed to 0
}
