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
