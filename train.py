from model import YNet
from data_model import DataGen
from trainer import Trainer
from utils import load_config
from sklearn.manifold import TSNE
from utils import *

def run():
    config_file = 'config.yaml'
    config = load_config(config_file)

    data_gen = DataGen(config=config)
    _data_stats = data_gen.get_data_stats()

    network_model = YNet(config=config, data_stats=_data_stats)
    trainer = Trainer(config=config)
    dann_acc, d_acc, dann_emb = trainer.train_and_evaluate(model=network_model, data_model=data_gen, verbose=True)
    print('MNIST-MNIST-M) accuracy:', dann_acc)
    print('Domain accuracy:', d_acc)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    dann_tsne = tsne.fit_transform(dann_emb)
        
    #plot_embedding
    plot_embedding(dann_tsne, data_gen.combined_test_labels.argmax(1), data_gen.combined_test_domain.argmax(1), 'Domain Adaptation')

if __name__ == '__main__':
    run()
