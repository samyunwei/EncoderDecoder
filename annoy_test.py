from annoy import AnnoyIndex
import random
import numpy as np

if __name__ == '__main__':
    word_vector_path = "seq2seq/craw1.npz"
    word_vector = np.load(word_vector_path, allow_pickle=True)["embeddings"]

    f = 300 #建索引的维度
    t = AnnoyIndex(f) #初始化一个对象
    for index,ele in enumerate(word_vector):
        t.add_item(index, ele)

    t.build(20)  # 10 trees
    t.save('seq2seq/test.ann')

