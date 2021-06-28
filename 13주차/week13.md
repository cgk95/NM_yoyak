``` python3
def vocab_creater(text_lists, VOCAB_SIZE):
  tokenizer = Tokenizer(num_words=VOCAB_SIZE)
  tokenizer.fit_on_texts(text_lists)
  dictionary = tokenizer.word_index
  word2idx = {}
  idx2word = {}
  for k, v in dictionary.items():
      if v < VOCAB_SIZE:
          word2idx[k] = v-1
          idx2word[v-1] = k
      if v >= VOCAB_SIZE:
          break
  return word2idx, idx2word

```
``` python3
word2idx, idx2word = vocab_creater(data['article']+data['highlights'],VOCAB_SIZE)
word_index = {}
word_index['word2idx'] = word2idx
word_index['idx2word'] = idx2word
pickle.dump(word_index, open('word_idx.pkl','wb'))

```
``` python3
word_index = pickle.load(open('word_idx.pkl','rb'))
print(word_index.keys())
idx2word = word_index['idx2word']
word2idx = word_index['word2idx']
del word_index
print(word2idx)
print(idx2word)

```
``` python3

input_seq = pad_sequences(input_seq, maxlen = max_input_len, dtype='int32', padding='post', truncating='post')
output_seq = pad_sequences(output_seq, maxlen = max_output_len, dtype='int32', padding='post', truncating='post')

```
