> 데이터셋 크기를 늘려서 LSTM-ATTN 모델 튜닝
> > 표본이 되는 데이터셋의 크기를 3만쌍까지 늘려서 똑같은 구조에 주입해 보았습니다. 이전과는 달리 loss와 val_loss가 여러 epoch를 거치며 굉장히 잘 수렴했습니다만, 실제 예측 결과를 확인해보니 overfiting이 생겼다는 것을 알게 되었습니다. 

![image](https://user-images.githubusercontent.com/73059667/123746302-6b5bec80-d8ec-11eb-9fa3-f4e8d5db6b12.png)


>  > loss는 이전에 비해 훨씬 줄였음에도 불구하고 예측 요약문이 예전보다 오히려 더 안 좋아졌습니다. 이전에는 그래도 원문과 상관관계를 가진 텍스트가 출력되었는데 지금은 전혀 관계가 없는 단어들이 출력됩니다.

> 워드 임베딩 glove 활용
> > 그 동안은 케라스에서 제공하는 API를 사용하여 원-핫-인코딩 방식의 워드 임베딩을 사용하고 있었습니다. 원-핫-인코딩의 경우 단어들 간의 관계를 알 수 없다는 문제가 있으므로 glove라는 사전 훈련된 워드 임베딩을 활용해 보았습니다.
> > 아래 코드 블럭은 그 주요 코드입니다.


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
