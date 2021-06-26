# Scinetific paper abstractive summarization (Capstone design 2021-1)
## Overview
* 개인적으로 논문을 읽을 때 줄 간격이 좁고 글자 크기가 작아 재빨리 독해를 하는데 어려움을 겪어 논문내용을 요약하여 \
독해를 도와주는 서비스를 필요로 하게 되었다.
* 서비스의 완성까지는 감당하기 어려운 난이도이므로, 자연어처리 공부에 입문하고 향후 해당 기능을 만드는데 도움이 되도록 \
 text-summarization 모델의 설계를 주제로 하게 되었다.
 
* 4주차 - sequence to sequence-attention abstractive summarization의 이해
* 5주차 - 데이터셋 탐색과 선택, baseline study (NLP-kr/tensorflow-ml-nlp-tf2를 참조함)
* 6주차 - transformer 모델의 대략적인 이해, 데이터셋 살펴보기, 대략적인 모델의 설계.
* 10주차 - lemmatizing, val_loss가 nan이 나오는 문제 해결, transformer 모델 공부.
* 13주차 - LSTM 사용 Seq2Seq 모델 튜닝, LSTM-ATTN 모델 튜닝, 워드 임베딩 glove 활용, 결과보고서 초안 작성.
* 14주차 - resource exhausted error에 대해 해결법 탐색, 잘못된 예측이 나오는 결과에 대한 해결법 탐색. 
* 15주차 - 작은 길이의 문장 데이터셋으로 축소(kaggle inshorts news data), Rouge metric을 사용한 평가 적용.
* 16주차 - 최종발표.

## 과제 수행 방법
> 데이터 수집 및 전처리
> > 축약문 펼치기
* 아포스트로피 등으로 축약된 단어들을 길게 펼쳐준다
> > 불용어 처리
* 중요도가 떨어지거나 실제로는 쓰이지 않는 단어들을 삭제하거나 변경하여 준다
> > 표제어 추출
* 단어의 본디 의미를 살리며 동시에 연산량을 줄이기 위해서 표제어 추출을 수행한다.


> 훈련 시퀀스 준비
> > 
> 딥러닝 모델 
> >
> 모델 평가
*
> 추가 과제 제안
*

## Schedule
| Contents | March | April |  May  | June  |   Progress   |
|----------|-------|-------|-------|-------|--------------|
|Baseline study|    0   |    0   |       |       |    https://github.com/seung-00/text_summarization_project    |
|  데이터셋 수집  |  0     |   0    |       |       |        |
|  모델 설계  |       |   0    |    0   |    0   |        |
|  중간보고서제출  |       |       |    0   |       |        |
| 최종보고서제출 |       |      |       |     0  |        |

## Results
* 주요 코드
``` python3
def expand_contractions():
     # 정규표현식을 사용해서 축약된 단어들을 모두 찾아와서 바꾸기
    contractions_keys = '|'.join(contraction_map.keys()) # contractio_map의 key에 '|'를 추가
    contractions_pattern = re.compile(f'({contractions_keys})', flags=re.DOTALL) # contraction_pattern에 정규표현식 형태를 저장 ; 컴파일 옵션으로 . 메타 문자가 줄바꿈 문자(\n)를 포함하여 모든 문자와 매치되도록함

    def expand_match(contraction):# 축약문에 맞는 문자열을 찾는 함수
        # 조건에 맞는 전체 하위 문자열을 가져오기
        match = contraction.group(0)
        expanded_contraction = contraction_map.get(match)
        if not expand_contractions:
            print(match)
            return match
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text) # 축약문을 해당하는 문자열로 변환
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text 

 
```

``` python3
def lematize_text():
    data = nltk.word_tokenize(data)
    lemtzr=WordNetLemmatizer()
    out_data=" "
    for words in data:
        out_data+= lemtzr.lemmatize(words)+" "
    return out_data

```

``` python3
def delete_stopwords():
     _stopwords = stopwords.words('english') 
    _stopwords.append('cnn') # 불용어 사전에 (cnn)을 추가함. 이는 문서의 의미에 영향을 끼치지 않을 것이기 때문. 괄호지우는 작업도 정규표현식으로 더했더니 cnn만 남아서 그냥 cnn은 다 지워 버리는걸로 결정함
    text = text.split()
    word_list = [word for word in text if word not in _stopwords]
    return ' '.join(word_list)

```

``` python3
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True) # transposed key와 행렬곱

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # normalize를 위해 차원수로 나눠주기

    if mask is not None: # 마스킹할 영역 준비, mask가 있으면
        scaled_attention_logits += (mask * -1e9)  

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) #소프트맥스에 통과시켜 확률값으로 변환

    output = tf.matmul(attention_weights, v) # value와 행렬곱
    return output, attention_weights # 컨텍스트 벡터를 리턴
```

``` python3
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads # 몇개로 나눠줄 것인가
        self.d_model = d_model # 임베딩 차원수

        assert d_model % self.num_heads == 0 

        self.depth = d_model // self.num_heads #헤드 갯수로 나주어준다

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model) 
        
    def split_heads(self, x, batch_size): # 헤드 나누기
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)  # 각각 linear dense에 통과시킨후

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size) #배치사이즈에 따라 나눈다.

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask) #스케일드 닷 어텐션

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model)) # 원래 모양으로 돌려주기 위해
        output = self.dense(concat_attention) # concat
            
        return output, attention_weights
```
## Rouge Score
| Contents | baseline | myModel |  States of art  |
|----------|----------|---------|-----------------|
|  Rouge-1 |          |         |                 |
|  Rouge-2 |          |         |                 | 
|  Rouge-L |          |         |                 | 

## 수행 예시


## Conclusion
* inshorts news dataset을 바탕으로 텍스트 생성요약 모델을 작성하고 rouge 점수를 평가하였습니다.
* 이전에 비슷한 과제를 수행하였던 학우의 과제를 베이스라인으로 삼아 공부하고, 그것보다 긴 길이의 문장을 요약하는데 성공하였습니다. 그러나 온전히 저 자신의 힘으로 완성한 것은 아니었고 일단 모델을 완성시키기 위해 급하게 논문의 소스코드를 많이 차용하였습니다.
* 또한 Rouge Score에 맹점이 있는 것을 발견하였습니다. 시스템이 생성한 요약문장이 꽤나 훌륭한 경우에도 불구하고, 정답요약문에 쓰인 단어들과는 다른 단어를 사용하여 요약하였다면 Rouge Score의 점수가 좋지 못합니다. 따라서 정답요약문과 시스템요약문이 얼마나 겹치는가를 평가하는 기존의 Rouge metric과는 다르게, 시스템이 생성한 요약문의 단어가 워드임베딩에서 차지하는 벡터와 정답요약문의 단어가 워드임베딩에서 가지고 있는 벡터가 서로 얼마나 가까이 있느냐를 비교하는 metric을 만들면 조금 더 정확한 텍스트 요약 평가를 할 수 있을 것입니다.   

## ToDo
* 장문의 텍스트를 그보다 작은 길이로 잘라서 각개 요약을 하고 다시 합쳐서 정답요약문과 같은지 확인하기
* Rouge metric의 맹점-정답요약문과 다른 단어들을 사용하여 훌륭한 요약을 하였을 때 rouge 점수가 0정이 나오는 문제- 를 \n
해결할 방법 제시하기. 
* 한국어 문서도 요약해보기
