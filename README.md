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
* 아포스트로피 등으로 축약된 단어들을 길게 펼쳐주어야 합니다. 해당 작업을 진행 하지 않을 경우, 축약문들이 모두 unknown 토큰으로 처리될 확률이 높기 때문에 요약성능을 하락시키게 됩니다.
* contraction 라이브러리를 이용하여 함수를 구현하였습니다.

![image](https://user-images.githubusercontent.com/73059667/123746489-acec9780-d8ec-11eb-985d-6c090eac96fe.png)
![image](https://user-images.githubusercontent.com/73059667/123746550-c392ee80-d8ec-11eb-979e-1d3effcb5e4b.png)
![image](https://user-images.githubusercontent.com/73059667/123746572-c8f03900-d8ec-11eb-9f45-b2d53338441e.png)


> > 불용어 처리
* 중요도가 떨어지거나 실제로는 쓰이지 않는 단어들을 삭제하거나 변경하여 줍니다. 문장의 의미를 유지하며 컴퓨터의 연산량을 줄이기 위한 작업입니다.
* nltk의 stopword 라이브러리를 사용하여 구현하였습니다.


> > 표제어 추출
* 단어의 본디 의미를 살리며 동시에 연산량을 줄이기 위해서 표제어 추출을 수행합니다. 표제어추출은 어간추출과는 다르게 단어들이 다른 형태를 가지더라도 그 뿌리를 찾아가 훈련과정에서 사용할 단어의 수를 줄여주는 역할을 합니다.
* 마찬가지로 

![image](https://user-images.githubusercontent.com/73059667/123746628-dad1dc00-d8ec-11eb-95ee-4c08c399e99c.png)



> 훈련 시퀀스 준비
> > 토크나이징
* 전처리를 한 텍스트 데이터셋을 정수 시퀀스의 단어 벡터의 집합으로 만들어 주기 위하여 공백을 기준으로 토크나이징을 합니다. keras.preprocessing 의 토크나이징 함수를 사용하였습니다.
> > 정수 시퀀스로의 변형
* 토크나이징을 한 텍스트를 각각의 고유한 정수로 변형하여 줍니다. 마찬가지로 keras.preprocessing의 함수를 사용하였습니다. 추가적으로, 사전 훈련된 언어모델을 사용하고 싶다면, 이 단계에서 내 텍스트 시퀀스와 언어모델의 시퀀스를 서로 매치시켜주어 사용할 수 있습니다.
> 딥러닝 모델 
> > Sequence to Sequence
* 입력된 시퀀스로부터 다른 도메인의 시퀀스를 출력하는 시스템입니다. 예를 들어기사 원문텍스트의 정수시퀀스를 가지고 정답 요약문의 정수시퀀스를 출력하는 식입니다.
* Sequence to Sequence는 크게는 인코더와 디코더라는 두 구조물로 구성됩니다. 
* 인코더는 입력 문장의 모든 단어들을 순차적으로 입력받은 후 마지막에 이 모든 단어 정보들을 압축해서 하나의 컨텍스트 벡터(context vector)를 생성하여 디코더에 넘겨줍니다.
* 디코더는 컨텍스트 벡터를 받아서 그에 따라 예측한 시퀀스를 한 개씩 순차적으로 출력합니다.

> > transformer (Attention mechanism)
* attention mechanism의 기본 아이디어는 디코더에서 출력 단어를 예측하는 매 시점마다, 인코더에서의 전체 입력 문장을 다시 한 번 참고하자는 것입니다.
* Attention(Q, K, V) = Attention Value
* 주어진 Query에 대해서 모든 Key와의 유사도를 각각 구합니다. 이후 이 유사도를 키와 맵핑되어있는 각각의 Value에 반영해줍니다. 그리고 유사도가 반영된 Value들을 모두 더해서 리턴하는데 이 리턴값이 Attention Value 가 됩니다.

![image](https://user-images.githubusercontent.com/73059667/123746759-081e8a00-d8ed-11eb-85e4-935284fb0ea6.png)


> 모델 평가

![image](https://user-images.githubusercontent.com/73059667/123746856-297f7600-d8ed-11eb-9cf4-324b934bb169.png)

> > Rouge metrics

* Rouge Score에서 Recall은 참조 요약본을 구성하는 단어 중 몇 개의 단어가 시스템 요약본의 단어들과 겹치는지를 보는 점수입니다.
* Rouge Score에서 Precision은 Recall과는 반대로 모델이 생성한 시스템 요약본 중 참조 요약본과 겹치는 단어들이 얼마나 많이 있는지를 보는 지표입니다.
* Recall과 Precision을 가지고 F1 score를 계산하여 최종적으로 Rouge Score를 구합니다.
* ROUGE-N: unigram, bigram, trigram 등 문장 간 중복되는 n-gram을 비교하는 지표입니다.
* ROUGE-L: LCS 기법을 이용해 최장 길이로 매칭되는 문자열을 측정합니다. 

![image](https://user-images.githubusercontent.com/73059667/123746957-4320bd80-d8ed-11eb-9bc9-a54c8231cccb.png)

> 추가 과제 제안
* 영어보다 한국어를 처리하는 것이 어려우므로, 한국어 데이터셋에 대하여 요약모델을 설계해보고 싶습니다.
* 한국어 요약모델을 만드는것에 성공하면, 거기에서 다시 법률문서를 요약하는 과제에 도전해보고 싶습니다.
* Rouge Score가 완벽한 평가지표가 아님을 알았으므로, 단순히 precision-reacall을 활용하는 것이 아니라, 생성한 요약문장의 단어들이 언어모델 특징벡터공간에서 정답요약문의 단어들과 얼마나 가까이 있는지를 가지고 평가지표를 만들어 보는 것을 추가과제로 제안합니다.

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
|  Rouge-1 |  0.2143  | 0.3534  |        0.4438       |
|  Rouge-2 |      0   |  0.2397 |       0.2153          | 
|  Rouge-L | 0.2143   |  0.3402 |         0.4117        | 

* baseline 모델은 한 두 문장을 두셋의 단어로 요약하는 모델이므로 둘 이상의 단어가 연속적으로 겹치는 정도를 평가하는 Rouge-2 점수는 0점이 나올수 밖에 없습니다..
* myModel의 경우 점수가 0점이 나오는 표본들을 제하고 점수를 계산하였기 때문에 해당 점수는 고평가 되어 있습니다.

## 수행 예시


## Conclusion
* inshorts news dataset을 바탕으로 텍스트 생성요약 모델을 작성하고 rouge 점수를 평가하였습니다.
* 이전에 비슷한 과제를 수행하였던 학우의 과제를 베이스라인으로 삼아 공부하고, 그것보다 긴 길이의 문장을 요약하는데 성공하였습니다. 그러나 온전히 저 자신의 힘으로 완성한 것은 아니었고 일단 모델을 완성시키기 위해 급하게 논문의 소스코드를 많이 차용하였습니다.
* 또한 Rouge Score에 맹점이 있는 것을 발견하였습니다. 시스템이 생성한 요약문장이 꽤나 훌륭한 경우에도 불구하고, 정답요약문에 쓰인 단어들과는 다른 단어를 사용하여 요약하였다면 Rouge Score의 점수가 좋지 못합니다. 따라서 정답요약문과 시스템요약문이 얼마나 겹치는가를 평가하는 기존의 Rouge metric과는 다르게, 시스템이 생성한 요약문의 단어가 워드임베딩에서 차지하는 벡터와 정답요약문의 단어가 워드임베딩에서 가지고 있는 벡터가 서로 얼마나 가까이 있느냐를 비교하는 metric을 만들면 조금 더 정확한 텍스트 요약 평가를 할 수 있을 것입니다.   

## ToDo
* 장문의 텍스트를 그보다 작은 길이로 잘라서 각개 요약을 하고 다시 합쳐서 정답요약문과 같은지 확인하기
* Rouge metric의 맹점-정답요약문과 다른 단어들을 사용하여 훌륭한 요약을 하였음에도 불구하고 그 결과로는 rouge 점수가 0점이 나오는 문제- 를 \n
해결할 방법 제시하기.  ->> (https://github.com/cgk95/abv_rouge)
* 한국어 문서도 요약해보기 ->> (https://github.com/cgk95/korabs_tudy)
