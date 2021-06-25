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


## Schedule
| Contents | March | April |  May  | June  |   Progress   |
|----------|-------|-------|-------|-------|--------------|
|Baseline study|    0   |    0   |       |       |    https://github.com/seung-00/text_summarization_project    |
|  데이터셋 수집  |  0     |   0    |       |       |        |
|  모델 설계  |       |   0    |    0   |       |        |
|  중간보고서제출  |       |       |    0   |       |        |
| 최종보고서제출 |       |      |       |     0  |        |

## Results
* 주요 코드
``` python3
def expand_contractions():
 
```
``` python3
def lematize_text():
 
```
``` python3
def delete_stopwords():
 
```
``` python3
def dot_product_attention():
 
```
``` python3
class MultiHeadAttention():
 
```
## Rouge Score


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
