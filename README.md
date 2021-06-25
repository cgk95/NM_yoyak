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
|Baseline study|    0   |    0   |       |       |     Link1    |
|  데이터셋 수집  |  0     |   0    |       |       |     Link2    |
|  모델 설계  |       |   0    |    0   |       |     Link3    |
|  중간보고서제출  |       |       |    0   |       |     Link4    |
| 최종보고서제출 |       |      |       |     0  |     Link5    |

## Results
* 텍스트 전처리에 사용한 코드
``` python3
def expand_contractions():
 
```
* Web link



## Conclusion
* Summary, contribution, ...

## ToDo
* 장문의 텍스트를 그보다 작은 길이로 잘라서 각개 요약을 하고 다시 합쳐서 정답요약문과 같은지 확인하기
* Rouge metric의 맹점-정답요약문과 다른 단어들을 사용하여 훌륭한 요약을 하였을 때 rouge 점수가 0정이 나오는 문제- 를 \n
해결할 방법 제시하기. 
* 한국어 문서도 요약해보기
