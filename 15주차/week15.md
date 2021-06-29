> 작은 길이의 문장 데이터셋으로 축소
> >  장문의 데이터셋인 CNN/Dailymail에서 Inshorts NEws Dataset (https://www.kaggle.com/shashichander009/inshorts-news-data) 로 변경을 하였습니다.

> 텍스트 전처리
> >i’m, you’ll 등의 축약문을 펼쳐주는 contractions 단어사전을 이전과 같이 사용하였습니다.
> > nltk stopword library를 사용한 stopword 삭제는 득보다 실이 많은 듯하여 수행하지 않았습니다. (having 등의 문자가 통째로 사라지는 문제가 발생하는 것을 발견하였습니다.)
> > lemmatization, 표제어 추출은 decoder에서 사용할 시퀀스에만 적용하기로 하였습니다. 생각하여 보니, 만들어진 요약모델을 일반적인 환경에서 사용할 때는 lemmatize된 텍스트가 입력되는 것이 아니기 때문에  encoder 시퀀스에까지 표제어 추출을 적용하게 된다면 정제된 입력데이터와 날것의 텍스트입력데이터 간에 요약문장이 차이가 나게 될 것입니다. 
> > 그동안 시퀀스의 최대 길이를 정의하고 그에 미치지 못하는 시퀀스에 대하여 padding만 진행하였었는데 최대길이를 넘어가는 시퀀스에 대하여 잘라주는 작업을 하지 않았다는 것을 깨달았습니다. 아마 이 때문에 제대로 된 결과가 안 나온듯 합니다.

> Rouge metric
> > 만들어진 요약모델을 평가하기 위한 metric입니다.
> > Rouge에서 recall은 다음과 같이 정의합니다. 
> > 
> > ![image](https://user-images.githubusercontent.com/73059667/123745970-f4265880-d8eb-11eb-9042-88aca1c5620b.png)
> > 
>>  Rouge에서 precision은 다음과 같이 정의합니다.
>>  
>>  ![image](https://user-images.githubusercontent.com/73059667/123746025-086a5580-d8ec-11eb-8369-ee80810ca602.png)
>>  
> >  Rouge-1은 시스템 요약본과 참조 요약본 간 겹치는 unigram의 수를 나타냅니다.
> > Rouge-2는 시스템 요약본과 참조 요약본 간 겹치는 bigram의 수를 나타냅니다.
> >Rouge-N은 trigram, 4th-gram...
> > Rouge-L은 최장 길이로 매칭되는 문자열을 측정합니다. Rouge-N과는 다르게 단어들의 연속적 매칭을 요구하지 않고, 어떻게든 문자열 내에서 발생하는 매칭을 측정하기 때문에 보다 유연한 성능 비교가 가능합니다.
> > Rouge-S는 특정 Window size가 주어졌을 때, Window size 내에 위치하는 단어쌍들을 묶어 해당 단어쌍들이 얼마나 중복되게 나타나는지를 측정합니다.(skip-gram)

> 단편적인 결과
> > 어느 정도 연관이 있는 단어들이 생성되는 것을 확인하였습니다.
> > ![image](https://user-images.githubusercontent.com/73059667/123746143-3354a980-d8ec-11eb-94e3-0a1f77b67a77.png)

