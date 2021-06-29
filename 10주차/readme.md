> CNN/DailyMail 텍스트 데이터 전처리
> > 이전 주에 빠트렸던 Lemmatizing을 추가하였습니다. lemmatization은 번역하면 표제어 추출이며, 단어의 원형을 찾는 작업이라고 말할 수 있습니다. 단어 원형을 찾는 일에는 Stemming과 lemmatization 두 종류의 작업이 있는데 표제어 추출이 어근추출보다 단어의 의미를 잘 보존한 원형을 찾아낼 수 있으므로 표제어 추출방식을 적용하였습니다.

> Sequence to Sequence
> > loss가 nan이 나오는 문제를 해결하였습니다. 지금껏 잘못된 입력을 주입하고 있었습니다.
