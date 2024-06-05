## 관련된 모듈 설치
- scikit-learn 설치

```
pip install scikit-learn

or
in windows,

py -m pip install scikit-learn
```

## 상태 다이어그램
```mermaid
		stateDiagram-v2
	    [*] --> 챗봇_인스턴스_생성
	    챗봇_인스턴스_생성 --> 질문입력
	    질문입력 --> 종료
	    질문입력 --> !=종료
	    !=종료 --> find_best_answer()
	    find_best_answer() --> 질문입력
	    종료 --> [*]

```

## 클래스 다이어그램
```mermaid
		classDiagram
	    SimpleChatBot <|-- SimpleChatBotWithCosineSimilarity 
	    SimpleChatBot <|-- SimpleChatBotWithCalcDistance 
	    SimpleChatBot : questions
	    SimpleChatBot : answers
	    SimpleChatBot: vectorizer
	    SimpleChatBot: question_vectors
	    SimpleChatBot: load_data()
	    SimpleChatBot: find_best_answer()
	    class SimpleChatBotWithCosineSimilarity {
	      find_best_answer()
	    }
	    class SimpleChatBotWithCalcDistance {
	      find_best_answer()
	      calc_distance()
	    }
```

## 과제 설명
```
AI 개발 실무 2차 레포트과제
레벤슈타인 거리를 이용한 챗봇 구하기


14주차 실습 자료를 참고하여,
기존 TF-IDF와 Consine Similarlity를 이용해 챗봇을 구현한 코드를
레벤슈타인 거리를 기반으로 한 챗봇으로 수정하여 구현하시오.

학습 데이터 : 14주차 실습 데이터에 포함된 ChatbotData.csv

과제 상세 설명:  

1. 학습데이터의 질문과 chat의 질문의 유사도를 레벤슈타인 거리를 이용해 구하기
2. chat의 질문과 레벤슈타인 거리와 가장 유사한 학습데이터의 질문의 인덱스를 구하기
3. 학습 데이터의 인덱스의 답을 chat의 답변을 채택한 뒤 출력

참고 : 기말고사와 제출기한이 겹치는 대신 기말고사의 부담을 줄이기 위해 10문항으로 출제됩니다.

과제 제출방법: 과제를 구현한 코드를 본인의 github에 올린뒤, github의 주소를 과제로 제출하기

과제 조건

1. 코드에 대한 상세한 주석달기 (올바르게 작동되는 코드와 주석을 모두 제출했다면 100점)
2. gihub을 이용하지 못하는 학생은 워드 파일로 코드와 결과 화면 캡처후 제출 (부분 점수 80점)
3. 코드가 돌아가지 못하면 돌아가지 못하는 부분에 대한 주석을 달고 부분점수 획득 가능
3. 절대로 타인과 과제를 공유하지 말 것 (캡처본 파일을 검사하여 동일 파일일 경우 0점 처리)
6. 과제를 제대로 수행하지 못했지만, 제출만 했을 경우, 제출 점수 20점

```