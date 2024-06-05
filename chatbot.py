import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from abc import *
import random

#                                 챗봇(추상클래스)
#                             +--------------------+
#                             |   SimpleChatBot    |
#                             +--------------------+
#                                      ^    ^
#                                      |    |
#                                      |    |
#                                      |    |
#                 +--------------------+    +------------------+
#                 |                                            |
#                 |                                            |
#  +-----------------------------------+   +-------------------------------+
#  | SimpleChatBotWithCosineSimilarity |   | SimpleChatBotWithCalcDistance |
#  +-----------------------------------+   +-------------------------------+
#      코사인 유사도를 구현한 챗봇          레벤슈타인 거리 계산을 구현한 챗봇
#
#


#
# 챗봇
#
class SimpleChatBot(metaclass=ABCMeta):
    def __init__(self, filepath):
        self.questions, self.answers = self.load_data(filepath)
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.questions)  # 질문을 TF-IDF로 변환

    def load_data(self, filepath):
        data = pd.read_csv(filepath)
        questions = data['Q'].tolist()  # 질문열만 뽑아 파이썬 리스트로 저장
        answers = data['A'].tolist()   # 답변열만 뽑아 파이썬 리스트로 저장
        return questions, answers
    
    @abstractmethod
    def find_best_answer(self, input_sentence):
        pass

#
# 코사인 유사도를 구현한 챗봇
#
class SimpleChatBotWithCosineSimilarity(SimpleChatBot):
    def find_best_answer(self, input_sentence):
        '''코사인 유사도 값이 가장 큰 값에 해당하는 답을 리턴'''
        input_vector = self.vectorizer.transform([input_sentence])
        similarities = cosine_similarity(input_vector, self.question_vectors) # 코사인 유사도 값들을 저장
        
        best_match_index = similarities.argmax()   # 유사도 값이 가장 큰 값의 인덱스를 반환
        return self.answers[best_match_index]


#
# 레벤슈타인 거리 계산을 구현한 챗봇
#    
class SimpleChatBotWithCalcDistance(SimpleChatBot):
    def find_best_answer(self, input_sentence):
        '''가장 유사한 질문에 해당하는 인덱스에 해당하는 답을 리턴'''
        return self.answers[self.get_shortest_distance_index(input_sentence)]

    def get_shortest_distance_index(self, input_sentence):
        '''입력된 질문과 미리준비된 질문 리스트를 calc_distance() 함수를 호출하여 레벤슈타인 거리를 계산하고, 이를 기준으로 가장 유사한 질문에 해당하는 인덱스를 리턴'''
        shortest_distances = [] # (index, distance, question) 튜플 형식으로 저장
        for idx, question in enumerate(self.questions):
            distance = self.calc_distance(input_sentence, question)

            if not shortest_distances:                 #최초 수행시
                shortest_distances = [(idx, distance, question)]
            elif distance < shortest_distances[0][1]:  #거리가 짧은 것을 저장
                shortest_distances = [(idx, distance, question)]
            elif distance == shortest_distances[0][1]: #거리가 같을때는 append
                shortest_distances.append((idx, distance, question))

        #print(shortest_distances) # 테스트시 주석 해제

        # 같은 거리로 계산된 것중에 random하게 인덱스 리턴
        random_num_idx = random.randint(0,len(shortest_distances)-1)
        return shortest_distances[random_num_idx][0]

    def calc_distance(self, a, b):
        ''' 레벤슈타인 거리 계산하기 '''
        if a == b: return 0 # 같으면 0을 반환
        a_len = len(a) # a 길이
        b_len = len(b) # b 길이
        if a == "": return b_len
        if b == "": return a_len
        # 2차원 표 (a_len+1, b_len+1) 준비하기 --- (※1)
        # matrix 초기화의 예 : [[0, 1, 2, 3], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 0, 0, 0, 0], [4, 0, 0, 0, 0]]
        # [0, 1, 2, 3]
        # [1, 0, 0, 0]
        # [2, 0, 0, 0]
        # [3, 0, 0, 0] 
        matrix = [[] for i in range(a_len+1)] # 리스트 컴프리헨션을 사용하여 1차원 초기화
        for i in range(a_len+1): # 0으로 초기화
            matrix[i] = [0 for j in range(b_len+1)]  # 리스트 컴프리헨션을 사용하여 2차원 초기화
        # 0일 때 초깃값을 설정
        for i in range(a_len+1):
            matrix[i][0] = i
        for j in range(b_len+1):
            matrix[0][j] = j
        # 표 채우기 --- (※2)
        # print(matrix,'----------')
        for i in range(1, a_len+1):
            ac = a[i-1]
            # print(ac,'=============')
            for j in range(1, b_len+1):
                bc = b[j-1] 
                # print(bc)
                cost = 0 if (ac == bc) else 1  #  파이썬 조건 표현식 예:) result = value1 if condition else value2
                matrix[i][j] = min([
                    matrix[i-1][j] + 1,     # 문자 제거: 위쪽에서 +1
                    matrix[i][j-1] + 1,     # 문자 삽입: 왼쪽 수에서 +1   
                    matrix[i-1][j-1] + cost # 문자 변경: 대각선에서 +1, 문자가 동일하면 대각선 숫자 복사
                ])
                # print(matrix)
            # print(matrix,'----------끝')
        return matrix[a_len][b_len]
    
###########################################################################
# 프로그램 시작
###########################################################################

# CSV 파일 경로를 지정하세요.
filepath = 'ChatbotData.csv'

# 간단한 챗봇 인스턴스를 생성합니다.
#chatbot = SimpleChatBotWithCosineSimilarity(filepath)
chatbot = SimpleChatBotWithCalcDistance(filepath)

# '종료'라는 단어가 입력될 때까지 챗봇과의 대화를 반복합니다.
while True:
    input_sentence = input('You: ')
    if input_sentence.lower() == '종료':
        break
    response = chatbot.find_best_answer(input_sentence)
    print('Chatbot:', response)