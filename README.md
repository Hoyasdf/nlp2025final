# 자연어처리 2025-1 지정주제 기말 프로젝트: GPT-2 구축

가상환경 생성
* conda env create -f env.yml
* conda activate nlp_final  

# Sentiment Analysis
학습&테스트 수행
* python classifier.py --use_gpu
테스트만 수행
* python classifier.py --use_gpu --test_only --para_test "테스트 데이터셋"

# Paraphrase Detection
학습&기본 경로 테스트 수행
* python paraphrase_detection.py --use_gpu
테스트만 수행
* python paraphrase_detection.py --use_gpu --test_only --para_test "테스트 데이터셋 경로"
학습&테스트 수행
* python paraphrase_detection.py --use_gpu --para_test "테스트 데이터셋 경로"

테스트만 수행시 가중치 파일이 깃허브에 업로드 할 수 없는 관계로 다음 링크에서 받을 수 있음
* https://drive.google.com/drive/folders/1wbDCfleDOlcVtYrK-y5kUHo79BKDkq5k?usp=sharing
