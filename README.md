# GraduationWork
졸업작품 관련 코드


JsontToCsv -> 폴더내 전체 json파일 읽어서 dataframe에 저장후 csv파일로 저장. <br>
AllCsv -> 폴더내 전체 csv 파일 한개로 병합. <br>
Dest -> 폴더내 특정 갯수만큼 파일 추출. <br>
Reindex -> df index 순서 Passage, Summary, Style 순으로 통일. <br>


https://huggingface.co/eenzeenee/t5-small-korean-summarization model 변경후 학습예정 -> 0416

# 현재 화풍 종류

Realism, Sketch, Oriental ink, Pop art, Cartoon, Painterly art, Black and white, Carricature, Gray-scale

# 모델 학습 방식

https://huggingface.co/eenzeenee/t5-small-korean-summarization 기반으로 학습

Realism, Sketch, Oriental ink, Pop art, Cartoon, Painterly art, Black and white, Carricature, Gray-scale중

                    
                    Realism, Painterly art
                    Oriental ink, Gray-scale
                    Pop art, Cartoon
                    Sketch, Carricature
                    Black and white, Gray-scale
                    
                    기준으로 설정
# Inference 분리

ACD에 Inference 분리해서 학습시간 따로 

