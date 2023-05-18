# GraduationWork
졸업작품 관련 코드


JsontToCsv -> 폴더내 전체 json파일 읽어서 dataframe에 저장후 csv파일로 저장. <br>
AllCsv -> 폴더내 전체 csv 파일 한개로 병합. <br>
Dest -> 폴더내 특정 갯수만큼 파일 추출. <br>
Reindex -> df index 순서 Passage, Summary, Style 순으로 통일. <br>


https://huggingface.co/eenzeenee/t5-small-korean-summarization model 변경후 학습예정 -> 0416

# 권장 사양
GPU DRAM 12G이상

# Model Ckpt
Stable Diffusion : https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt <br>
NLP Model: https://drive.google.com/file/d/1h5B8mCdGEMNgrbl6RJzje_LrXSdBo1XK/view?usp=share_link <br>
을 다운받고 폴더의 상위폴더에 저장

# Streamlits

`dasd`

# Reference

https://github.com/Stability-AI/stablediffusion

@InProceedings{Rombach_2022_CVPR,<br>
&nbsp;&nbsp;    author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\"orn},<br>
&nbsp;&nbsp;    title     = {High-Resolution Image Synthesis With Latent Diffusion Models},<br>
&nbsp;&nbsp;    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},<br>
&nbsp;&nbsp;    month     = {June},<br>
&nbsp;&nbsp;    year      = {2022},<br>
&nbsp;&nbsp;    pages     = {10684-10695}<br>
}<br>


