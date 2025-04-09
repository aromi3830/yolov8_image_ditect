# yolov8_image_ditect
### 이미지 분석하기
<b>  
사진 1.jpg가 바탕화면에 있어. 
코랩에서 실행할거야. 
content밑에 1.jpg저장하고 yolov8n.pt파일도 content 밑에 넣을거야. 
이미지 인식하기 위해서 yolov8을 사용하고 yolov8n.pt파일을 다운로드하게 해줘. 
인식하는 코드를 짜줘. 
결과는 output.jpg로 저장해줘. 
결과를 코랩에서 보여줘.

``` bash
!pip install ultralytics --upgrade -q
from google.colab import files

uploaded = files.upload()  # 여기서 3.jpg 선택
from ultralytics import YOLO

# YOLOv8n 모델 로드 (필요시 자동 다운로드)
model = YOLO('yolov8n.pt')

# 이미지 경로
image_path = '/content/3.jpg'

# 객체 탐지 수행
results = model(image_path)

# 감지된 객체 리스트
boxes = results[0].boxes
classes = boxes.cls.tolist()

# 사람 class = 0
person_count = sum(1 for cls in classes if int(cls) == 0)

print(f"사람 수: {person_count}명")
from PIL import Image
import matplotlib.pyplot as plt

# 결과 이미지 저장
results[0].save(filename='/content/output_people.jpg')

# 이미지 보여주기
img = Image.open('/content/output_people.jpg')
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis('off')
plt.title(f"감지된 사람 수: {person_count}명")
plt.show()
```
