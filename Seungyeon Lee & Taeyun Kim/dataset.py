import torch
import os
import pandas as pd
from PIL import Image


class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1]) # 파일이 있는 경로 불러오기
        boxes = []
        with open(label_path) as f: # with open(파일경로) as 파일 객체:
            # with as 구문을 빠져나가면서 자동으로 close 함수도 호출
            for label in f.readlines(): # 파일 객체로 한 줄씩 읽기 -> 파일 입출력으로 읽어오기
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    # if float(x)!=int(float(x)): float(x)
                    # else int(x)
                    for x in label.replace("\n", "").split() # "₩n"을 ""로 바꾸기 
                ]

                boxes.append([class_label, x, y, width, height]) # boxes에 추가

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0]) # 이미지 파일 있는 경로
        image = Image.open(img_path) # 이미지 불러오기
        boxes = torch.tensor(boxes) # boxes를 tensor로 바꾸기

        if self.transform:
            image, boxes = self.transform(image, boxes)

        
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist() # box를 다시 list로
            # 28번째 줄 참고
            class_label = int(class_label) # 정수형으로 바꾸기

            i, j = int(self.S * y), int(self.S * x) 
            # i: the cell row
            # j: the cell column
            x_cell, y_cell = self.S * x - j, self.S * y - i

        
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1

                box_coordinates = torch.tensor( # Box coordinates
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                label_matrix[i, j, class_label] = 1 # class_label one hot 인코딩

        return image, label_matrix