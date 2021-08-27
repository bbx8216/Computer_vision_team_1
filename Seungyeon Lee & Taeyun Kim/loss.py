import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):

    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

    
        self.S = S # grid의 크기
        self.B = B # bounding box의 수
        self.C = C # 예측하는 class 수

    
        self.lambda_noobj = 0.5
        self.lambda_coord = 5


    def compute_iou(self, bbox1, bbox2): # iou를 계산하는 함수
        N = bbox1.size(0)
        M = bbox2.size(0)

        
        lt = torch.max(
            bbox1[:, :2].unsqueeze(1).expand(N, M, 2), 
            bbox2[:, :2].unsqueeze(0).expand(N, M, 2)  
        )
        rb = torch.min(
            bbox1[:, 2:].unsqueeze(1).expand(N, M, 2), 
            bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)  
        )
        
        wh = rb - lt   
        wh[wh < 0] = 0 
        inter = wh[:, :, 0] * wh[:, :, 1] 

        
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]) # 첫번째 네모 박스
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]) # 두번쨰 네모 박스
        area1 = area1.unsqueeze(1).expand_as(inter) # 
        area2 = area2.unsqueeze(0).expand_as(inter) 

        
        union = area1 + area2 - inter # 합집합 부분
        iou = inter / union  # 교집합/합집합         

        return iou
    

    def forward(self, predictions, target):
        # 각 grid cell마다 2개의 bounding box를 예측
        # confidence score가 높은 1개의 bounding box를 학습에 사용
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        # predictions[..., 21:25]는 첫 번째 bounding box의 좌표값
        # predictions[..., 26:30]는 두 번째 bounding box의 좌표값
        # 위 두 가지를 정답에 해당하는 target의 죄표값과 비교해 iou를 계산
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        iou_maxes, bestbox = torch.max(ious, dim=0) # best_box에는 iou값이 더 큰 box의 index
        exists_box = target[..., 20].unsqueeze(3) 
        # grid cell에 ground truth box의 중심이 존재하는지 여부 파악
        # 존재한다면 exists_box=1 존재하지 않는다면 exists_box=0
        

        # localization loss 계산
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 26:30] #bestbox를 활용해 iou 값이 더 큰 박스 사용
                + (1 - bestbox) * predictions[..., 21:25]
            )
        )

        box_targets = exists_box * target[..., 21:25]

        
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        ) # bounding box 좌표값에 대해 mse 구하기

        
        # confidence loss
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )
        # predictions[..., 25:26]: 첫 번쨰 box의 confidence score
        # predictions[..., 20:21]: 첫 번쨰 box의 confidence score

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        

        # class loss
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2,),
            torch.flatten(exists_box * target[..., :20], end_dim=-2,),
        )
        # predictions[..., :20]에 해당하는, 
        # class의 score를 target과 비교하여 mse loss를 구함

        loss = (
            self.lambda_coord * box_loss  
            + object_loss  
            + self.lambda_noobj * no_object_loss  
            + class_loss  
        )

        return loss