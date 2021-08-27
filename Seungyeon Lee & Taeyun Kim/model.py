import torch
import torch.nn as nn


architecture_config = [
    # (kernal_size, num_filters, stride, padding) 순으로 튜플에 저장
    (7, 64, 2, 3), # 7*7*64
    "M",           # M은 Maxpool Layer
    # Max pooling이란 map을 M*N으로 잘라낸 후 그 중 가장 큰 값을 고르는 것
    (3, 192, 1, 1), # 3*3*192
    "M",
    (1, 128, 1, 0), # 1*1*128
    (3, 256, 1, 1), # 3*3*256
    (1, 256, 1, 0), # 1*1*256
    (3, 512, 1, 1), # 3*3*512
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], #list일 경우 마지막 index 정수만큼 반복
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    # CNN은 Convolution Neural Network
    # 기본적으로 convolution layer -> pooling layer -> FC layer 순으로 진행
    # CNNBlock은 CNN에서 block을 담당하는 듯..
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module): # Yolo v1 모델
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture) # config 배열대로 conv layer 만들기
        # convolution layer는 입력 data로부터 feature를 추출하는 역할을 함
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x): 
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture): # config 배열을 architecture로 전달
        layers = []
        in_channels = self.in_channels # 값은 3

        for x in architecture: # config 안을 for문으로 
            if type(x) == tuple: #convolution layer일 경우
                layers += [ # CNN block 만들어서 layer 배열에 추가
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]

            elif type(x) == str: # Maxpool layer일 경우
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
                # pytorch에서 Maxpool 만들어서 layer에 추가하기

            elif type(x) == list: # 그림에서 *4, *2처럼
                conv1 = x[0] # convolution layer
                conv2 = x[1] # convolution layer
                num_repeats = x[2] # layer를 반복할 숫자

                for _ in range(num_repeats): # for문을 이용해 반복
                    layers += [
                        CNNBlock( # 첫번쨰 convolution layer를 활용해 만든 CNN block
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock( # 두번째 convolution layer를 활용해 만든 CNN block
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes): #fc layer
        # Fully connected layer -> 완전 연결 되었다
        # 한 층의 모든 뉴런이 그 다음 층의 모든 뉴런과 연결된 상태
        # 1차원 배열의 형태로 평탄화된 행렬을 통해 이미지를 분류하는데 사용되는 계층(?)
        S, B, C = split_size, num_boxes, num_classes


        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),
        )
