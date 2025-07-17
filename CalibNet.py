import torch
from torch import nn
import torch.nn.functional as F
from Modules import resnet18
import torch
from torch import nn
import torch.nn.functional as F
from Modules import resnet18
class Aggregation(nn.Module):
    def __init__(self, inplanes=768, planes=96, final_feat=(5,2)):
        super(Aggregation, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes*4, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(planes*4)
        self.conv2 = nn.Conv2d(in_channels=planes*4, out_channels=planes*4, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(planes*4)
        self.conv3 = nn.Conv2d(in_channels=planes*4, out_channels=planes*2, kernel_size=(2,1), stride=2)
        self.bn3 = nn.BatchNorm2d(planes*2)
        
        self.tr_conv = nn.Conv2d(in_channels=planes*2, out_channels=planes, kernel_size=1, stride=1)
        self.tr_bn = nn.BatchNorm2d(planes)
        self.rot_conv = nn.Conv2d(in_channels=planes*2, out_channels=planes, kernel_size=1, stride=1)
        self.rot_bn = nn.BatchNorm2d(planes)
        
        self.tr_drop = nn.Dropout2d(p=0.2)
        self.rot_drop = nn.Dropout2d(p=0.2)
        
        self.tr_pool = nn.AdaptiveAvgPool2d(output_size=final_feat)
        self.rot_pool = nn.AdaptiveAvgPool2d(output_size=final_feat)
        
        self.fc1 = nn.Linear(planes*final_feat[0]*final_feat[1], 3)  # translation
        self.fc2 = nn.Linear(planes*final_feat[0]*final_feat[1], 3)  # rotation
        
        # FIXED: Better initialization for calibration task
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
        
        # FIXED: Smaller initial weights for transformation prediction
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x: torch.Tensor):
        # FIXED: Add activation functions that were missing
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Translation branch
        x_tr = F.relu(self.tr_bn(self.tr_conv(x)))
        x_tr = self.tr_drop(x_tr)
        x_tr = self.tr_pool(x_tr)
        x_tr = self.fc1(x_tr.view(x_tr.shape[0], -1))
        
        # Rotation branch
        x_rot = F.relu(self.rot_bn(self.rot_conv(x)))
        x_rot = self.rot_drop(x_rot)
        x_rot = self.rot_pool(x_rot)
        x_rot = self.fc2(x_rot.view(x_rot.shape[0], -1))
        
        # FIXED: Apply tanh to limit the output range for better training stability
        x_tr = torch.tanh(x_tr) * 0.5  # Limit translation to [-0.5, 0.5]
        x_rot = torch.tanh(x_rot) * 0.2  # Limit rotation to [-0.2, 0.2] radians (~11 degrees)
        
        return x_rot, x_tr

class CalibNet(nn.Module):
    def __init__(self, backbone_pretrained=False, depth_scale=100.0):
        super(CalibNet, self).__init__()
        self.scale = depth_scale
        
        # FIXED: Add proper feature extraction with more layers
        self.rgb_resnet = resnet18(inplanes=3, planes=64)
        
        # FIXED: Better depth processing with proper scaling
        self.depth_resnet = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            resnet18(inplanes=32, planes=32),
        )
        
        # FIXED: Proper channel dimensions (512 + 256 -> 768)
        self.aggregation = Aggregation(inplanes=512+256, planes=96)
        
        # FIXED: Add feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 768, kernel_size=3, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if backbone_pretrained:
            try:
                self.rgb_resnet.load_state_dict(torch.load("resnetV1C.pth")['state_dict'], strict=False)
                print("Loaded pretrained RGB backbone")
            except:
                print("Warning: Could not load pretrained RGB backbone")
        
        self.to(self.device)
    
    def forward(self, rgb: torch.Tensor, depth: torch.Tensor):
        # rgb: [B,3,H,W]
        # depth: [B,1,H,W]
        
        # FIXED: Better depth preprocessing
        x2 = depth.clone()
        x2 = x2 / self.scale
        
        # FIXED: Add depth normalization
        x2 = torch.clamp(x2, 0, 1)  # Clamp to [0,1]
        
        # Extract features
        x1 = self.rgb_resnet(rgb)[-1]  # [B, 512, H', W']
        x2 = self.depth_resnet(x2)[-1]  # [B, 256, H', W']
        
        # FIXED: Ensure spatial dimensions match
        if x1.shape[-2:] != x2.shape[-2:]:
            x2 = F.interpolate(x2, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        
        # Feature fusion
        feat = torch.cat((x1, x2), dim=1)  # [B, 768, H', W']
        
        # FIXED: Apply feature fusion
        feat = self.feature_fusion(feat)
        
        # Get transformation parameters
        x_rot, x_tr = self.aggregation(feat)
        
        return x_rot, x_tr

if __name__=="__main__":
    x = (torch.rand(2,3,1242,375).cuda(),torch.rand(2,1,1242,375).cuda())
    model = CalibNet(backbone_pretrained=False).cuda()
    model.eval()
    rotation,translation = model(*x)
    print("translation size:{}".format(translation.size()))
    print("rotation size:{}".format(rotation.size()))


