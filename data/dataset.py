import torch
from torch.utils.data import Dataset
import PIL
import nibabel as nib

from utils.normalize import *
from utils.transform import *

class CustomDataset(Dataset):
    def __init__(self, axial_info, sagittal_info, coronal_info, transform):
        self.all_features=[]
        self.labels=[]
        self.transform=transform

        self.asd=0
        self.tc=0

        for i in range(len(axial_info)):
            person_data=torch.Tensor([])
            
            path=axial_info.iloc[i].values[-1]
            nii=nib.load(path)
            img=nii.get_fdata()

            #gks
            #aixal
            axial_features=[]
            axial_volume=img.transpose(2,1,0)

            slice=axial_volume.shape[0]
            axial_volume=axial_volume[slice//4:-slice//4]
            axial_entropy=axial_info.iloc[i].values[0:20]

            for e in axial_entropy:
                axial_features.append(self.transform(PIL.Image.fromarray(gray_to_rgb(axial_volume[e]).astype(np.uint8))))
            
            person_data=torch.stack(axial_features, 0)

            #sagittal
            sagittal_features=[]
            sagittal_volume=img.transpose(0,2,1)
            sagittal_volume=np.flip(sagittal_volume, axis=1)

            slice=sagittal_volume.shape[0]
            sagittal_volume=sagittal_volume[slice//4:-slice//4]
            sagittal_entropy=sagittal_info.iloc[i].values[0:20]

            for e in sagittal_entropy:
                sagittal_features.append(self.transform(PIL.Image.fromarray(gray_to_rgb(sagittal_volume[e]).astype(np.uint8))))
            
            sagittal_data=torch.stack(sagittal_features, 0)
            person_data=torch.stack([person_data, sagittal_data], 0)

            #coronal
            coronal_features=[]
            coronal_volume=img.transpose(1,2,0)
            coronal_volume=np.flip(coronal_volume, axis=1)

            slice=coronal_volume.shape[0]
            coronal_volume=coronal_volume[slice//4:-slice//4]
            coronal_entropy=coronal_info.iloc[i].values[0:20]

            for e in coronal_entropy:
                coronal_features.append(self.transform(PIL.Image.fromarray(gray_to_rgb(coronal_volume[e]).astype(np.uint8))))
            
            coronal_data=torch.stack(coronal_features, 0)
            person_data=torch.cat((person_data, coronal_data.unsqueeze(0)), 0)
            #self.transform(Image.fromarray(gray_to_rgb(data).astype(np.uint8)))
            
            #print(person_data.shape)
            self.all_features.append(person_data)
            if 'ASD' in path:
                self.labels.append(np.array([0,1]))
                self.asd+=1
            else:
                self.labels.append(np.array([1,0]))
                self.tc+=1
                           
        #print(np.array(self.all_features).shape)
    def __getitem__(self, index):
        data=self.all_features[index]
        #print(data.shape
        label=self.labels[index]

        return data, torch.from_numpy(label)
    
    def _return_asd(self):
        return self.asd
    
    def _return_tc(self):
        return self.tc
    
    def __len__(self):
        return len(self.labels)
