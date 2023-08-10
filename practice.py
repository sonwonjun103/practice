import torch
from utils.config import CFG

axial=torch.randn(4,20,2048)
sagittal=torch.randn(4,20,2048)
coronal = torch.rand(4,20,2048)

person_fc=[]

for i in range(CFG['BATCH_SIZE']):
    concat_result1 = torch.concat([axial[i], sagittal[i], coronal[i]], dim=1)
    concat_result1 = torch.reshape(concat_result1, (3, 20, 2048))

    person_fc.append(concat_result1)

person=torch.stack(person_fc, 0)
print(person.shape)