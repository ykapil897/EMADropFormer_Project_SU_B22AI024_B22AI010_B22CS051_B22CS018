import os, glob, librosa, torch, pandas as pd, numpy as np
import torch.nn as nn
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader

device="cuda" if torch.cuda.is_available() else "cpu"

DATASET_PATH="./RAVDESS"
SR=16000
MAX_LEN=SR*4
BATCH=8
EPOCHS=10

emotion_map={
"01":"neutral","02":"calm","03":"happy","04":"sad",
"05":"angry","06":"fearful","07":"disgust","08":"surprised"
}
label_to_idx={v:i for i,v in enumerate(emotion_map.values())}

files=glob.glob(os.path.join(DATASET_PATH,"Actor_*","*.wav"))
rows=[]
for f in files:
    code=os.path.basename(f).split("-")[2]
    rows.append([f,label_to_idx[emotion_map[code]]])

df=pd.DataFrame(rows,columns=["path","label"])
trdf,tedf=train_test_split(df,test_size=0.2,stratify=df.label,random_state=42)

processor=Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav=Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)

for p in wav.parameters():
    p.requires_grad=False

class DS(Dataset):
    def __init__(self,df): self.df=df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        p=self.df.loc[i,"path"]
        y=self.df.loc[i,"label"]
        a,_=librosa.load(p,sr=SR)
        a=np.pad(a,(0,max(0,MAX_LEN-len(a))))[:MAX_LEN]
        return a,y

def collate(batch):
    aud=[b[0] for b in batch]
    y=torch.tensor([b[1] for b in batch])
    x=processor(aud,sampling_rate=SR,padding=True,return_tensors="pt")
    return x.input_values,y

tr=DataLoader(DS(trdf),batch_size=BATCH,shuffle=True,collate_fn=collate)
te=DataLoader(DS(tedf),batch_size=BATCH,collate_fn=collate)

head=nn.Sequential(
    nn.Linear(768,256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256,8)
).to(device)

opt=torch.optim.Adam(head.parameters(),2e-4)
lossfn=nn.CrossEntropyLoss()

for e in range(EPOCHS):
    head.train()
    for x,y in tqdm(tr):
        x,y=x.to(device),y.to(device)
        with torch.no_grad():
            h=wav(x).last_hidden_state.mean(1)
        p=head(h)
        loss=lossfn(p,y)
        opt.zero_grad(); loss.backward(); opt.step()

head.eval()
pred=[]; true=[]
with torch.no_grad():
    for x,y in te:
        h=wav(x.to(device)).last_hidden_state.mean(1)
        p=head(h).argmax(1).cpu().numpy()
        pred.extend(p); true.extend(y.numpy())

acc=accuracy_score(true,pred)*100
f1=f1_score(true,pred,average="weighted")*100

print("Wav2Vec2 Accuracy:",acc)
print("Wav2Vec2 F1:",f1)