import os, glob, librosa, torch, numpy as np, pandas as pd
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_PATH = "./RAVDESS"
SR = 16000
MAX_LEN = SR * 3
BATCH = 16
EPOCHS = 15

emotion_map = {
    "01":"neutral","02":"calm","03":"happy","04":"sad",
    "05":"angry","06":"fearful","07":"disgust","08":"surprised"
}
label_to_idx = {v:i for i,v in enumerate(emotion_map.values())}

files = glob.glob(os.path.join(DATASET_PATH,"Actor_*","*.wav"))
rows=[]

for f in files:
    code=os.path.basename(f).split("-")[2]
    rows.append([f,label_to_idx[emotion_map[code]]])

df=pd.DataFrame(rows,columns=["path","label"])

def feat(path):
    y,_=librosa.load(path,sr=SR)
    y=np.pad(y,(0,max(0,MAX_LEN-len(y))))[:MAX_LEN]

    mfcc=librosa.feature.mfcc(y=y,sr=SR,n_mfcc=40)
    mel=librosa.feature.melspectrogram(y=y,sr=SR,n_mels=64)
    chroma=librosa.feature.chroma_stft(y=y,sr=SR)

    x=np.concatenate([
        np.mean(mfcc,axis=1),
        np.mean(mel,axis=1),
        np.mean(chroma,axis=1)
    ])
    return x.astype(np.float32)

X=[];Y=[]
for _,r in tqdm(df.iterrows(),total=len(df)):
    X.append(feat(r.path))
    Y.append(r.label)

X=np.array(X); Y=np.array(Y)

Xtr,Xte,Ytr,Yte=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=42)

class DS(Dataset):
    def __init__(self,X,Y):
        self.X=torch.tensor(X,dtype=torch.float32)
        self.Y=torch.tensor(Y,dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.Y[i]

tr=DataLoader(DS(Xtr,Ytr),batch_size=BATCH,shuffle=True)
te=DataLoader(DS(Xte,Yte),batch_size=BATCH)

class Net(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.m=nn.Sequential(
            nn.Linear(d,256),nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,128),nn.ReLU(),
            nn.Linear(128,8)
        )
    def forward(self,x): return self.m(x)

model=Net(X.shape[1]).to(device)
opt=torch.optim.Adam(model.parameters(),1e-3)
lossfn=nn.CrossEntropyLoss()

for e in range(EPOCHS):
    model.train()
    for x,y in tr:
        x,y=x.to(device),y.to(device)
        p=model(x)
        loss=lossfn(p,y)
        opt.zero_grad(); loss.backward(); opt.step()

model.eval()
pred=[]; true=[]
with torch.no_grad():
    for x,y in te:
        p=model(x.to(device)).argmax(1).cpu().numpy()
        pred.extend(p); true.extend(y.numpy())

acc=accuracy_score(true,pred)*100
f1=f1_score(true,pred,average="weighted")*100

print("MFCC Accuracy:",acc)
print("MFCC F1:",f1)