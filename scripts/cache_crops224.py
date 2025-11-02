import pandas as pd, cv2, pathlib
IN = pd.read_csv("data/train.csv")
OUT_DIR = pathlib.Path("data/processed_224/train"); OUT_DIR.mkdir(parents=True, exist_ok=True)
rows = []
for i, r in IN.iterrows():
    img = cv2.imread(r.image_path)
    if img is None: continue
    if pd.notna(r.xmin):
        x1,y1,x2,y2 = map(int,[r.xmin,r.ymin,r.xmax,r.ymax])
        h,w = img.shape[:2]
        x1=max(0,min(x1,w-1)); x2=max(1,min(x2,w)); y1=max(0,min(y1,h-1)); y2=max(1,min(y2,h))
        if x2>x1 and y2>y1: img = img[y1:y2, x1:x2]
    img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
    out_path = OUT_DIR / f"{i:07d}.jpg"
    cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    rows.append({"image_path": str(out_path), "bcs_5class": int(r.bcs_5class)})
pd.DataFrame(rows).to_csv("data/train_224.csv", index=False)
