import json
import sys
import cv2
import os
import numpy as np
from collections import deque
from ultralytics import YOLO
import torch
import easyocr
gpuDevice = "cuda" if torch.cuda.is_available() else "cpu"
vPath = os.path.join("models", "yolo26n.pt")
pPath = os.path.join("models", "yolo11n.pt")
video0 = os.path.join("data", "0.mp4")
video1 = os.path.join("data", "1.mp4")
jsonFile = "data.json"
smoothFrames = 8
def processParking(frame, carModel, slotHistory):
  outFrame = frame.copy()
  with open(jsonFile, "r") as f:
    slots = json.load(f)
  if len(slotHistory) == 0:
    for _ in slots:
      slotHistory.append(deque(maxlen=smoothFrames))
  res = carModel(outFrame, verbose=False, conf=0.1, classes=[9, 10], device=gpuDevice)[0]
  carCenters = []
  if res.obb is not None and len(res.obb) > 0:
    for obbItem in res.obb:
      pts4 = obbItem.xyxyxyxy[0].cpu().numpy().astype(int)
      cv2.polylines(outFrame, [pts4.reshape((-1, 1, 2))], True, (0, 0, 0), 1)
      cx = int(pts4[:, 0].mean())
      cy = int(pts4[:, 1].mean())
      carCenters.append((cx, cy))
  occupiedCount = 0
  for i, slot in enumerate(slots):
    pts = np.array(slot["points"], np.int32)
    isUsedNow = any(cv2.pointPolygonTest(pts, c, False) >= 0 for c in carCenters)
    slotHistory[i].append(1 if isUsedNow else 0)
    isUsed = sum(slotHistory[i]) > len(slotHistory[i]) / 2
    if isUsed:
      occupiedCount += 1
    sClr = (0, 0, 255) if isUsed else (0, 255, 0)
    cv2.polylines(outFrame, [pts.reshape((-1, 1, 2))], True, sClr, 1)
  statsText = f"{occupiedCount}/{len(slots)}"
  (tW, tH), tB = cv2.getTextSize(statsText, 0, 1.2, 1)
  tX, tY = 20, 50
  cv2.rectangle(outFrame, (tX - 5, tY - tH - 5), (tX + tW + 5, tY + tB + 5), (255, 255, 255), -1)
  cv2.rectangle(outFrame, (tX - 5, tY - tH - 5), (tX + tW + 5, tY + tB + 5), (0, 0, 0), 1)
  cv2.putText(outFrame, statsText, (tX, tY), 0, 1.2, (0, 0, 0), 1)
  return outFrame
def processPlates(frame, vModel, pModel, ocr):
  outFrame = frame.copy()
  vOptions = {"verbose": False, "conf": 0.5, "device": gpuDevice}
  vResults = vModel(outFrame, **vOptions)
  if vResults and vResults[0].boxes is not None and len(vResults[0].boxes) > 0:
    for vBox in vResults[0].boxes:
      vX1, vY1, vX2, vY2 = map(int, vBox.xyxy[0])
      vCrop = frame[vY1:vY2, vX1:vX2]
      if vCrop.size == 0:
        continue
      pResults = pModel(vCrop, **vOptions)
      if pResults and pResults[0].boxes is not None and len(pResults[0].boxes) > 0:
        for pBox in pResults[0].boxes:
          pX1, pY1, pX2, pY2 = map(int, pBox.xyxy[0])
          gX1, gY1, gX2, gY2 = vX1 + pX1, vY1 + pY1, vX1 + pX2, vY1 + pY2
          pImg = vCrop[pY1:pY2, pX1:pX2]
          if pImg.size == 0:
            continue
          pGray = cv2.cvtColor(pImg, cv2.COLOR_BGR2GRAY)
          pLarge = cv2.resize(pGray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
          ocrOut = ocr.readtext(pLarge, detail=0)
          pText = ocrOut[0].upper() if ocrOut else "N/A"
          cv2.rectangle(outFrame, (gX1, gY1), (gX2, gY2), (0, 255, 0), 1)
          cv2.putText(outFrame, pText, (gX1, gY1 - 10), 0, 0.5, (255, 255, 255), 1)
  return outFrame
def runLive(vidPath, isPark, modelA=None, modelB=None, modelC=None):
  cap = cv2.VideoCapture(vidPath)
  if not cap.isOpened():
    return
  winName = f"{os.path.basename(vidPath)}"
  cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)
  slotHistory = []
  while cap.isOpened():
    ok, img = cap.read()
    if not ok:
      break
    if isPark:
      processed = processParking(img, modelA, slotHistory)
    else:
      processed = processPlates(img, modelA, modelB, modelC)
    cv2.imshow(winName, processed)
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break
  cap.release()
  cv2.destroyWindow(winName)
def main():
  if len(sys.argv) < 2:
    return
  mode = sys.argv[1]
  if mode == "26":
    if os.path.exists(video0) and os.path.exists(jsonFile):
      carModel = YOLO(vPath)
      runLive(video0, True, modelA=carModel)
  elif mode == "11":
    if os.path.exists(video1):
      vehicleModel = YOLO(pPath)
      plateModel = YOLO(pPath)
      ocrReader = easyocr.Reader(["en"])
      runLive(video1, False, modelA=vehicleModel, modelB=plateModel, modelC=ocrReader)
  cv2.destroyAllWindows()
main()