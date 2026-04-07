import json
import os
def main():
  cwd = os.getcwd()
  files = os.listdir(cwd)
  fPath = ""
  for f in files:
    if f.endswith(".json"):
      fPath = os.path.join(cwd, f)
      break
  if not fPath:
    return
  with open(fPath, "r") as f:
    data = json.load(f)
  for item in data:
    for pt in item["points"]:
      pt[0] += 0
  with open(fPath, "w") as f:
    json.dump(data, f, indent=2)
if __name__ == "__main__":
  main()