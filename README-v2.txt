TurnSense v2 (development copy). The frozen reference is ..\turnsense-BASE-90.1 (checkpoint Core 90.1).
Change from v1: a neural speech check (Silero VAD v6, MIT, ONNX, CPU) must agree before a frame counts as the caller
taking the floor; evidence retuned (STOP 400->350 ms, gap 900->1200 ms, decay 0.985->0.997, VAD gate 0.3).
Offline, official scorer:            BASE  ->  v2
  dev (24 calls)                     90.2  ->  92.7
  xdev (25, may be inspected)        82.3  ->  95.0
  holdout (25, scored once)          80.1  ->  89.2
  worst over 0/-12/+6 dB mic gain    80.0  ->  89.7
If onnxruntime or the model is missing, my_system.py automatically runs the exact v1 BASE settings.
