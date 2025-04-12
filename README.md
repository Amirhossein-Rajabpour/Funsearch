# Funsearch for Generating Can't Stop Strategies

Funsearch implementation adapted from https://github.com/google-deepmind/funsearch.

Can't Stop implementation adapted from https://github.com/zahrabashir98/Cant_Stop.

To run the program in the background (detached from terminal):
```
nohup python3 funsearch.py >/dev/null 2>&1 &
```

To stop it, look up the PID:
```
ps aux | grep funsearch
```

Terminate it:
```
kill -9 <PID>
```
