FOR /F "tokens=1-5 delims=," %%A IN (configs.txt)^
DO python nn.py --max-iter=%%A --Alpha=%%B --Beta=%%C --suffix=%%D --hidden-size=%%E
