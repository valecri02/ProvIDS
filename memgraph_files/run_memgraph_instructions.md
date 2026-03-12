## 1) Activate your conda environment

```
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate ctdg_pyg
```
## 2) Install Python packages needed by the uploader

Inside ctdg_pyg:
```
conda install -c conda-forge -y natsort dask pandas tqdm numpy pyarrow
pip install gqlalchemy
```
Quick import test:
```
python -c "from gqlalchemy import Memgraph; import dask.dataframe as dd; import pandas as pd; from natsort import natsorted; from tqdm import tqdm; import numpy as np; print('imports ok')"
```
## 3) Start the Memgraph container (theia)

First remove any old container with the same name:
```
docker rm -f theia 2>/dev/null
```
Run Memgraph and mount your dataset parent folder (encoding) to /data:
```
docker run -itd --name theia \
  -p 7689:7687 -p 7446:7444 -p 3002:3000 \
  -v /Users/valentinacristoferi/THESIS/ProvIDS/darpa_preprocessing/data/encoding:/data \
  memgraph/memgraph-platform:2.6.5-memgraph2.5.2-lab2.4.0-mage1.6
```
Verify the container sees the data:
```
docker exec -it theia ls -lah /data/e3_theia
```
## 4) Run the upload script correctly

Important: the script requires --type dataset.

From the folder containing upload_to_memgraph.py (your src folder), run:
```
python -u upload_to_memgraph.py \
  --type dataset \
  --dataset_local /Users/valentinacristoferi/THESIS/ProvIDS/darpa_preprocessing/data/encoding/e3_theia \
  --dataset_memgraph /data/e3_theia \
  --port 7689 \
  --redo
```


## 5) Verify Memgraph has data

Open mgconsole:
```
docker exec -it theia mgconsole
```
Run:
```
MATCH (n) RETURN count(n);
MATCH ()-[r]->() RETURN count(r);
```
(You got: 22225 nodes, 28295 edges ✅)

Exit:
`:quit`
## 6) Open the UI and query

Because you mapped -p 3002:3000, open:

MAGE UI: http://localhost:3002

Example query:
```
MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 1000;
```
Tip: to see a “bigger” coherent subgraph, start from a high-degree node:


7) (Optional) Stop / delete container later

Stop:
```
docker stop theia
```
Delete:
```
docker rm theia
```
Force delete:
```
docker rm -f theia
```