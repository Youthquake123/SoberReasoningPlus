# SoberReasoning+

```bash
git clone --recurse-submodules https://github.com/Youthquake123/SoberReasoningPlus.git
conda create -n sober python=3.12
conda activate sober
bash install.sh
cd lighteval && pip install -e . && cd ..
```
Teperature 0, 0.2, 0.4, 0.6, 0.8 1.0

Top_p: 0.7, 0.8, 0.9, 0.95, 1.0

Small dataset: 16 seeds

Big datasets: 3 seeds

If flash-attn installation fails due to older Linux system version, refer to: https://github.com/Dao-AILab/flash-attention/issues/1708
