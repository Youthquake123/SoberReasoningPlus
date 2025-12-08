# SoberReasoning+

```bash
git clone --recurse-submodules https://github.com/Youthquake123/SoberReasoningPlus.git
conda create -n sober python=3.12
conda activate sober
bash install.sh
cd lighteval && pip install -e . && cd ..
```

If flash-attn installation fails due to older Linux system version, refer to: https://github.com/Dao-AILab/flash-attention/issues/1708
