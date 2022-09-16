# spyglass-dlc
Position pipeline to incorporate DeepLabCut into Spyglass

### Usage

### Developer Installation
1. Install miniconda (or anaconda) if it isn't already installed.
2. INSTALL CORRECT DRIVER for GPU, see https://stackoverflow.com/questions/30820513/what-is-the-correct-version-of-cuda-for-my-nvidia-driver/30820690
3. git clone https://github.com/LorenFrankLab/spyglass-dlc.git
4. Setup editiable package with dependencies
```bash
cd spyglass-dlc
conda env create -f environment.yml
conda env update -f environment.yaml
```
after installing do the following:
```bash
conda activate spyglass-dlc
conda env config vars set LD_LIBRARY_PATH=~/path/to/anaconda3/envs/spyglass-dlc/lib/
mamba install -c conda-forge wxpython
mamba install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
cd spyglass
pip install -e .
cd spyglass-dlc
pip install -e .
```
Add the following environment variables (e.g. in ~/.bashrc). If from the Frank Lab, we assume that you are interacting with the database on a computer that has mounted nimbus at /nimbus (if the mount location is different, change accordingly). For this to take effect, log out and log back in, or run source ~/.bashrc in the terminal.
```bash
nano ~/.bashrc
```
and add:
```bash
export DLC_PROJECT_PATH="/nimbus/deeplabcut/projects"
export DLC_VIDEO_PATH="/nimbus/deeplabcut/video"
export DLC_OUTPUT_PATH="/nimbus/deeplabcut/output"
```

