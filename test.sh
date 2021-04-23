wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda config --set auto_update_conda no
conda update -q conda
git clone https://github.com/pyomeca/bioptim.git
cd bioptim
conda env update -n root -f environment.yml
conda install scp -cconda-forge
python setup.py install
cd ..
git clone https://github.com/Steakkk/JumperOCP
cd JumperOCP/optimization_biorbdOptim/
ln -s ../../bioptim/bioptim
python script.py
