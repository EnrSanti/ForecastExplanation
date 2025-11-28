echo "Creating conda environment 'gpu_tobac'..."
conda create --name gpu_tobac python=3.10 -y

# Activate environment
echo "Activating environment..."
conda activate gpu_tobac || source ~/miniconda3/etc/profile.d/conda.sh && conda activate gpu_tobac

# Install GPU PyTorch
echo "Installing PyTorch + CUDA..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Install other libraries
echo "Installing supporting libraries..."
conda install -c conda-forge pillow scikit-image numpy matplotlib scikit-learn opencv numba seaborn -y

# Install TOBAC
echo "Installing TOBAC..."
conda install -c conda-forge tobac -y

# Install cfgrib for reading GRIB files
echo "Installing cfgrib..."
conda install -c conda-forge cfgrib -y

echo "Setup complete! To use the environment: conda activate gpu_tobac and then python main.py"

