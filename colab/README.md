# Google Colab

Install this repository without cloning (replace URL and branch):

```python
!pip install "git+https://github.com/<YOUR_GITHUB_USER>/FIND_Delft_Assignment.git@main"
```

Then in Python:

```python
import find_delft_assignment
from find_delft_assignment.config import SamplingConfig

print(find_delft_assignment.__version__)
```

Upload your checkpoint to Colab (or mount Drive) and pass its path into your notebook once sampling is implemented.

Torch on Colab is preinstalled; if you need a specific `torch` build for your checkpoint, pin it in a notebook cell before importing your model code.
