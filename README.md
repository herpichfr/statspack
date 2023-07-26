# StatsPack - Statistical Visualization Package

StatsPack is a Python package designed for statistical visualization. It provides functions to create density contour plots, confidence intervals, and other statistical visualizations directly related with percentiles of 2D distributions. It was specially desined to be lightweight and avoid memory overusage.

## Requirements
```
Python3.8+
numpy
scipy
matplotlib
colorlog
logging
datetime
```

## Installation

You can install the StatsPack package using pip:

```
pip install statspack
```

## Usage

Import the package in your Python code
```python
import statspack
```

### Bining Data for Contour Plots

```python
import numpy as np
import matplotlib.pyplot as plt

# Example data
x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)

# Create contour plot data
X, Y, Z = statspack.bining(x, y, z, nbins=10, xlim=(None, None), ylim=(None, None))

# Plot the contour
plt.contour(X, Y, Z)
plt.colorbar()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Contour Plot')
plt.show()
```

### Finding Confidence Intervals

```python
import numpy as np

# Example PDF data
hist_pdf = np.random.rand(100)

# Find confidence interval
confidence_interval = statspack.find_confidence_interval(hist_pdf, prc=0.95)
print(f"95% Confidence Interval: {confidence_interval}")
```

### Density Contour Plot

```python
import numpy as np
import matplotlib.pyplot as plt

# Example data
xdata = np.random.rand(100)
ydata = np.random.rand(100)
binsx = 10
binsy = 10

# Create density contour plot
contours, levels = statspack.density_contour(xdata, ydata, binsx, binsy, verbose=True)
plt.colorbar(contours[0], label='Density')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Density Contour Plot')
plt.show()
```

### Contour PDF

```python
import numpy as np
import matplotlib.pyplot as plt

# Example data
x_axis = np.random.rand(100)
y_axis = np.random.rand(100)

# Create contour PDF plot
contours = statspack.contour_pdf(x_axis, y_axis, nbins=10, percent=[10, 50, 90], colors=['blue', 'green', 'red'])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Contour PDF Plot')
plt.show()
```

These are some of the functions provided by the StatsPack package for statistical visualization. You can refer to the function documentation in the source code for more details on their parameters and usage.

## License

StatsPack is licensed under the [GNU General Public License v3.0](LICENSE). You can find the full text of the license in the `LICENSE` file included with the package.
