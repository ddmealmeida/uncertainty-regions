# uncertainty-regions

Repository for a tool that identifies the regions with most and least uncertainty for a given Machine Learning model's output.

## interface

- Install requirements
- Run `python main.py ../configs.txt` inside de `interface/` folder

## TODO

- Use more descriptive names in 2d subgroups plot (name of the classe sinstead of numbers of the classes: maybe extra column?)
- Dendrogram is still negative iin x_axis
- Venn diagram doesn't have a builtin method inside plotly. Use some kinf of conversion from matplotlib
- Refactor code: create more files to abstract function and change variable names to better ones
