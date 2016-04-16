import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys
import os
model_name = sys.argv[1]

nbfn = 'Visualize_blueprint.ipynb'

with open(nbfn) as f:
    nb = nbformat.read(f,as_version=4)

nb['cells'][0]['source'] = u'# {} model visualisation'.format(model_name)
nb['cells'][2]['source'] = nb['cells'][2]['source'].replace('generic_model',model_name)

ep = ExecutePreprocessor(timeout=1800,kernel_name='python2')

ep.preprocess(nb,{})

with open('executed_nb.ipynb',mode='wt') as f:
    nbformat.write(nb,f)

os.system('jupyter nbconvert executed_nb.ipynb')
os.system('mv executed_nb.html results/{}.html'.format(model_name))
