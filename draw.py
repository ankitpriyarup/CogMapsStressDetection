import numpy as np
import matplotlib.pyplot as plt

height = [
    0.022105, 0.022105, 0.022105, 0.022105, 0.022105, 0.012773, 0.010977,
    0.146086, 0.146086, 0.146086, 0.146086, 0.146086, 0, 0, -0.005485,
    -0.005485, -0.005485, -0.005485, -0.005485, -0.021490, 0.032440, 0.004838,
    0.003911, 0.004330, 0.004385, 0.003905, -0.012335, -0.005656
]

bars = ('EDA_MAX', 'EDA_MIN', 'EDA_MEAN', 'EDA_MEDIAN', 'EDA_MODE', 'EDA_SD',
        'EDA_DIFF', 'HR_MAX', 'HR_MIN', 'HR_MEAN', 'HR_MEDIAN', 'HR_MODE',
        'HR_SD', 'HR_DIFF', 'SKT_MAX', 'SKT_MIN', 'SKT_MEAN', 'SKT_MEDIAN',
        'SKT_MODE', 'SKT_SD', 'SKT_DIFF', 'BVP_MAX', 'BVP_MIN', 'BVP_MEAN',
        'BVP_MEDIAN', 'BVP_MODE', 'BVP_SD', 'BVP_DIFF')

zipped = list(zip(height, bars))
res = sorted(zipped, key=lambda x: x[0], reverse=True)
height = []
bars = []
for i in res:
    height.append(i[0])
    bars.append(i[1])
print(height)
y_pos = np.arange(len(bars))

plt.bar(y_pos, height, color=(0.5, 0.1, 0.5, 0.6))

plt.xlabel('Features')
plt.ylabel('Corelation with respect to STATE')

plt.ylim(-0.05, 0.2)

plt.xticks(y_pos, bars, rotation='vertical', fontsize=8)
plt.tight_layout()

plt.savefig('histogram.png', dpi=200)
