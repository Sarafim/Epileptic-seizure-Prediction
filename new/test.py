# import matplotlib.pyplot as plt
# data = [0 for i in range(88)]
# for i in range(63):
#     data.append(1)
# for i in range(47):
#     data.append(2)
# for i in range(49):
#     data.append(3)
# for i in range(52):
#     data.append(4)
# for i in range(55):
#     data.append(5)
# for i in range(30):
#     data.append(6)
# for i in range(48):
#     data.append(7)
# plt.hist(data, bins=[0, 1, 2, 3,4,5,6,7])
# # plt.grid(True)
# plt.show()
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

ind = np.arange(8)                # the x locations for the groups
width = 0.7                      # the width of the bars
menMeans = [88,63,47,49,52,55,30,48]
## the bars
rects1 = ax.bar(ind, menMeans, width, color='green')
ax.set_ylabel('Точність передбачення, %', fontsize=15)
ax.set_xlabel('Власні числа в порядку зростання', fontsize=15)
# ax.grid(True)
# plt.show()

path_result = 'D:\Diploma\eigvalues.png'

fig.savefig(path_result)

plt.close(fig)