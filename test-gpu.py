# import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)
# else:
#     print("No GPUs found")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

samples = tf.random.normal([1, 100])
print(samples)
# 繪製直方圖
plt.hist(samples, bins=30, density=True, alpha=0.6, color='gray')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Normal Distribution')
# 儲存成圖片
plt.savefig('visual_result/gan-normal_distribution_histogram.png', bbox_inches='tight', pad_inches=0)

# 顯示圖片（可選）
plt.show()
