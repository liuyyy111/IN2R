from evaluation import evalrank, ensemble_evalrank
import os
import logging

# 创建 logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # 设置全局日志级别

# 创建日志格式
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
# 创建控制台处理器
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# 将处理器添加到 logger
logger.addHandler(console_handler)

# evalrank(os.path.join("output/2025_12_11_23_02_17", "model_best.pth.tar"), split="test")
# evalrank(os.path.join("output/l2rm_seed_0.2_2025_12_12_12_19_15", "model_best.pth.tar"), split="test")
# evalrank(os.path.join("output/l2rm_seed_0.2_infonce_2025_12_12_15_36_57", "model_best.pth.tar"), split="test")
# evalrank(os.path.join("output/2025_12_17_23_29_57", "model_best.pth.tar"), split="test")
# evalrank(os.path.join("output/2025_12_17_23_00_49", "model_best.pth.tar"), split="test")
# evalrank(os.path.join("output/2025_12_17_", "model_best.pth.tar"), split="test")

# ensemble_evalrank(os.path.join("output/0.8/cl_contrast_ti2", "model_best.pth.tar"), os.path.join("output/0.8/cl_contrast_i2t", "model_best.pth.tar"), split="test")
# evalrank(os.path.join("output/0.2/coco/i2tandt2i_k=5_drop0.3_wo_innermodal", "model_best.pth.tar"), split="testall", fold5=True)
# evalrank(os.path.join("output/0.2/coco/i2tandt2i_k=5_drop0.3_wo_innermodal", "model_best.pth.tar"), split="testall", fold5=False)

# for i in range(21, 27):
# for i in range(11, 25):
for i in range(15, 40):
    # evalrank(f"output/cc152k/final_drop0.3_448000/checkpoint_{i}.pth.tar", split="test")
    # evalrank(f"output/0.8/coco/final_/checkpoint_{i}.pth.tar", split="testall", fold5=True)
    evalrank(f"output/0.8/f30k/final_/checkpoint_{i}.pth.tar", split="test")

# cc152k
# 2026-01-20 13:16:01,958 - INFO - rsum: 380.8
# 2026-01-20 13:16:01,958 - INFO - Average i2t Recall: 63.7
# 2026-01-20 13:16:01,958 - INFO - Image to text: 45.2 69.0 76.8 2.0 20.9
# 2026-01-20 13:16:01,958 - INFO - Average t2i Recall: 63.3
# 2026-01-20 13:16:01,958 - INFO - Text to image: 43.3 68.3 78.2 2.0 23.5
# 2026-01-20 13:16:02,361 - INFO - training epoch: 38

# 2026-01-20 13:21:29,859 - INFO - rsum: 381.3
# 2026-01-20 13:21:29,859 - INFO - Average i2t Recall: 63.6
# 2026-01-20 13:21:29,859 - INFO - Image to text: 44.7 68.7 77.4 2.0 20.1
# 2026-01-20 13:21:29,859 - INFO - Average t2i Recall: 63.5
# 2026-01-20 13:21:29,859 - INFO - Text to image: 43.9 68.4 78.2 2.0 22.6

# 0.2
# rsum: 506.5
# Average i2t Recall: 90.1
# Image to text: 78.8 94.2 97.4 1.0 2.3
# Average t2i Recall: 78.7
# Text to image: 59.4 85.5 91.3 1.0 6.3

# 2026-01-07 13:35:40,815 - INFO - rsum: 504.0
# 2026-01-07 13:35:40,816 - INFO - Average i2t Recall: 89.9
# 2026-01-07 13:35:40,816 - INFO - Image to text: 78.2 94.3 97.2 1.0 2.2
# 2026-01-07 13:35:40,816 - INFO - Average t2i Recall: 78.1
# 2026-01-07 13:35:40,816 - INFO - Text to image: 58.3 85.0 90.9 1.0 7.2


# dropout 0.4
# 2026-01-07 19:25:27,639 - INFO - rsum: 505.6
# 2026-01-07 19:25:27,639 - INFO - Average i2t Recall: 90.1
# 2026-01-07 19:25:27,639 - INFO - Image to text: 78.2 94.9 97.2 1.0 2.3
# 2026-01-07 19:25:27,639 - INFO - Average t2i Recall: 78.4
# 2026-01-07 19:25:27,639 - INFO - Text to image: 59.0 85.2 91.1 1.0 6.3


# 0.8
# rsum: 438.4
# Average i2t Recall: 80.2
# Image to text: 63.0 85.9 91.8 1.0 4.9
# Average t2i Recall: 65.9
# Text to image: 43.6 72.5 81.6 2.0 14.3

# drop 0.3
# 2026-01-07 17:10:37,672 - INFO - rsum: 440.6
# 2026-01-07 17:10:37,672 - INFO - Average i2t Recall: 80.4
# 2026-01-07 17:10:37,673 - INFO - Image to text: 63.2 85.8 92.2 1.0 5.0
# 2026-01-07 17:10:37,673 - INFO - Average t2i Recall: 66.5
# 2026-01-07 17:10:37,673 - INFO - Text to image: 44.5 73.2 81.7 2.0 14.3

# drop 0.4
# 2026-01-07 18:11:35,856 - INFO - rsum: 440.6
# 2026-01-07 18:11:35,857 - INFO - Average i2t Recall: 80.3
# 2026-01-07 18:11:35,857 - INFO - Image to text: 62.7 86.3 91.9 1.0 4.9
# 2026-01-07 18:11:35,857 - INFO - Average t2i Recall: 66.6
# 2026-01-07 18:11:35,857 - INFO - Text to image: 44.4 73.0 82.4 2.0 13.4


# 2026-01-07 21:38:24,639 - INFO - rsum: 438.2
# 2026-01-07 21:38:24,639 - INFO - Average i2t Recall: 79.4
# 2026-01-07 21:38:24,639 - INFO - Image to text: 61.9 85.5 90.8 1.0 4.8
# 2026-01-07 21:38:24,639 - INFO - Average t2i Recall: 66.7
# 2026-01-07 21:38:24,639 - INFO - Text to image: 44.2 73.1 82.6 2.0 13.6

# 0.8   
# 2026-01-07 23:07:05,535 - INFO - rsum: 444.0
# 2026-01-07 23:07:05,535 - INFO - Average i2t Recall: 80.5
# 2026-01-07 23:07:05,535 - INFO - Image to text: 62.2 87.1 92.1 1.0 4.9
# 2026-01-07 23:07:05,535 - INFO - Average t2i Recall: 67.5
# 2026-01-07 23:07:05,535 - INFO - Text to image: 45.7 73.7 83.2 2.0 12.9