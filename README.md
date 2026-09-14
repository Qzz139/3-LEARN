
 ## 原始任务要求
 <img width="527" height="881" alt="image" src="https://github.com/user-attachments/assets/ef595745-81d9-4c13-b8ec-4079ec06ba68" />

核心设备：robomaster ep + jetson orinX
设定两类物体：网球+550ml标准水瓶

实现方案要点（建设中ing)：
1.机械臂初始姿态为摄像机位置服务，确保在一个视野良好，识别率和识别范围都舒适的角度和位置
2.机械臂大部分时候处于两个关节均回收的“收回”状态（包括但不限于零态、抓取到物品后的底盘旋转、移动）。仅在取物品时伸出
3.底盘的初始位置距物品较远，为摄像机有较好的位置。
