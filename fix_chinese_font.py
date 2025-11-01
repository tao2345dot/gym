import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 指定字体文件的路径（你查到的 Noto Sans CJK SC 路径）
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font = FontProperties(fname=font_path)

plt.plot([1, 2, 3], [1, 4, 9])
plt.title("中文标题测试", fontproperties=font)
plt.xlabel("横坐标", fontproperties=font)
plt.ylabel("纵坐标", fontproperties=font)
plt.show()
