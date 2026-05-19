import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO, settings


def convert_visdrone_to_yolo(visdrone_base_path):
    """学术级/工业级 VisDrone 标签一键极速转换为标准 YOLO 格式"""
    print("[*] 正在启动标签格式深度清洗引擎...")

    # 定义需要转换的子集
    subsets = ['VisDrone2019-DET-train', 'VisDrone2019-DET-val']

    for subset in subsets:
        subset_path = visdrone_base_path / subset
        img_dir = subset_path / 'images'
        label_dir = subset_path / 'labels'  # 注意：此时应为重命名后的 labels 文件夹

        if not label_dir.exists() or not img_dir.exists():
            continue

        print(f"[*] 正在清洗 {subset} 的标签数据...")

        # 遍历目录下所有的 txt 标注文件
        txt_files = list(label_dir.glob('*.txt'))
        converted_count = 0

        for txt_file in txt_files:
            # 检查是否已经是符合规范的 YOLO 格式（通过首行分隔符判断，防止重复转换）
            with open(txt_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
            if first_line and ' ' in first_line and ',' not in first_line:
                # 已经是空格分隔，说明已经转换过，跳过当前文件
                continue

            # 获取对应图片的物理尺寸用来计算归一化
            img_path = img_dir / f"{txt_file.stem}.jpg"
            if not img_path.exists():
                continue

            with Image.open(img_path) as img:
                img_w, img_h = img.size

            # 读取天大原版格式并转换
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            yolo_lines = []
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue

                # 提取原始类别 (天大从1开始计数，YOLO需要从0开始)
                cls_id = int(parts[5]) - 1
                # 过滤掉忽略区域 (0) 以及一些不关心的特殊标注
                if cls_id < 0 or int(parts[4]) == 0:
                    continue

                # 提取绝对像素级坐标 [左, 上, 宽, 高]
                l, t, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])

                # 【核心数学映射】：转换为符合 YOLO 规范的归一化中心点相对坐标
                x_center = (l + w / 2.0) / img_w
                y_center = (t + h / 2.0) / img_h
                width = w / img_w
                height = h / img_h

                # 格式化输出：空格分隔
                yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # 覆写原 txt 文件
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines))
            converted_count += 1

        if converted_count > 0:
            print(f"[+] 成功将 {converted_count} 个天大标注转换为标准 YOLO 空格格式！")
    print("[+] 标签物理层格式格式化完毕，已完美对齐 YOLO 核心训练引擎。")


def start_drone_training():
    # ==========================================
    # 1. 路径严格锁死与配置重置
    # ==========================================
    project_root = Path(__file__).resolve().parent
    dataset_dir = project_root / "datasets"
    visdrone_path = dataset_dir / "VisDrone"

    settings.update({"datasets_dir": str(dataset_dir.as_posix())})
    print(f"[+] 数据集检索根目录已硬锁定: {dataset_dir}")

    if not visdrone_path.exists():
        print(f"[!] 错误：找不到 {visdrone_path} 目录！")
        return

    # ==========================================
    # 2. 强制执行本地格式清洗（破坑核心）
    # ==========================================
    # 清理先前运行失败产生的错误缓存文件，迫使 YOLO 重新扫描
    for cache_name in ['labels.cache', 'VisDrone2019-DET-train/labels.cache', 'VisDrone2019-DET-val/labels.cache']:
        cache_path = visdrone_path / cache_name
        if cache_path.exists():
            os.remove(cache_path)
            print(f"[*] 已清理历史错误缓存: {cache_name}")

    # 调用转换清洗函数
    convert_visdrone_to_yolo(visdrone_path)

    # ==========================================
    # 3. 生成对齐官方检索拓扑的配置文件
    # ==========================================
    yaml_content = f"""
path: {visdrone_path.as_posix()}
train: VisDrone2019-DET-train/images
val: VisDrone2019-DET-val/images

names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
"""
    yaml_path = project_root / "visdrone_drone.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content.strip())
    print(f"[+] 战术配置文件已重新校验生成: {yaml_path}")

    # ==========================================
    # 4. 算力全开，直接启动训练
    # ==========================================
    print("[*] 正在载入预训练 Backbone (yolov8m.pt)...")
    model = YOLO("yolov8m.pt")

    print("[*] 显卡就位（RTX 4060 8GB），开始注入 DFL+Mosaic 专项巡检算子...")
    model.train(
        data=str(yaml_path),
        epochs=200,
        imgsz=640,
        batch=16,  # 如果 4060 还是报 OOM，改这里为 8
        workers=4,
        device=0,

        # 论文核心增强参数
        mosaic=1.0,
        mixup=0.15,
        degrees=15.0,
        perspective=0.0001,
        box=7.5,
        dfl=2.5,

        lr0=0.01,
        lrf=0.01,
        val=True,
        plots=True,
        save=True,
        project="drone_project",
        name="yolov8m_optimized"
    )


if __name__ == '__main__':
    start_drone_training()