import csv

def calc_rois_gray_to_csv(rois_txt, src_dir, save_csv):
    import os
    import cv2
    # 读取ROI列表
    rois = []
    with open(rois_txt, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 4:
                rois.append(tuple(map(int, parts)))
    # 获取所有jpg图片
    files = [f for f in os.listdir(src_dir) if f.lower().endswith('.jpg')]
    files.sort()
    # 写入csv
    with open(save_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for fname in files:
            img_path = os.path.join(src_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                row = ['error']*len(rois)
                writer.writerow(row)
                continue
            row = []
            for x, y, w, h in rois:
                roi = img[y:y+h, x:x+w]
                if roi.size == 0:
                    row.append('nan')
                else:
                    row.append(int(roi.mean()))
            writer.writerow(row)

import cv2

def select_rois_and_save(file, save_path, regioncounts=1):
    # 读取图片
    img = cv2.imread(file)
    if img is None:
        print("Failed to load image:", file)
        return

    # 选择 ROI，返回 (x, y, w, h) 列表
    rois = cv2.selectROIs("Select ROIs", img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    # 只保留前 regioncounts 个 ROI
    rois = rois[:regioncounts]

    # 保存坐标到文件
    with open(save_path, 'w') as f:
        for roi in rois:
            x, y, w, h = roi
            f.write(f"{x},{y},{w},{h}\n")

def hello_world():
    print("Hello, world!")

def rename_file_in_directory(dir):
    import os
    import pathlib
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    # 获取文件的修改时间并排序
    files_with_mtime = [(f, os.path.getmtime(os.path.join(dir, f))) for f in files]
    files_with_mtime.sort(key=lambda x: x[1])
    num_files = len(files)
    width = len(str(num_files))
    for idx, (filename, _) in enumerate(files_with_mtime, 1):
        ext = pathlib.Path(filename).suffix
        new_name = f"{str(idx).zfill(width)}{ext}"
        src = os.path.join(dir, filename)
        dst = os.path.join(dir, new_name)
        os.rename(src, dst)
    return num_files
    
def rename_and_compare(pic_path, spec_path):
    print("Renaming files in 2 directories...")
    pic_count = rename_file_in_directory(pic_path)
    spec_count = rename_file_in_directory(spec_path)
    print(f"{pic_path} file count: {pic_count}")
    print(f"{spec_path} file count: {spec_count}")
    print(f"Are the file counts equal: {pic_count == spec_count}")

def remove_lines_and_save(file_path, save_dir, num_lines=14):
    import os
    num_lines = int(num_lines)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='gbk') as f:
            lines = f.readlines()
    new_lines = lines[num_lines:]
    base_name = os.path.basename(file_path)
    save_path = os.path.join(save_dir, base_name)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

def get_files_in_directory(directory):
    import os
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def remove_lines(source_dir, save_dir, num_lines=14):
    import os
    num_lines = int(num_lines)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    files = get_files_in_directory(source_dir)
    for file in files:
        file_path = os.path.join(source_dir, file)
        remove_lines_and_save(file_path, save_dir, num_lines)

def recaculate_to_int_file(file, save_dir):
    import os
    from collections import defaultdict
    base_name = os.path.basename(file)
    save_path = os.path.join(save_dir, base_name)
    groups = defaultdict(list)
    # 读取数据并分组
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                key = int(float(parts[0]))
                value = float(parts[1])
                groups[key].append(value)
            except Exception:
                continue
    # 计算平均值并写入新文件
    with open(save_path, 'w', encoding='utf-8') as f:
        for key in sorted(groups.keys()):
            avg = int(sum(groups[key]) / len(groups[key]))
            f.write(f"{key}\t{avg}\n")

def recalculate_to_int(source_dir, save_dir):
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    files = get_files_in_directory(source_dir)
    for file in files:
        file_path = os.path.join(source_dir, file)
        recaculate_to_int_file(file_path, save_dir)

def cut_to_visible_range_file(file, save_dir, min_range=400, max_range=800):
    import os
    base_name = os.path.basename(file)
    save_path = os.path.join(save_dir, base_name)
    with open(file, 'r', encoding='utf-8') as fin, open(save_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 1:
                continue
            try:
                value = float(parts[0])
            except Exception:
                continue
            if min_range <= value <= max_range:
                fout.write(line)

def cut_to_visible_range(source_dir, save_dir, min_range=400, max_range=800):
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    files = get_files_in_directory(source_dir)
    for file in files:
        file_path = os.path.join(source_dir, file)
        cut_to_visible_range_file(file_path, save_dir, min_range, max_range)

def select_regions(file, save_path, regioncounts=1):
    # 读取图片
    img = cv2.imread(file)
    if img is None:
        print("Failed to load image:", file)
        return

    # 交互式选择ROI，强制只选regioncounts个
    rois = []
    for i in range(regioncounts):
        roi = cv2.selectROI(f"Select ROI {i+1}/{regioncounts}", img, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        if roi == (0,0,0,0):
            break
        rois.append(roi)

    # 保存坐标到文件
    with open(save_path, 'w') as f:
        for roi in rois:
            x, y, w, h = roi
            f.write(f"{x},{y},{w},{h}\n")

def spectrum_to_csv(src_dir, save_csv):
    import os
    import csv
    files = [f for f in os.listdir(src_dir) if f.lower().endswith('.txt')]
    # 按文件名中的数字排序（假设文件名如0001.txt）
    files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    spectra = []
    for fname in files:
        path = os.path.join(src_dir, fname)
        values = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        values.append(int(float(parts[1])))
                    except Exception:
                        values.append('nan')
        # 只保留401个数据
        values = values[:401]
        # 若不足401列则补nan
        if len(values) < 401:
            values += ['nan'] * (401 - len(values))
        spectra.append(values)
    # 写入csv
    with open(save_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in spectra:
            writer.writerow(row)
    

