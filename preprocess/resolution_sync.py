import os
import shutil
import imageio
import numpy as np
from glob import glob
from tqdm import tqdm
from argparse import ArgumentParser
from PIL import Image
import tempfile
import json
from datetime import datetime  # 添加在文件开头的导入部分

def analyze_aspect_ratios(img_dirs):
    """ 分析场景中的宽高比分布（替换原有validate） """
    aspect_stats = {}
    for img_dir in img_dirs:
        image_files = glob(os.path.join(img_dir, '*.png'))
        for img_file in image_files[:10]:  # 增加采样数量
            with Image.open(img_file) as img:
                w, h = img.size
                aspect = round(w / h, 3)
                aspect_stats[aspect] = aspect_stats.get(aspect, 0) + 1
    return aspect_stats

def backup_original(img_file, backup_dir):
    """ 增强版备份函数 """
    try:
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, os.path.basename(img_file))
        # 使用硬链接节省空间（同分区时）
        if not os.path.exists(backup_path):
            try:
                os.link(img_file, backup_path)
            except OSError:
                shutil.copy2(img_file, backup_path)
    except Exception as e:
        print(f"备份失败: {img_file} -> {backup_dir} - {str(e)}")

def resize_with_padding(img, target_size):
    """ 保持宽高比的缩放并添加填充 """
    original_width, original_height = img.size
    target_width, target_height = target_size
    
    # 计算缩放比例
    ratio = min(target_width/original_width, target_height/original_height)
    new_size = (int(original_width*ratio), int(original_height*ratio))
    
    # 缩放图像
    img = img.resize(new_size, Image.LANCZOS)
    
    # 创建新画布
    new_img = Image.new(img.mode, target_size, (0, 0, 0))
    
    # 计算粘贴位置
    paste_pos = (
        (target_width - new_size[0]) // 2,
        (target_height - new_size[1]) // 2
    )
    
    new_img.paste(img, paste_pos)
    return new_img

def get_dynamic_target_size(img_dirs, strategy='max'):
    """ 动态获取目标尺寸（支持多种策略） """
    sizes = []
    for img_dir in img_dirs:
        image_files = glob(os.path.join(img_dir, '*.png'))
        for img_file in image_files[:5]:  # 采样检查
            with Image.open(img_file) as img:
                sizes.append(img.size)
    
    # 策略选择
    if strategy == 'max':
        target_w = max(s[0] for s in sizes)
        target_h = max(s[1] for s in sizes)
    elif strategy == 'min':
        target_w = min(s[0] for s in sizes)
        target_h = min(s[1] for s in sizes)
    elif strategy == 'median':
        target_w = int(np.median([s[0] for s in sizes]))
        target_h = int(np.median([s[1] for s in sizes]))
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    
    return (target_w, target_h)

def adaptive_resize(img, target_size, mode='padding'):
    """ 增强版缩放函数（支持多种模式） """
    orig_w, orig_h = img.size
    target_w, target_h = target_size
    
    if mode == 'padding':
        return resize_with_padding(img, target_size)
    elif mode == 'crop':
        # 计算裁剪比例
        scale = max(target_w/orig_w, target_h/orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # 缩放后裁剪
        img = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        return img.crop((left, top, left+target_w, top+target_h))
    elif mode == 'scale':
        # 简单缩放
        return img.resize(target_size, Image.LANCZOS)
    else:
        raise ValueError(f"未知的缩放模式: {mode}")

def update_camera_params_v2(camera_dir, original_size, processed_size, mode='padding'):
    """ 增强版参数更新（支持多种处理模式） """
    param_file = os.path.join(camera_dir, 'intrinsics.txt')
    if not os.path.exists(param_file):
        return

    with open(param_file, 'r') as f:
        params = {k: float(v) for line in f for k, v in [line.strip().split(': ')]}

    orig_w, orig_h = original_size
    target_w, target_h = processed_size
    pad_x, pad_y = 0, 0

    # 计算缩放比例
    if mode == 'padding':
        ratio = min(target_w/orig_w, target_h/orig_h)
        new_w = int(orig_w * ratio)
        new_h = int(orig_h * ratio)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
    elif mode == 'crop':
        ratio = max(target_w/orig_w, target_h/orig_h)
    elif mode == 'scale':
        ratio_w = target_w / orig_w
        ratio_h = target_h / orig_h
    else:
        raise ValueError(f"不支持的参数模式: {mode}")

    # 更新参数
    if mode in ['padding', 'crop']:
        params['fx'] *= ratio
        params['fy'] *= ratio
        params['cx'] = params['cx'] * ratio + pad_x
        params['cy'] = params['cy'] * ratio + pad_y
    elif mode == 'scale':
        params['fx'] *= ratio_w
        params['fy'] *= ratio_h
        params['cx'] *= ratio_w
        params['cy'] *= ratio_h

    # 保存参数并记录元数据
    meta = {
        'original_size': f"{orig_w}x{orig_h}",
        'processed_mode': mode,
        'scale_ratio': ratio if mode != 'scale' else f"{ratio_w:.2f}x{ratio_h:.2f}",
        'padding': f"{pad_x},{pad_y}"
    }
    
    with open(param_file, 'w') as f:
        f.write(f"# Meta Data: {json.dumps(meta)}\n")
        for k, v in params.items():
            f.write(f"{k}: {v:.6f}\n")

def process_scene(scene_dir, rgb_dirname, camera_dirs):
    """ 兼容不同宽高比的场景处理（修复跨设备问题版） """
    # 在每个场景目录创建临时文件夹（可选）
    scene_temp_dir = os.path.join(scene_dir, '.processing_temp')
    os.makedirs(scene_temp_dir, exist_ok=True)

    img_dirs = [os.path.join(scene_dir, rgb_dirname, cam) 
                for cam in camera_dirs 
                if os.path.exists(os.path.join(scene_dir, rgb_dirname, cam))]
    
    if not img_dirs:
        return

    try:
        # 步骤1：分析场景特性
        aspect_stats = analyze_aspect_ratios(img_dirs)
        print(f"场景 {os.path.basename(scene_dir)} 宽高比分布: {aspect_stats}")

        # 步骤2：动态确定处理策略
        #strategy = 'max' if len(aspect_stats) == 1 else 'median'
        strategy= 'median'
        resize_mode = 'padding'

        # 步骤3：确定目标尺寸
        target_size = get_dynamic_target_size(img_dirs, strategy=strategy)
        print(f"目标处理尺寸: {target_size} | 模式: {resize_mode}")

        # 步骤4：处理每个相机
        for cam_dir in img_dirs:
            # 在相机目录创建专用临时文件夹
            cam_temp_dir = os.path.join(cam_dir, '.process_temp')
            os.makedirs(cam_temp_dir, exist_ok=True)

            # 获取原始参数（带安全检查）
            image_files = glob(os.path.join(cam_dir, '*.png'))
            if not image_files:
                print(f"跳过空相机目录: {cam_dir}")
                continue
                
            try:
                with Image.open(image_files[0]) as img:
                    original_size = img.size
            except Exception as e:
                print(f"无法获取初始尺寸: {cam_dir} - {str(e)}")
                continue

            # 处理图像（带跨设备安全）
            for img_file in tqdm(image_files, desc=f'Processing {os.path.basename(cam_dir)}'):
                try:
                    # 使用相机本地临时文件
                    with tempfile.NamedTemporaryFile(
                        dir=cam_temp_dir,  # 关键修改：确保同分区
                        suffix='.png',
                        delete=False
                    ) as tmp_file:
                        # 备份原始图像
                        backup_original(img_file, os.path.join(cam_dir, 'backup'))
                        
                        # 处理图像
                        with Image.open(img_file) as img:
                            processed_img = adaptive_resize(img, target_size, mode=resize_mode)
                            processed_img.save(tmp_file, format='PNG', quality=95, compress_level=3)
                        
                        # 跨设备安全替换
                        try:
                            os.replace(tmp_file.name, img_file)
                        except OSError as e:
                            if e.errno == 18:  # EXDEV: Cross-device link
                                shutil.copy2(tmp_file.name, img_file)
                                os.remove(tmp_file.name)
                            else:
                                raise
                            
                except Exception as e:
                    print(f"处理失败: {img_file} - {str(e)}")
                    # 清理残留临时文件
                    if 'tmp_file' in locals() and os.path.exists(tmp_file.name):
                        try:
                            os.remove(tmp_file.name)
                        except:
                            pass

            # 更新相机参数
            update_camera_params_v2(cam_dir, original_size, target_size, mode=resize_mode)
            
            # 清理临时目录
            shutil.rmtree(cam_temp_dir, ignore_errors=True)

        # 保存场景配置
        scene_config = {
            'target_size': target_size,
            'aspect_stats': aspect_stats,
            'processing_time': datetime.now().isoformat(),
            'device_id': os.stat(scene_dir).st_dev  # 记录存储设备ID
        }
        with open(os.path.join(scene_dir, 'processing_meta.json'), 'w') as f:
            json.dump(scene_config, f, indent=2)

        # 清理场景临时目录
        shutil.rmtree(scene_temp_dir, ignore_errors=True)

    except Exception as e:
        print(f"场景处理失败: {scene_dir} - {str(e)}")
        # 紧急清理临时文件
        shutil.rmtree(scene_temp_dir, ignore_errors=True)

if __name__ == "__main__":
    # ... [保留原有的参数解析代码]
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/waymo/processed/training')
    parser.add_argument(
        "--scene_ids",
        default=None,
        type=int,
        nargs="+",
        help="scene ids to be processed, a list of integers separated by space. Range: [0, 798] for training, [0, 202] for validation",
    )
    parser.add_argument(
        "--split_file", type=str, default=None, help="Split file in data/waymo_splits"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="If no scene id or split_file is given, use start_idx and num_scenes to generate scene_ids_list",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=200,
        help="number of scenes to be processed",
    )
    parser.add_argument('--rgb_dirname', type=str, default="inputs/images")
    args = parser.parse_args()

    if args.scene_ids is not None:
        scene_ids_list = args.scene_ids
    elif args.split_file is not None:
        with open(args.split_file, "r") as f:
            split_file = f.readlines()[1:]
        scene_ids_list = [int(line.strip().split(",")[0]) for line in split_file]
    else:
        scene_ids_list = np.arange(args.start_idx, args.start_idx + args.num_scenes)

    camera_dirs = ['CAM_B','CAM_D','CAM_E', 'CAM_F', 'CAM_G', 'CAM_H', 'CAM_I', 'CAM_J']
    
    # 主循环保持不变
    for scene_id in tqdm(scene_ids_list, desc='Processing scenes'):
        scene_id_str = str(scene_id).zfill(3)
        scene_dir = os.path.join(args.data_root, scene_id_str)
        process_scene(scene_dir, args.rgb_dirname, camera_dirs)