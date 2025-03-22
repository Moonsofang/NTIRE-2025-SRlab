import os
import pyiqa
import argparse
from tqdm import tqdm

def test_image_quality(image_dir, metrics, weight_paths):
    """
    测试指定文件夹中所有 PNG 图像的质量指标。

    Args:
        image_dir (str): 包含 PNG 图像的文件夹路径。
        metrics (list): 需要测试的指标列表，例如 ['musiq', 'maniqa', 'clipiqa'].
        weight_paths (dict): 每个指标的本地权重文件路径。
    """
    # 初始化指标模型
    metric_models = {}
    for metric in metrics:
        if metric in weight_paths:
            # 如果提供了本地权重路径，则加载本地权重
            model = pyiqa.create_metric(metric, pretrained_model_path=weight_paths[metric])
        else:
            # 否则使用默认权重（需要网络下载）
            model = pyiqa.create_metric(metric)
        metric_models[metric] = model

    # 获取所有 PNG 图像路径
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
    if not image_paths:
        print(f"未找到 PNG 图像：{image_dir}")
        return


    # 遍历图像并计算指标
    results = {metric: [] for metric in metrics}
    for image_path in tqdm(image_paths, desc="Processing images"):
        for metric, model in metric_models.items():
            score = model(image_path)  # 计算指标分数
            results[metric].append(score.item())  # 将分数添加到结果中

    # 打印结果
    print("\n测试结果：")
    for metric, scores in results.items():
        avg_score = sum(scores) / len(scores)
        # print(f"{metric.upper()} - 平均分数: {avg_score:.4f}")
        print(avg_score)
        # print(f"{metric.upper()} - 单张图像分数: {scores}")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="测试图像质量指标")
    parser.add_argument("--image_dir", type=str, required=True, help="包含 PNG 图像的文件夹路径")
    args = parser.parse_args()

    # 需要测试的指标
    metrics_to_test = ['musiq', 'maniqa', 'clipiqa']

    # 每个指标的本地权重文件路径
    weight_paths = {
        'musiq': '/media/ssd8T/wyw/Pretrained/musiq/musiq_koniq_ckpt-e95806b9.pth',
        'maniqa': '/media/ssd8T/wyw/Pretrained/clipiqa/ckpt_koniq10k.pt',
    }

    # 运行测试
    test_image_quality(args.image_dir, metrics_to_test, weight_paths)