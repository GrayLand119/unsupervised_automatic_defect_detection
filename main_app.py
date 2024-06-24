import time
from glob import glob
from typing import Optional

import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
from tkinter import Tk, filedialog
import os
import sys
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch.backends.mps

from anomalib.data.utils import generate_output_image_filename, get_image_filenames, read_image
from anomalib.deploy import OpenVINOInferencer
from anomalib.post_processing import Visualizer
import cv2 as cv2
import matplotlib.pyplot as plt


# sys.path.insert(0, './tools/inference/')
# 无监督自动缺陷检测
def on_browse():
    data_type = 'Folder'
    root = Tk()
    root.mainloop()
    root.attributes("-topmost", True)
    root.withdraw()

    if data_type == "Files":
        filenames = filedialog.askopenfilenames()
        if len(filenames) > 0:
            root.destroy()
            return str(filenames)
        else:
            filename = "Files not seleceted"
            root.destroy()
            return str(filename)

    elif data_type == "Folder":
        filename = filedialog.askdirectory()
        if filename:
            if os.path.isdir(filename):
                root.destroy()
                return str(filename)
            else:
                root.destroy()
                return str(filename)
        else:
            filename = "Folder not seleceted"
            root.destroy()
            return str(filename)


def on_select(selected):
    for ele in selected:
        path = ele.file
    print(selected)
    return ""


def on_start_training(x, pg=gr.Progress()):
    # pg: gr.Progress=gr.Progress()
    pg(0, desc="Starting...")
    # time.sleep(1)
    # pg(0.05)
    for i in pg.tqdm(range(500), unit='epochs'):
        #     # pg_text = f"{i}%"
        #     # print(dir(i))
        time.sleep(0.02)

    gr.Info("训练完成!")
    return "训练完成!"


g_category = ''
inferencer: Optional[OpenVINOInferencer] = None

# inferencer = OpenVINOInferencer(path='results/bottle_crate/stfpm/mvtec/run/weights/openvino/model.bin',
#                                 metadata='results/bottle_crate/stfpm/mvtec/run/weights/openvino/metadata.json',
#                                 device='CPU')
# visualizer: Optional[Visualizer] = None


visualizer = Visualizer(mode='simple',
                        task='segmentation')


def on_reload_model():
    pass


def on_select_model(category):
    global g_category
    global inferencer

    if g_category != category:
        g_category = category  # results/pcb_anomaly/stfpm/mvtec/run/weights/openvino
        device = 'CPU' if torch.backends.mps.is_available() else 'GPU'
        inferencer = OpenVINOInferencer(path=f'results/{category}/stfpm/mvtec/run/weights/openvino/model.bin',
                                        metadata=f'results/{category}/stfpm/mvtec/run/weights/openvino/metadata.json',
                                        device=device)
    print(f'current category:{category}')
    # gr.update(examples=glob(f'datasets/MVTec/{category}/[bad,good]*/*.png'))
    # if category == 'bottle_crate' and image is not None:
    #     image.update(width=640, height=512)

    # shape = shapes_map[category] # W, H
    # ratio = shape[0] / shape[1]
    # max_size = max(shape[0], shape[1])
    return {
        example_bottle_crate: gr.update(visible=category == 'bottle_crate'),
        example_pcb_anomaly: gr.update(visible=category == 'pcb_anomaly'),
        image_bottle: gr.update(visible=category == 'bottle_crate'),
        image_pcb: gr.update(visible=category == 'pcb_anomaly'),
        # image: gr.Image.update(width=shape[0]//2.0, height=shape[1]//2.0)
        # image: gr.Image.update(height=320, width=int(320 * ratio) + 1)
    }

    # return gr.Examples(,
    #                    inputs=image,
    #                    outputs=result_image,
    #                    cache_examples=True,
    #                    examples_per_page=20,
    #                    preprocess=False,
    #                    postprocess=False,
    #                    fn=on_pred)


# def on_change_to_editor_image(image: gr.Image):
#     image.tool = 'editor'
#     return gr.Image.update()
# return gr.Image(label="待检图片",
#              tool='editor',
#              # width=760,
#              # tool='editor',
#              # type='pil',
#              scale=1)
# def on_change_to_painting_image(image: gr.Image):
#     image.tool = 'color-sketch'
# image = gr.Image(label="待检图片",
#                  tool='color-sketch',
#                  # width=760,
#                  # tool='editor',
#                  # type='pil',
#                  scale=1)
# return image

def on_pred(image_bottole: np.array, image_pcb: np.array, pixel_threshold):
    global g_category
    global inferencer

    if inferencer is None:
        print('inferencer is nont load it')
        on_select_model('bottle_crate')

    image: np.array = None
    if g_category == 'bottle_crate':
        image = image_bottole
        image_threshold = 0.05
    elif g_category == 'pcb_anomaly':
        image = image_pcb
        image_threshold = 0.008

    if isinstance(image, str):
        print("is base64 image")
        return
    if isinstance(image, dict):
        mask = image['mask']
        image = image['image']
        mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2RGB)
        image = cv2.merge(image, mask)


    # print(f"image name:{name}")
    print(f'input image: {image.shape} {image.dtype}')
    # image = read_image('datasets/MVTec/bottle_crate/bad/bottle_crate_03.png')
    # inferencer.metadata['pixel_threshold'] = pixel_threshold
    t1 = time.time()
    predictions = inferencer.predict(image=image, metadata={'pixel_threshold': pixel_threshold})
    # time.sleep(1)
    t2 = time.time()
    cost_time = (t2 - t1) * 1000.0

    output = visualizer.visualize_image(predictions)
    t3 = time.time()
    cost_time2 = (t3 - t2) * 1000.0


    res = f"检测: {cost_time:.2f} ms, 合成检测结果: {cost_time2:.2f} ms\nScore: {predictions.pred_score} - {'OK' if predictions.pred_score < image_threshold else 'NG'}"
    return output, res


def load_train_dataset(dir_path):
    images = []
    paths = glob(f'datasets/MVTec/{dir_path[0]}/*.png')
    for i in paths:
        images.append(cv2.cvtColor(cv2.imread(i, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    return [f'{dir_path[0]}/*.png', images[:50]]


select_shape = None


def on_select_example(evt: gr.SelectData):
    global select_shape
    path = evt.target.value[evt.index]['name']
    img = cv2.cvtColor(cv2.imread(path, 1), cv2.COLOR_BGR2RGB)
    select_shape = img.shape
    print(f'Select Image size: {img.shape}')
    # img = cv2.resize(img, fx=2.5, fy=2.5)
    return img


def on_tool_radio(evt, image: gr.Image):
    # image.update(tool='editor')
    pass


# image = None

shapes_map = {
    'pcb_anomaly': (1482, 794),
    'bottle_crate': (640, 512)
}

with gr.Blocks() as demo:
    gr.Markdown("""# SIMOTECH在线缺陷检测系统
    特点:
    - 仅需少量的良品图片进行训练， 即完成特征提取
    - 不需要不良品图片参与训练
    - 数分钟内即可完成训练，上线应用""")

    with gr.Tab('训练'):
        dropdown = gr.Dropdown(label='选择模型', choices=['STFPM', 'PaDiM', 'PatchCore', 'EfficientAD-S'], value='STFPM', interactive=True)
        tb_img_path = gr.Textbox(label='训练集')
        tb_img = gr.Gallery(label='训练集', columns=6)
        # intput = gr.Text(label='Input Path')
        output = gr.Text(label='Train Path', show_label=False)
        btn = gr.Button("开始训练")
        btn.click(on_start_training, tb_img_path, output)

        gr.Markdown("## 范例")
        train_dataset = gr.Dataset(samples=[['bottle_crate/good'],
                                            # ['weld_dot/train/good'],
                                            ['pcb_anomaly/good']],
                                   components=[tb_img_path])
        train_dataset.select(fn=load_train_dataset, inputs=train_dataset, outputs=[tb_img_path, tb_img])
        dropdown.change(fn=on_reload_model, inputs=None, outputs=train_dataset)

    with gr.Tab('检测'):
        # bottle_crate - 640 512
        select_model = gr.Dropdown(label='检测产品类型',
                                   choices=['bottle_crate', 'pcb_anomaly'],
                                   value='bottle_crate',
                                   interactive=True)

        with gr.Row():
            with gr.Column():
                # tool_radio = gr.Radio(label='编辑工具', choices=['全局编辑', '裁切工具', '涂画工具'], value='全局编辑')
                # image_normal = gr.Image(label="待检图片",
                #                  tool='editor',
                #                  visible=True,
                #                  scale=1)
                # image_sketch = gr.Image(label="待检图片",
                #                  tool='color-sketch',
                #                  visible=False,
                #                  scale=1)
                image_bottle = gr.Image(label="待检图片",
                                        # tool='color-sketch',
                                        shape=shapes_map['bottle_crate'],
                                        visible=True,
                                        )
                image_pcb = gr.Image(label="待检图片",
                                     # tool='color-sketch',
                                     shape=shapes_map['pcb_anomaly'],
                                     visible=False
                                     )
                # tool_radio.change(fn=on_tool_radio, inputs=[tool_radio, image])

                # with gr.Row():
                #     painting_btn = gr.Button("绘图")
                #     normal_btn = gr.Button("普通")
                pixel_threshold_slider = gr.Slider(0, 0.05, 0.0085, label='Pixel threshold')
                pred_btn = gr.Button("开始检测")
            with gr.Column():
                result_image = gr.Image(label="检测结果",
                                        # shape=shapes_map['bottle_crate']
                                        )
                result_text = gr.Textbox(label='耗时')

            # def on_change_to_editor_image():
            #     global image
            #     image_normal.update(visible=False)
            #     image_sketch.update(visible=True)
            #     image = image_normal
            #
            # def on_change_to_sketch_image():
            #     global image
            #     image_normal.update(visible=True)
            #     image_sketch.update(visible=False)
            #     image = image_sketch

            pred_btn.click(on_pred, [image_bottle, image_pcb, pixel_threshold_slider], [result_image, result_text])
            # painting_btn.click(on_change_to_sketch_image)
            # normal_btn.click(on_change_to_editor_image)

        gr.Markdown("## 范例")
        with gr.Row(visible=True) as example_bottle_crate:
            example_gallery = gr.Gallery(glob('datasets/MVTec/bottle_crate/[bad,good]*/*.png'),
                                         columns=12, height='160px', allow_preview=False)
            example_gallery.select(fn=on_select_example, outputs=image_bottle)
            # gr.Examples(glob('datasets/MVTec/bottle_crate/[bad,good]*/*.png'),
            #             inputs=image,
            #             outputs=result_image,
            #             cache_examples=False,
            #             examples_per_page=12,
            #             preprocess=True,
            #             postprocess=False)
        with gr.Row(visible=False) as example_pcb_anomaly:
            # gr.Examples(glob('datasets/MVTec/pcb_anomaly/[anomaly,good]*/*.png'),
            #             inputs=image,
            #             outputs=result_image,
            #             cache_examples=False,
            #             examples_per_page=12,
            #             preprocess=True,
            #             postprocess=False,
            #             fn=on_pred)

            example2_gallery = gr.Gallery(glob('datasets/MVTec/pcb_anomaly/[anomaly,good]*/*.png'),
                                          columns=12, height='160px', allow_preview=False)
            example2_gallery.select(fn=on_select_example, outputs=image_pcb)

        select_model.change(on_select_model, inputs=[select_model],
                            outputs=[example_bottle_crate, example_pcb_anomaly, image_bottle, image_pcb])
        demo.css = "footer {visibility: hidden}"

# demo.queue(concurrency_count=10).launch(inbrowser=True)
demo.queue().launch(inbrowser=True, server_name='0.0.0.0')
