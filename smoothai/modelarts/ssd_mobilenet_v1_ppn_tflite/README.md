适用于华为云ModelArts服务AI市场算法[SSD_MobileNet_v1_PPN（物体检测/TensorFlow）](https://marketplace.huaweicloud.com/markets/aihub/modelhub/detail/?id=17c9de47-20cf-4f0f-96c5-5cba6aa32823)

训练完成后得到以下文件：
```shell
模型输出目录
  |- frozen_graph
    |- model.pb
  |- model
    |- ...
    |- index
    |- ...
  |- model.ckpt-xxx
  |- ...
```

要使用tflite，就必须将模型的前后处理摘除（tflite不支持），并在推理脚本中用numpy接口实现

使用`convert_ssd_mobilenet_ppn_tflite.py`这个脚本将`frozen_graph/model.pb`转换为`converted_model.tflite`

`convert_ssd_mobilenet_ppn_tflite.py`脚本的一些解释

输入节点：
- `Preprocessor/sub`: 预处理后的图片，shape为`[1, 300, 300, 3]`
- `Preprocessor/map/TensorArrayStack_1/TensorArrayGatherV3`: 输入图片与处理前的shape，推理时用不到，可以不填

输出节点：
- `Concatenate/concat`: 检测候选框anchors，常量
- `concat`: 表示box_encodings，检测heads输出中的坐标预测，经过后处理后可以变成预测框坐标
- `concat_1`: 表示class_predictions_with_background，校测heads输出中的类别预测，经过后处理后可以标称分类预测结果

得到tflite模型后使用`infer_ssd_mobilenet_ppn_tflite.py`接可以推理了，脚本中使用的`/tmp/tflite/index`文件可以在训练输出中获取，自行替换需要预测的图片路径和tflite模型路径