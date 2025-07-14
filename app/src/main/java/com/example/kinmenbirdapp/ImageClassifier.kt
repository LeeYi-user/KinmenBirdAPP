package com.example.kinmenbirdapp

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.DataType
import org.json.JSONObject

class ImageClassifier(context: Context, private val detector: Detector) {

    private val interpreter: Interpreter
    private val imageProcessor: ImageProcessor
    private val inputImage: TensorImage
    private val outputBuffer: TensorBuffer
    private val labelMap: Map<Int, String>

    init {
        // 載入 EfficientNetV2B0-InferenceOnly 模型
        val model = FileUtil.loadMappedFile(context, "EfficientNetV2B0-InferenceOnly.tflite")
        interpreter = Interpreter(model)

        imageProcessor = ImageProcessor.Builder()
            .add(org.tensorflow.lite.support.image.ops.ResizeOp(224, 224, org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod.BILINEAR))
            .build()

        inputImage = TensorImage(DataType.FLOAT32)
        outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 251), DataType.FLOAT32)

        // 載入 index → 學名 的對應表
        labelMap = loadLabelMap(context)
    }

    private fun loadLabelMap(context: Context): Map<Int, String> {
        val labels = mutableMapOf<Int, String>()
        val jsonStr = context.assets.open("class_indices.json").bufferedReader().use { it.readText() }
        val jsonObj = JSONObject(jsonStr)
        for (key in jsonObj.keys()) {
            labels[key.toInt()] = jsonObj.getString(key)
        }
        return labels
    }

    fun classify(bitmap: Bitmap): Pair<Bitmap?, String> {
        val boundingBoxes = detector.detect(bitmap)

        if (boundingBoxes.isNotEmpty()) {
            val box = boundingBoxes[0]
            val imageWidth = bitmap.width
            val imageHeight = bitmap.height

            val left = (box.x1 * imageWidth).toInt().coerceIn(0, imageWidth - 1)
            val top = (box.y1 * imageHeight).toInt().coerceIn(0, imageHeight - 1)
            val right = (box.x2 * imageWidth).toInt().coerceIn(0, imageWidth)
            val bottom = (box.y2 * imageHeight).toInt().coerceIn(0, imageHeight)

            val cropWidth = (right - left).coerceAtLeast(1)
            val cropHeight = (bottom - top).coerceAtLeast(1)

            val croppedBitmap = Bitmap.createBitmap(bitmap, left, top, cropWidth, cropHeight)

            inputImage.load(croppedBitmap)
            val processed = imageProcessor.process(inputImage)

            interpreter.run(processed.buffer, outputBuffer.buffer.rewind())

            val confidences = outputBuffer.floatArray
            val maxIndex = confidences.indices.maxByOrNull { confidences[it] } ?: -1
            val result = labelMap[maxIndex] ?: "未知類別"

            return Pair(croppedBitmap, result)
        }

        return Pair(null, "未檢測到鳥類")
    }
}
