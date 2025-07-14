package com.example.kinmenbirdapp

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class Detector(
    private val context: Context,
    private val modelPath: String,
    private val detectorListener: DetectorListener,
    private val message: (String) -> Unit
) {

    private var interpreter: Interpreter
    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0

    private val imageProcessor = ImageProcessor.Builder()
        .add(org.tensorflow.lite.support.common.ops.NormalizeOp(0f, 255f))
        .build()

    init {
        val model = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(model)

        val inputShape = interpreter.getInputTensor(0)?.shape()
        val outputShape = interpreter.getOutputTensor(0)?.shape()

        if (inputShape != null) {
            tensorWidth = inputShape[1]
            tensorHeight = inputShape[2]
        }

        if (outputShape != null) {
            numChannel = outputShape[1]
            numElements = outputShape[2]
        }
    }

    fun close() {
        interpreter.close()
    }

    fun detect(frame: Bitmap): List<BoundingBox> {
        if (tensorWidth == 0 || tensorHeight == 0 || numChannel == 0 || numElements == 0) {
            return emptyList()
        }

        val inferenceTime = SystemClock.uptimeMillis()

        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)

        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer

        val output = TensorBuffer.createFixedSize(intArrayOf(1, numChannel, numElements), DataType.FLOAT32)
        interpreter.run(imageBuffer, output.buffer)

        val bestBoxes = bestBox(output.floatArray)

        return bestBoxes // 返回檢測結果
    }

    private fun bestBox(array: FloatArray): List<BoundingBox> {
        val boundingBoxes = mutableListOf<BoundingBox>()

        for (c in 0 until numElements) {
            var maxConf = 0.3F
            var maxIdx = -1
            var j = 4
            var arrayIdx = c + numElements * j
            while (j < numChannel) {
                if (array[arrayIdx] > maxConf) {
                    maxConf = array[arrayIdx]
                    maxIdx = j - 4
                }
                j++
                arrayIdx += numElements
            }

            // 只检测鸟类 (class 14)
            if (maxConf > 0.3F && maxIdx == 14) {
                val cx = array[c]
                val cy = array[c + numElements]
                val w = array[c + numElements * 2]
                val h = array[c + numElements * 3]
                val x1 = cx - (w / 2F)
                val y1 = cy - (h / 2F)
                val x2 = cx + (w / 2F)
                val y2 = cy + (h / 2F)

                // 扩大边界框，增加 5% 的 padding
                val padding = 0.05F
                val boxWidth = x2 - x1
                val boxHeight = y2 - y1

                var newX1 = x1 - boxWidth * padding
                var newY1 = y1 - boxHeight * padding
                var newX2 = x2 + boxWidth * padding
                var newY2 = y2 + boxHeight * padding

                // 确保边界框不超出图片边界
                newX1 = maxOf(0F, newX1)
                newY1 = maxOf(0F, newY1)
                newX2 = minOf(1F, newX2)  // 假设图像归一化为 [0, 1]，如果是像素则调整为图像宽高
                newY2 = minOf(1F, newY2)

                // 计算正方形边界框
                val boxSize = maxOf(newX2 - newX1, newY2 - newY1)
                val centerX = (newX1 + newX2) / 2
                val centerY = (newY1 + newY2) / 2
                val halfSize = boxSize / 2

                newX1 = maxOf(0F, centerX - halfSize)
                newY1 = maxOf(0F, centerY - halfSize)
                newX2 = minOf(1F, centerX + halfSize)
                newY2 = minOf(1F, centerY + halfSize)

                // 将结果添加到边界框列表
                boundingBoxes.add(
                    BoundingBox(
                        x1 = newX1, y1 = newY1, x2 = newX2, y2 = newY2,
                        cx = centerX, cy = centerY, w = boxSize, h = boxSize,
                        cnf = maxConf, cls = maxIdx, clsName = "bird"
                    )
                )
            }
        }

        return boundingBoxes
    }

    interface DetectorListener {
        fun onEmptyDetect()
        fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long)
    }

    data class BoundingBox(
        val x1: Float, val y1: Float, val x2: Float, val y2: Float,
        val cx: Float, val cy: Float, val w: Float, val h: Float,
        val cnf: Float, val cls: Int, val clsName: String
    )
}
