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

            // 只檢測鳥類 (class 14)
            if (maxConf > 0.3F && maxIdx == 14) {
                val cx = array[c]
                val cy = array[c + numElements]
                val w = array[c + numElements * 2]
                val h = array[c + numElements * 3]
                val x1 = cx - (w / 2F)
                val y1 = cy - (h / 2F)
                val x2 = cx + (w / 2F)
                val y2 = cy + (h / 2F)

                boundingBoxes.add(
                    BoundingBox(
                        x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                        cx = cx, cy = cy, w = w, h = h,
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
