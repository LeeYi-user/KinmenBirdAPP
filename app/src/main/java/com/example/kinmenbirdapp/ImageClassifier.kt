import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.DataType
import org.json.JSONObject

class ImageClassifier(context: Context) {

    private val interpreter: Interpreter
    private val imageProcessor: ImageProcessor
    private val inputImage: TensorImage
    private val outputBuffer: TensorBuffer
    private val labelMap: Map<Int, String>

    init {
        // 載入模型
        val model = FileUtil.loadMappedFile(context, "EfficientNetV2B0-InferenceOnly.tflite")
        interpreter = Interpreter(model)

        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
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

    fun classify(bitmap: Bitmap): String {
        inputImage.load(bitmap)
        val processed = imageProcessor.process(inputImage)

        interpreter.run(processed.buffer, outputBuffer.buffer.rewind())

        val confidences = outputBuffer.floatArray
        val maxIndex = confidences.indices.maxByOrNull { confidences[it] } ?: -1
        return labelMap[maxIndex] ?: "未知類別"
    }
}
