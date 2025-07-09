import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.DataType

class ImageClassifier(context: Context) {

    private val interpreter: Interpreter
    private val imageProcessor: ImageProcessor
    private val inputImage: TensorImage
    private val outputBuffer: TensorBuffer

    init {
        // Load模型
        val model = FileUtil.loadMappedFile(context, "EfficientNetV2B0-InferenceOnly.tflite")
        interpreter = Interpreter(model)

        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        inputImage = TensorImage(DataType.FLOAT32)

        // output dims: [1, NUM_CLASSES]
        outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 251), DataType.FLOAT32)
    }

    fun classify(bitmap: Bitmap): Int {
        // 加载并处理
        inputImage.load(bitmap)
        val processed = imageProcessor.process(inputImage)

        // 执行推理
        interpreter.run(processed.buffer, outputBuffer.buffer.rewind())

        // 获取最高概率类别
        val confidences = outputBuffer.floatArray
        return confidences.indices.maxByOrNull { confidences[it] } ?: -1
    }
}
