import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ImageClassifier(context: Context) {

    private val interpreter: Interpreter

    init {
        val modelBuffer: ByteBuffer = FileUtil.loadMappedFile(context, "EfficientNetV2B0-InferenceOnly.tflite")
        interpreter = Interpreter(modelBuffer)
    }

    fun classify(bitmap: Bitmap): Int {
        val inputSize = 224
        val inputBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                val pixel = resized.getPixel(x, y)
                inputBuffer.putFloat(((pixel shr 16 and 0xFF) / 255.0f))
                inputBuffer.putFloat(((pixel shr 8 and 0xFF) / 255.0f))
                inputBuffer.putFloat(((pixel and 0xFF) / 255.0f))
            }
        }

        val output = Array(1) { FloatArray(251) } // 修改成模型實際的輸出大小
        interpreter.run(inputBuffer, output)

        return output[0].indices.maxByOrNull { output[0][it] } ?: -1
    }
}
