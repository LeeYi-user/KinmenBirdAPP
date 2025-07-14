package com.example.kinmenbirdapp

import android.app.Activity
import android.content.Intent
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.*
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity(), Detector.DetectorListener {

    private lateinit var imageView: ImageView
    private lateinit var textViewResult: TextView
    private lateinit var detector: Detector
    private lateinit var classifier: ImageClassifier

    companion object {
        private const val REQUEST_CODE_IMAGE_PICK = 1001
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        textViewResult = findViewById(R.id.textViewResult)
        val buttonSelect = findViewById<Button>(R.id.buttonSelect)

        detector = Detector(
            context = this,
            modelPath = "yolov8s_float32.tflite", // 修改為你 assets 中的模型名稱
            detectorListener = this,
            message = { msg ->
                runOnUiThread {
                    Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
                }
            }
        )

        classifier = ImageClassifier(this, detector)

        buttonSelect.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(intent, REQUEST_CODE_IMAGE_PICK)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_CODE_IMAGE_PICK && resultCode == Activity.RESULT_OK) {
            data?.data?.let { uri ->
                val bitmap = uriToBitmap(uri)
                imageView.setImageBitmap(bitmap)
                val (croppedBitmap, resultName) = classifier.classify(bitmap)
                if (croppedBitmap != null) {
                    imageView.setImageBitmap(croppedBitmap)
                } else {
                    imageView.setImageBitmap(bitmap) // 如果沒偵測到鳥，就顯示原圖
                }
                textViewResult.text = "預測結果：$resultName"
            }
        }
    }

    private fun uriToBitmap(uri: Uri): android.graphics.Bitmap {
        val inputStream = contentResolver.openInputStream(uri)
        return BitmapFactory.decodeStream(inputStream!!)
    }

    override fun onEmptyDetect() {
        runOnUiThread {
            Toast.makeText(this, "未檢測到任何目標", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onDetect(boundingBoxes: List<Detector.BoundingBox>, inferenceTime: Long) {
        // 可選：顯示推論時間、框的資訊等
    }
}
