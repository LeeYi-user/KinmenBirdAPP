package com.example.kinmenbirdapp

import android.app.Activity
import android.content.Intent
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.viewpager2.widget.ViewPager2

class MainActivity : AppCompatActivity(), Detector.DetectorListener {

    private lateinit var textViewResult: TextView
    private lateinit var detector: Detector
    private lateinit var classifier: ImageClassifier
    private lateinit var viewPager: ViewPager2
    private lateinit var resultsAdapter: ResultsAdapter

    companion object {
        private const val REQUEST_CODE_IMAGE_PICK = 1001
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        textViewResult = findViewById(R.id.textViewResult)
        val buttonSelect = findViewById<Button>(R.id.buttonSelect)
        viewPager = findViewById(R.id.viewPager)

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

                val results = classifier.classify(bitmap)

                if (results.isNotEmpty()) {
                    // 更新 ViewPager
                    resultsAdapter = ResultsAdapter(results)
                    viewPager.adapter = resultsAdapter
                    textViewResult.text = "找到 ${results.size} 筆預測結果"
                } else {
                    textViewResult.text = "未檢測到鳥類"
                }
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
