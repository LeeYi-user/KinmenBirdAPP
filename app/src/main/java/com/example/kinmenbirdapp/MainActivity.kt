package com.example.kinmenbirdapp

import ImageClassifier
import android.app.Activity
import android.content.Intent
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.*
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var textViewResult: TextView
    private lateinit var classifier: ImageClassifier

    companion object {
        private const val REQUEST_CODE_IMAGE_PICK = 1001
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        classifier = ImageClassifier(this)

        imageView = findViewById(R.id.imageView)
        textViewResult = findViewById(R.id.textViewResult)
        val buttonSelect = findViewById<Button>(R.id.buttonSelect)

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
                val resultIndex = classifier.classify(bitmap)
                textViewResult.text = "預測結果 index: $resultIndex"
            }
        }
    }

    private fun uriToBitmap(uri: Uri): android.graphics.Bitmap {
        val inputStream = contentResolver.openInputStream(uri)
        return BitmapFactory.decodeStream(inputStream!!)
    }
}
