package com.example.kinmenbirdapp

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.example.kinmenbirdapp.R

class ResultsAdapter(private val results: List<Pair<android.graphics.Bitmap, String>>) : RecyclerView.Adapter<ResultsAdapter.ViewHolder>() {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_result, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val result = results[position]
        holder.imageView.setImageBitmap(result.first)
        holder.textView.text = "預測結果: ${result.second}"
    }

    override fun getItemCount(): Int {
        return results.size
    }

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val imageView: ImageView = view.findViewById(R.id.imageView)
        val textView: TextView = view.findViewById(R.id.textView)
    }
}
