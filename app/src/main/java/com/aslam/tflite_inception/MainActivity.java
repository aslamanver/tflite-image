package com.aslam.tflite_inception;

import androidx.appcompat.app.AppCompatActivity;
import androidx.databinding.DataBindingUtil;

import android.content.Intent;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;

import com.aslam.tflite_image.TFLiteImage;
import com.aslam.tflite_inception.databinding.ActivityMainBinding;

import java.util.List;
import java.util.Map;

public class MainActivity extends AppCompatActivity {

    ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = DataBindingUtil.setContentView(this, R.layout.activity_main);
        binding.btnChoose.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pickImage();
            }
        });
    }

    public void pickImage() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        intent.setType("image/*");
        intent.putExtra("return-data", true);
        startActivityForResult(intent, 1);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && requestCode == 1) {
            Bundle extras = data.getExtras();
            if (extras != null) {
                binding.progressBar.setVisibility(View.VISIBLE);
                binding.imgView.setImageURI(data.getData());
                binding.txtResult.setText("Analysing..");
                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        TFLiteImage tfLite = TFLiteImage.getInstance(MainActivity.this, "inception_quant.tflite", "labels.txt", TFLiteImage.TYPE.QUANT);
                        List<Map<String, String>> results = tfLite.predictImage(binding.imgView);
                        String result = "";
                        for (Map<String, String> map : results) {
                            result += map.get("LABEL") + " - " + map.get("CONFIDENCE") + "\n";
                        }
                        final String finalResult = result;
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                binding.txtResult.setText(finalResult);
                                binding.progressBar.setVisibility(View.GONE);
                            }
                        });
                    }
                }).start();
            }
        }
    }
}
