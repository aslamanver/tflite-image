package com.aslam.tflite_image;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.widget.ImageView;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class TFLiteImage {

    public enum TYPE {
        QUANT, FLOAT
    }

    // presets for rgb conversion
    private static final int RESULTS_TO_SHOW = 5;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;
    // options for model interpreter
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();
    // tflite graph
    private Interpreter tflite;
    // holds all the possible labels for model
    private List<String> labelList;
    // holds the selected image data as bytes
    private ByteBuffer imgData = null;
    // holds the probabilities of each label for non-quantized graphs
    private float[][] labelProbArray = null;
    // holds the probabilities of each label for quantized graphs
    private byte[][] labelProbArrayB = null;
    // array that holds the labels with the highest probabilities
    private String[] topLables = null;
    // array that holds the highest probabilities
    private String[] topConfidence = null;
    // input image dimensions for the Inception Model
    private int DIM_IMG_SIZE_X = 299;
    private int DIM_IMG_SIZE_Y = 299;
    private int DIM_PIXEL_SIZE = 3;
    // int array to hold image data
    private int[] intValues;
    // application context
    private Context context;
    private String model;
    private String label;
    private TYPE type;

    private TFLiteImage(Context context, String model, String label, TYPE type, int DIM_IMG_SIZE_X, int DIM_IMG_SIZE_Y) {
        this.context = context;
        this.model = model;
        this.label = label;
        this.type = type;
        this.DIM_IMG_SIZE_X = DIM_IMG_SIZE_X;
        this.DIM_IMG_SIZE_Y = DIM_IMG_SIZE_Y;
        // initialize array that holds image data
        intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];
        //initilize graph and labels
        try {
            tflite = new Interpreter(loadModelFile(), tfliteOptions);
            labelList = loadLabelList();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        // initialize byte array. The size depends if the input data needs to be quantized or not
        if (type == TYPE.QUANT) {
            imgData = ByteBuffer.allocateDirect(DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
        } else {
            imgData = ByteBuffer.allocateDirect(4 * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
        }
        imgData.order(ByteOrder.nativeOrder());
        // initialize probabilities array. The datatypes that array holds depends if the input data needs to be quantized or not
        if (type == TYPE.QUANT) {
            labelProbArrayB = new byte[1][labelList.size()];
        } else {
            labelProbArray = new float[1][labelList.size()];
        }
        // initialize array to hold top labels
        topLables = new String[RESULTS_TO_SHOW];
        // initialize array to hold top probabilities
        topConfidence = new String[RESULTS_TO_SHOW];
    }

    public static TFLiteImage getInstance(Context context, String model, String label, TYPE type, int DIM_IMG_SIZE_X, int DIM_IMG_SIZE_Y) {
        return new TFLiteImage(context, model, label, type, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y);
    }

    public static TFLiteImage getInstance(Context context, String model, String label, TYPE type, int DIM_IMG_SIZE) {
        return new TFLiteImage(context, model, label, type, DIM_IMG_SIZE, DIM_IMG_SIZE);
    }

    public static TFLiteImage getInstance(Context context, String model, String label, TYPE type) {
        return new TFLiteImage(context, model, label, type, 299, 299);
    }

    public List<Map<String, String>> predictImage(Bitmap bitmapFile) {
        // resize the bitmap to the required input size to the CNN
        Bitmap bitmap = getResizedBitmap(bitmapFile, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y);
        // convert bitmap to byte array
        convertBitmapToByteBuffer(bitmap);
        // pass byte data to the graph
        if (type == TYPE.QUANT) {
            tflite.run(imgData, labelProbArrayB);
        } else {
            tflite.run(imgData, labelProbArray);
        }
        // return the results
        return topKLabels();
    }

    public List<Map<String, String>> predictImage(ImageView imageView) {
        Bitmap bitmap_orig = ((BitmapDrawable) imageView.getDrawable()).getBitmap();
        return predictImage(bitmap_orig);
    }

    // priority queue that will hold the top results from the CNN
    private PriorityQueue<Map.Entry<String, Float>> sortedLabels = new PriorityQueue<>(RESULTS_TO_SHOW, new Comparator<Map.Entry<String, Float>>() {
        @Override
        public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
            return (o1.getValue()).compareTo(o2.getValue());
        }
    });

    // loads tflite grapg from file
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(model);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // converts bitmap to byte array which is passed in the tflite graph
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) return;
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // loop through all pixels
        int pixel = 0;
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                // get rgb values from intValues where each int holds the rgb values for a pixel.
                // if quantized, convert each rgb value to a byte, otherwise to a float
                if (type == TYPE.QUANT) {
                    imgData.put((byte) ((val >> 16) & 0xFF));
                    imgData.put((byte) ((val >> 8) & 0xFF));
                    imgData.put((byte) (val & 0xFF));
                } else {
                    imgData.putFloat((((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((val) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                }
            }
        }
    }

    // loads the labels from the label txt file in assets into a string array
    private List<String> loadLabelList() throws IOException {
        List<String> labelList = new ArrayList<String>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(context.getAssets().open(label)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    // print the top labels and respective confidences
    private List<Map<String, String>> topKLabels() {
        // add all results to priority queue
        for (int i = 0; i < labelList.size(); ++i) {
            if (type == TYPE.QUANT) {
                sortedLabels.add(new AbstractMap.SimpleEntry<>(labelList.get(i), (labelProbArrayB[0][i] & 0xff) / 255.0f));
            } else {
                sortedLabels.add(new AbstractMap.SimpleEntry<>(labelList.get(i), labelProbArray[0][i]));
            }
            if (sortedLabels.size() > RESULTS_TO_SHOW) {
                sortedLabels.poll();
            }
        }
        // get top results from priority queue
        List<Map<String, String>> resultList = new ArrayList<>();
        final int size = sortedLabels.size();
        for (int i = 0; i < size; ++i) {
            Map.Entry<String, Float> label = sortedLabels.poll();
            topLables[i] = label.getKey();
            topConfidence[i] = String.format("%.0f%%", label.getValue() * 100);
            // insert to result list
            Map<String, String> result = new HashMap<>();
            result.put("LABEL", topLables[i]);
            result.put("CONFIDENCE", topConfidence[i]);
            resultList.add(result);
        }
        Collections.reverse(resultList);
        return resultList;
    }

    // resizes bitmap to given dimensions
    private Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
        int width = bm.getWidth();
        int height = bm.getHeight();
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);
        Bitmap resizedBitmap = Bitmap.createBitmap(bm, 0, 0, width, height, matrix, false);
        return resizedBitmap;
    }
}
