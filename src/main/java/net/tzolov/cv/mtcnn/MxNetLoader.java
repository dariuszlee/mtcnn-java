package net.tzolov.cv.mtcnn;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.mxnet.infer.javaapi.Predictor;
import org.apache.mxnet.javaapi.Context;
import org.apache.mxnet.javaapi.DType;
import org.apache.mxnet.javaapi.DataDesc;
import org.apache.mxnet.javaapi.Layout;
import org.apache.mxnet.javaapi.NDArray;
import org.apache.mxnet.javaapi.Shape;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class MxNetLoader {
  private final Predictor proposeNetGraphRunner;
  private final Java2DNativeImageLoader imageLoader;
  private List<Context> ctx;
  private Shape inputShape;

  public MxNetLoader(int[] scales, String modelPath) {
    this.imageLoader = new Java2DNativeImageLoader();

    this.ctx = new ArrayList<>();
    this.ctx.add(Context.cpu()); // Choosing CPU Context here

    this.inputShape = new Shape(scales);

    List<DataDesc> inputDesc = new ArrayList<>();
    inputDesc.add(
        new DataDesc(
            "data",
            inputShape,
            DType.Float32(),
            Layout.NCHW()));
    proposeNetGraphRunner = new Predictor(modelPath, inputDesc, this.ctx, 0);
  }

  public List<INDArray> runPropose(INDArray ndImage3HW) {
    float[] ints = ndImage3HW.data().asFloat();
    NDArray img = new NDArray(ints, this.inputShape, this.ctx.get(0));

    List<NDArray> imgs = new ArrayList<NDArray>();
    imgs.add(img);

    List<NDArray> outs = proposeNetGraphRunner.predictWithNDArray(imgs);

    NDArray out1 = outs.get(0);
    INDArray out1_n = Nd4j.create(out1.toArray(), out1.shape().toArray());
    out1_n = out1_n.permute(0, 2, 3, 1);

    NDArray out2 = outs.get(1);
    INDArray out2_n = Nd4j.create(out2.toArray(), out2.shape().toArray());
    out2_n = out2_n.permute(0, 2, 3, 1);

    List<INDArray> outs2 = new ArrayList<>();
    outs2.add(out1_n);
    outs2.add(out2_n);
    return outs2;
  }

  public List<INDArray> runRefine(INDArray ndImage3HW) {
    float[] out2_a = new float[]{};
    float[] out2_b = new float[]{};
    for (int bunch = 0; bunch < ndImage3HW.size(0); bunch++){
      INDArray bunchData = ndImage3HW.get(point(bunch), all(), all(), all()).reshape(this.inputShape.shape().toArray()).dup();
      float[] ints = bunchData.data().asFloat();

      NDArray img = new NDArray(ints, this.inputShape, this.ctx.get(0));

      List<NDArray> imgs = new ArrayList<NDArray>();
      imgs.add(img);
      List<NDArray> outs = proposeNetGraphRunner.predictWithNDArray(imgs);

      NDArray out1 = outs.get(0);
      out2_a = ArrayUtils.addAll(out2_a, out1.toArray());

      NDArray out2 = outs.get(1);
      out2_b = ArrayUtils.addAll(out2_b, out2.toArray());
    }
    INDArray out1s_ind = Nd4j.create(out2_a, new int[]{(int)ndImage3HW.size(0), 4});
    INDArray out2s_ind = Nd4j.create(out2_b, new int[]{(int)ndImage3HW.size(0), 2});

    List<INDArray> outs2 = new ArrayList<>();
    outs2.add(out1s_ind);
    outs2.add(out2s_ind);

    return outs2;
  }
  
  public List<INDArray> runOutput(INDArray ndImage3HW) {
    float[] out2_a = new float[]{};
    float[] out2_b = new float[]{};
    float[] out2_c = new float[]{};
    for (int bunch = 0; bunch < ndImage3HW.size(0); bunch++){
      INDArray bunchData = ndImage3HW.get(point(bunch), all(), all(), all()).reshape(this.inputShape.shape().toArray()).dup();
      float[] ints = bunchData.data().asFloat();

      NDArray img = new NDArray(ints, this.inputShape, this.ctx.get(0));

      List<NDArray> imgs = new ArrayList<NDArray>();
      imgs.add(img);
      List<NDArray> outs = proposeNetGraphRunner.predictWithNDArray(imgs);

      NDArray out1 = outs.get(0);
      out2_a = ArrayUtils.addAll(out2_a, out1.toArray());

      NDArray out2 = outs.get(1);
      out2_b = ArrayUtils.addAll(out2_b, out2.toArray());

      NDArray out3 = outs.get(2);
      out2_c = ArrayUtils.addAll(out2_c, out3.toArray());
    }
    INDArray out1s_ind = Nd4j.create(out2_a, new int[]{(int)ndImage3HW.size(0), 10});
    INDArray out2s_ind = Nd4j.create(out2_b, new int[]{(int)ndImage3HW.size(0), 4});
    INDArray out3s_ind = Nd4j.create(out2_c, new int[]{(int)ndImage3HW.size(0), 2});

    List<INDArray> outs2 = new ArrayList<>();
    outs2.add(out1s_ind);
    outs2.add(out2s_ind);
    outs2.add(out3s_ind);

    return outs2;
  }

  public void Run() throws IOException {
    String imgPath =
        "/home/dzly/projects/countr_face_recognition/faceclient/src/test/resources/multi-face.jpg";

    try (InputStream imageInputStream = new FileInputStream(imgPath)) {
      BufferedImage inputImage = ImageIO.read(imageInputStream);
      Java2DNativeImageLoader imageLoader = new Java2DNativeImageLoader();
      INDArray ndImage3HW =
          imageLoader.asMatrix(inputImage).get(point(0), interval(0, 3), all(), all());
      float[] ints = ndImage3HW.data().asFloat();
      Shape inputShape2 =
          new Shape(new int[] {1, 4, inputImage.getWidth(), inputImage.getHeight()});
      NDArray img = new NDArray(ints, inputShape2, this.ctx.get(0));
      img = img.reshape(new int[] {1, 3, 25, 25});

      List<NDArray> imgs = new ArrayList<NDArray>();
      imgs.add(img);

      List<NDArray> outs = proposeNetGraphRunner.predictWithNDArray(imgs);
    }
  }

  private BufferedImage imageFromINDArray(INDArray array) {
    long[] shape = array.shape();
 
    long height = shape[2];
    long width = shape[3];
    BufferedImage image = new BufferedImage((int)width, (int)height, BufferedImage.TYPE_INT_RGB);
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int red = array.getInt(0, 2, y, x);
            int green = array.getInt(0, 1, y, x);
            int blue = array.getInt(0, 0, y, x);
 
            //handle out of bounds pixel values
            red = Math.min(red, 255);
            green = Math.min(green, 255);
            blue = Math.min(blue, 255);
 
            red = Math.max(red, 0);
            green = Math.max(green, 0);
            blue = Math.max(blue, 0);
            // image.setRGB(x, y, new Color(red, green, blue).getRGB());
        }
    }
    return image;
  
  }
}